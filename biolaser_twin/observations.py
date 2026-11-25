"""Unified named observation schema, masking, and empirical noise covariance estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from biolaser_twin.extractor.frame_features import SweepFeatures
from biolaser_twin.identifiability.fisher import DesignConfig

if TYPE_CHECKING:
    from biolaser_twin.constants import NominalOpticalConstants


OBSERVABLE_NAMES: tuple[str, ...] = (
    "fsr_nm",
    "e_th_405_uj",
    "e_th_440_uj",
    "com_405_nm",
    "com_440_nm",
    "scat_norm",
)


@dataclass(frozen=True)
class ObservationRecord:
    """Named observable vector with explicit missingness/censoring mask and design metadata."""

    values: np.ndarray  # shape: (6,)
    mask: np.ndarray  # bool shape: (6,) — True where valid/observed
    design: DesignConfig
    lasing_405: bool
    lasing_440: bool


def build_observation_record(
    feats_405: SweepFeatures | None,
    feats_440: SweepFeatures | None,
    design: DesignConfig,
    max_pump_uj: float = 45.0,
) -> ObservationRecord:
    """Assemble the fixed-schema 6D observable vector from dual/single-excitation features."""
    vals = np.zeros(len(OBSERVABLE_NAMES), dtype=float)
    mask = np.zeros(len(OBSERVABLE_NAMES), dtype=bool)

    # FSR from whichever excitation resolved modal spacing
    fsr_candidates = [
        f.mean_fsr_nm
        for f in (feats_440, feats_405)
        if f is not None and f.mean_fsr_nm is not None and np.isfinite(f.mean_fsr_nm)
    ]
    if fsr_candidates:
        vals[0] = float(np.median(fsr_candidates))
        mask[0] = True

    if feats_405 is not None:
        if feats_405.lasing_detected and feats_405.threshold_energy_uj is not None:
            vals[1] = float(feats_405.threshold_energy_uj)
            mask[1] = True
        vals[3] = float(feats_405.center_of_mass_nm)
        mask[3] = True

    if feats_440 is not None:
        if feats_440.lasing_detected and feats_440.threshold_energy_uj is not None:
            vals[2] = float(feats_440.threshold_energy_uj)
            mask[2] = True
        vals[4] = float(feats_440.center_of_mass_nm)
        mask[4] = True

    # Normalized off-peak scattering observable (dimensionless)
    scat_vals = [
        max(0.0, float(f.off_peak_baseline)) / max(max_pump_uj, 1.0)
        for f in (feats_405, feats_440)
        if f is not None and np.isfinite(f.off_peak_baseline)
    ]
    if scat_vals:
        vals[5] = float(np.mean(scat_vals))
        mask[5] = True

    return ObservationRecord(
        values=vals,
        mask=mask,
        design=design,
        lasing_405=bool(feats_405 and feats_405.lasing_detected),
        lasing_440=bool(feats_440 and feats_440.lasing_detected),
    )


def estimate_noise_cov(records: list[ObservationRecord], min_std: float = 1e-3) -> np.ndarray:
    """Estimate diagonal empirical observable covariance Sigma from repeated dev sweeps."""
    n_obs = len(OBSERVABLE_NAMES)
    if not records:
        return np.diag(np.array([0.08, 2.0, 2.0, 0.3, 0.3, 0.02]) ** 2)

    mat = np.stack([r.values for r in records], axis=0)
    masks = np.stack([r.mask for r in records], axis=0)
    stds = np.zeros(n_obs, dtype=float)
    default_stds = np.array([0.08, 2.0, 2.0, 0.3, 0.3, 0.02], dtype=float)
    for j in range(n_obs):
        valid = mat[masks[:, j], j]
        if len(valid) >= 3:
            stds[j] = max(min_std, float(np.std(valid, ddof=1)))
        else:
            stds[j] = default_stds[j]
    return np.diag(stds**2)


def calibrate_nominal_constants_from_dev(
    dev_records: list[ObservationRecord],
    dev_latents: np.ndarray,
    base_constants: "NominalOpticalConstants | None" = None,
) -> "NominalOpticalConstants":
    """Calibrate gain_scale_kappa and scat_scale_nominal on Level 0 / dev sweeps only."""
    from biolaser_twin.constants import NominalOpticalConstants
    from biolaser_twin.identifiability.fisher import reduced_forward_observables

    c = base_constants or NominalOpticalConstants()
    if not dev_records:
        return c

    # Fit multiplicative scale kappa on E_th (E_th ~ 1/kappa -> kappa_fit = median(E_nom / E_obs))
    kappa_ratios = []
    scat_ratios = []
    for rec, z in zip(dev_records, dev_latents):
        nom = reduced_forward_observables(z, rec.design, c)
        for idx in (1, 2):
            if rec.mask[idx] and rec.values[idx] > 1e-3:
                kappa_ratios.append(nom[idx] / rec.values[idx])
        if rec.mask[5] and z[3] > 1e-3 and rec.values[5] > 0.0:
            scat_ratios.append(rec.values[5] / z[3])

    new_kappa = c.gain_scale_kappa * float(np.median(kappa_ratios)) if kappa_ratios else c.gain_scale_kappa
    new_scat = float(np.median(scat_ratios)) if scat_ratios else c.scat_scale_nominal
    return NominalOpticalConstants(
        n_water_nominal=c.n_water_nominal,
        mirror_reflectivity_nominal=c.mirror_reflectivity_nominal,
        tht_molar_absorptivity_412=c.tht_molar_absorptivity_412,
        spec_resolution_nm_nominal=c.spec_resolution_nm_nominal,
        nominal_tht_conc_mm=c.nominal_tht_conc_mm,
        gain_scale_kappa=new_kappa,
        scat_scale_nominal=max(1e-5, new_scat),
    )


@dataclass(frozen=True)
class KineticAdapterCalibration:
    """Affine calibration mapping raw SweepFeatures -> nominal (e_th, slope_eff, lambda_shift, [bound_eq])."""

    eth_scale: float = 1.0
    eth_offset: float = 0.0
    slope_scale: float = 1e-4
    slope_offset: float = 0.0
    lambda_scale: float = 1.0
    lambda_ref_nm: float = 478.0


def extract_kinetic_observable_series(
    sweep_features_list: list["SweepFeatures"],
    adapter: KineticAdapterCalibration | None = None,
    max_pump_uj: float = 45.0,
    fluo_mass_channel: np.ndarray | None = None,
    observable_set: str = "full_lasing",
) -> tuple[np.ndarray, np.ndarray]:
    """Adapt time-resolved SweepFeatures (+ optional fluorescence mass) to nominal observable schema."""
    cal = adapter or KineticAdapterCalibration()
    n_t = len(sweep_features_list)
    e_th = np.zeros(n_t, dtype=float)
    v_eth = np.ones(n_t, dtype=bool)
    slope = np.zeros(n_t, dtype=float)
    lam = np.zeros(n_t, dtype=float)

    for idx, f in enumerate(sweep_features_list):
        if f.threshold_energy_uj is not None and np.isfinite(f.threshold_energy_uj):
            e_th[idx] = float(f.threshold_energy_uj) * cal.eth_scale + cal.eth_offset
        else:
            e_th[idx] = max_pump_uj * cal.eth_scale + cal.eth_offset
            v_eth[idx] = False
        slope[idx] = max(0.0, float(f.slope_efficiency) * cal.slope_scale + cal.slope_offset)
        lam[idx] = (float(f.center_of_mass_nm) - cal.lambda_ref_nm) * cal.lambda_scale

    if observable_set == "e_th":
        return e_th[:, None], v_eth[:, None]
    if observable_set == "mass_only":
        if fluo_mass_channel is None:
            raise ValueError("fluo_mass_channel is required for observable_set='mass_only'.")
        return fluo_mass_channel[:, None], np.ones((n_t, 1), dtype=bool)
    if observable_set == "full_lasing":
        return np.column_stack([e_th, slope, lam]), np.column_stack([v_eth, np.ones(n_t, dtype=bool), np.ones(n_t, dtype=bool)])
    if observable_set == "lasing_plus_mass":
        if fluo_mass_channel is None:
            raise ValueError("fluo_mass_channel is required for observable_set='lasing_plus_mass'.")
        return np.column_stack([e_th, slope, lam, fluo_mass_channel]), np.column_stack(
            [v_eth, np.ones(n_t, dtype=bool), np.ones(n_t, dtype=bool), np.ones(n_t, dtype=bool)]
        )
    raise ValueError(f"Unknown observable_set: {observable_set}")


def calibrate_kinetic_adapter_from_dev(
    sweep_features_list: list["SweepFeatures"],
    y_nominal_dev: np.ndarray,
    max_pump_uj: float = 45.0,
) -> KineticAdapterCalibration:
    """Fit linear/affine channel calibration on dev sweeps to align simulator features to nominal map."""
    raw_eth, valid_eth, raw_slope, raw_com = [], [], [], []
    for idx, f in enumerate(sweep_features_list):
        if f.threshold_energy_uj is not None and np.isfinite(f.threshold_energy_uj):
            raw_eth.append(float(f.threshold_energy_uj))
            valid_eth.append(idx)
        raw_slope.append(float(f.slope_efficiency))
        raw_com.append(float(f.center_of_mass_nm))

    # Calibrate E_th scale+offset on uncensored frames (enforce positive physical slope)
    if len(valid_eth) >= 2:
        x_e = np.asarray(raw_eth)
        y_e = y_nominal_dev[valid_eth, 0]
        if np.std(x_e) > 1e-6 and np.std(y_e) > 1e-6:
            eth_scale, eth_offset = np.polyfit(x_e, y_e, 1)
        else:
            eth_scale, eth_offset = 0.0, 0.0
        if eth_scale <= 0.05:
            eth_scale = float(np.median(y_e) / max(np.median(x_e), 1e-6))
            eth_offset = 0.0
    else:
        eth_scale, eth_offset = 1.0, 0.0

    x_s = np.asarray(raw_slope)
    y_s = y_nominal_dev[:, 1]
    if np.std(x_s) > 1e-6:
        slope_scale, slope_offset = np.polyfit(x_s, y_s, 1)
    else:
        slope_scale, slope_offset = 1e-4, 0.0

    # Anchor lambda_ref_nm at t=0 pre-aggregation baseline so lambda_shift(t=0) ~ 0
    x_l = np.asarray(raw_com)
    y_l = y_nominal_dev[:, 2]
    lam_ref = float(x_l[0]) if len(x_l) > 0 else 485.0
    dx_l = x_l - lam_ref
    denom_l = float(np.dot(dx_l, dx_l))
    if denom_l > 1e-6:
        raw_lam_scale = float(np.dot(dx_l, y_l) / denom_l)
        if abs(raw_lam_scale) < 0.1:
            lam_scale = float(np.sign(raw_lam_scale) * 0.5) if raw_lam_scale != 0.0 else 1.0
        else:
            lam_scale = float(np.clip(raw_lam_scale, -10.0, 10.0))
    else:
        lam_scale = 1.0

    return KineticAdapterCalibration(
        eth_scale=float(max(1e-4, eth_scale)),
        eth_offset=float(eth_offset),
        slope_scale=float(max(1e-7, slope_scale)),
        slope_offset=float(slope_offset),
        lambda_scale=float(lam_scale),
        lambda_ref_nm=float(lam_ref),
    )


def estimate_kinetic_noise_std(
    replicate_series: list[np.ndarray],
    min_floors: tuple[float, ...] = (0.3, 0.02, 0.05, 0.05),
) -> np.ndarray:
    """Estimate empirical per-time, per-channel noise std across repeated simulator sweeps."""
    stack = np.stack(replicate_series, axis=0)
    n_channels = stack.shape[2]
    if len(min_floors) < n_channels:
        padded = list(min_floors) + [min_floors[-1]] * (n_channels - len(min_floors))
        floors = np.asarray(padded, dtype=float)
    else:
        floors = np.asarray(min_floors[:n_channels], dtype=float)
    if stack.shape[0] < 2:
        return np.broadcast_to(floors, stack.shape[1:]).copy()
    emp_std = np.std(stack, axis=0, ddof=1)
    return np.maximum(emp_std, floors[None, :])
