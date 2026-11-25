"""Independent truth photophysics and secondary-nucleation kinetics (never imported by solvers)."""

from dataclasses import dataclass
import warnings
import numpy as np


@dataclass(frozen=True)
class TruthSampleState:
    """True latent physical state of a simulated sample (hidden from solvers at test time)."""

    refractive_index: float
    cavity_length_um: float
    microviscosity_cp: float
    bound_fraction: float
    scattering_loss_cm1: float
    tht_conc_mm: float
    temperature_k: float = 298.15
    bound_tht_um: float | None = None
    forster_hoffmann_x: float = 0.55
    scattering_wavelengths_nm: tuple[float, ...] | None = None
    scattering_loss_spectrum_cm1: tuple[float, ...] | None = None


@dataclass(frozen=True)
class KineticTrajectory:
    """Mass-conserving aggregation trajectory from secondary nucleation & fragmentation."""

    times_min: np.ndarray
    monomer_frac: np.ndarray
    oligomer_frac: np.ndarray
    fibril_frac: np.ndarray


def simulate_secondary_nucleation_kinetics(
    times_min: np.ndarray,
    k_n: float = 1e-4,
    k_plus: float = 0.04,
    k_2: float = 0.08,
    k_off: float = 0.015,
) -> KineticTrajectory:
    """Simulate mass-conserving monomer -> oligomer -> fibril kinetics on arbitrary time grids."""
    warnings.warn(
        "simulate_secondary_nucleation_kinetics is deprecated in BioLaserTwin v0.2; "
        "use biolaser_twin.simulator.kinetics.solve_pbe instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    n_steps = len(times_min)
    m = np.zeros(n_steps, dtype=float)
    o = np.zeros(n_steps, dtype=float)
    f = np.zeros(n_steps, dtype=float)
    m[0] = 1.0

    for idx in range(1, n_steps):
        dt = max(0.0, float(times_min[idx] - times_min[idx - 1]))
        substeps = max(1, int(np.ceil(dt / 0.1))) if dt > 0.0 else 1
        h = dt / substeps
        mi, oi, fi = m[idx - 1], o[idx - 1], f[idx - 1]
        for _ in range(substeps):
            dm_to_o = (k_n * (mi**2) + k_2 * mi * fi) * h
            do_to_f = (k_plus * oi * (mi + 0.1) + k_off * oi) * h
            dm_to_o = min(dm_to_o, mi)
            do_to_f = min(do_to_f, oi + dm_to_o)
            mi = max(0.0, mi - dm_to_o)
            oi = max(0.0, oi + dm_to_o - do_to_f)
            fi = max(0.0, 1.0 - mi - oi)
        m[idx], o[idx], f[idx] = mi, oi, fi

    return KineticTrajectory(times_min=times_min, monomer_frac=m, oligomer_frac=o, fibril_frac=f)


def wavelength_dependent_scattering(
    scattering_loss_cm1: float,
    wavelengths_nm: np.ndarray,
    mie_exponent: float = 1.8,
) -> np.ndarray:
    """Compute Mie-like wavelength-dependent scattering loss alpha(lambda) in cm^-1."""
    return scattering_loss_cm1 * ((485.0 / wavelengths_nm) ** mie_exponent)


def spectral_reabsorption_cm1(
    tht_conc_mm: float,
    bound_fraction: float,
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """Compute blue-edge ThT self-absorption tail sigma_a(lambda) in cm^-1."""
    abs_center = 415.0 + 35.0 * bound_fraction
    return 0.18 * tht_conc_mm * np.exp(-0.5 * ((wavelengths_nm - abs_center) / 14.0) ** 2)


def effective_gain_and_emission(
    state: TruthSampleState,
    excitation_nm: float,
    pump_energy_uj: float,
    wavelengths_nm: np.ndarray,
    mirror_reflectivity: float | np.ndarray = 0.988,
    bleach_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute spectral gain g(lambda), total loss alpha(lambda), and true threshold E_th (uJ).

    Uses configured mirror_reflectivity (no hard-coded R) and smooth gain clamping.
    """
    eff_free = float(np.exp(-0.5 * ((excitation_nm - 408.0) / 18.0) ** 2))
    eff_bound = float(np.exp(-0.5 * ((excitation_nm - 442.0) / 16.0) ** 2))

    visc_factor = float((max(state.microviscosity_cp, 0.5) / 1.0) ** state.forster_hoffmann_x)
    c_eff = max(0.0, state.tht_conc_mm * bleach_factor)
    inner_filter = float(np.exp(-0.025 * c_eff))

    if state.bound_tht_um is None:
        free_contrib = (1.0 - state.bound_fraction) * eff_free * 0.12 * visc_factor
        bound_contrib = state.bound_fraction * eff_bound * 1.0
        total_exc = c_eff * (free_contrib + bound_contrib) * inner_filter
    else:
        # mM-ThT regime: gain driven by absolute bound ThT (uM) + weak free-ThT rotor term
        c_bound_eff = max(0.0, state.bound_tht_um * bleach_factor)
        c_free_mm = max(0.0, c_eff - 1e-3 * c_bound_eff)
        bound_gain_drive = 0.55 * c_bound_eff * eff_bound
        free_gain_drive = 0.08 * c_free_mm * eff_free * visc_factor
        total_exc = (bound_gain_drive + free_gain_drive) * inner_filter

    lambda_center = 478.0 + 8.0 * state.bound_fraction
    envelope = np.exp(-0.5 * ((wavelengths_nm - lambda_center) / 11.0) ** 2)

    l_cm = max(state.cavity_length_um * 1e-4, 1e-5)
    if np.ndim(mirror_reflectivity) > 0:
        r_clamped = np.clip(np.asarray(mirror_reflectivity, dtype=float), 0.5, 0.9999)
    else:
        r_clamped = min(max(float(mirror_reflectivity), 0.5), 0.9999)
    mirror_loss_cm1 = -np.log(r_clamped**2) / (2.0 * l_cm)

    alpha_scat = (
        np.interp(wavelengths_nm, state.scattering_wavelengths_nm, state.scattering_loss_spectrum_cm1)
        if (state.scattering_wavelengths_nm is not None and state.scattering_loss_spectrum_cm1 is not None)
        else wavelength_dependent_scattering(state.scattering_loss_cm1, wavelengths_nm)
    )
    alpha_reabs = spectral_reabsorption_cm1(c_eff, state.bound_fraction, wavelengths_nm)
    total_loss_lam = mirror_loss_cm1 + alpha_scat + alpha_reabs

    # Unsaturated small-signal peak gain per uJ of pump
    small_sig_gain_per_uj = 0.045 * total_exc
    min_loss = float(np.min(total_loss_lam))
    e_th_uj = min_loss / max(small_sig_gain_per_uj, 1e-7)

    # Smooth gain clamping above threshold (softplus transition avoids discontinuous FIM)
    raw_gain_lam = small_sig_gain_per_uj * pump_energy_uj * envelope
    # Clamp net round-trip gain below unity: G_eff <= 0.995 * total_loss_lam
    clamped_gain_lam = total_loss_lam * np.tanh(raw_gain_lam / np.maximum(total_loss_lam, 1e-4)) * 0.995

    # Smooth ASE + stimulated source intensity (overflow-safe softplus)
    ratio = pump_energy_uj / max(e_th_uj, 1e-3)
    smooth_above = np.logaddexp(0.0, 4.0 * (ratio - 1.0)) / 4.0
    source_spec = (0.04 * ratio + 1.8 * smooth_above) * envelope

    net_gain = clamped_gain_lam - total_loss_lam
    if np.ndim(mirror_reflectivity) == 0:
        net_gain = np.maximum(net_gain, -20.0)
    return source_spec, net_gain, float(e_th_uj)
