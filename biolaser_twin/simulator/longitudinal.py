"""Persistent longitudinal illumination history (photobleaching refill & thermal relaxation)."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LongitudinalParams:
    """Parameters governing diffusive dye refill and thermal relaxation between sweeps."""

    tau_rec_base_min: float = 15.0  # diffusive ThT refill time at 1 cP (min)
    tau_th_min: float = 5.0  # cavity thermal relaxation time (min)
    bleach_beta_per_uj: float = 0.0005  # photobleaching sensitivity (uJ^-1)
    heating_chi_k_per_uj: float = 0.004  # transient heating sensitivity (K / uJ)
    eta_ref_cp: float = 1.0  # reference solvent viscosity (cP)


@dataclass(frozen=True)
class SampleHistory:
    """Persistent cumulative optical dose and thermal state across longitudinal sweeps."""

    dose_uj: float = 0.0
    delta_temp_k: float = 0.0
    bleach_factor: float = 1.0
    last_time_min: float = 0.0


def diffusive_refill_time_min(
    microviscosity_cp: float,
    params: LongitudinalParams | None = None,
) -> float:
    """Compute viscosity-dependent diffusive dye recovery time tau_rec(eta_mu) in minutes."""
    p = params or LongitudinalParams()
    return p.tau_rec_base_min * (max(microviscosity_cp, 0.2) / max(p.eta_ref_cp, 1e-6))


def relax_sample_history(
    history: SampleHistory,
    current_time_min: float,
    microviscosity_cp: float,
    params: LongitudinalParams | None = None,
) -> SampleHistory:
    """Relax dose and temperature offset across elapsed interval dt_min = current_time - last_time."""
    p = params or LongitudinalParams()
    dt_min = max(0.0, float(current_time_min) - float(history.last_time_min))
    tau_rec = diffusive_refill_time_min(microviscosity_cp, p)
    d_relaxed = history.dose_uj * float(np.exp(-dt_min / max(tau_rec, 1e-9)))
    dt_relaxed = history.delta_temp_k * float(np.exp(-dt_min / max(p.tau_th_min, 1e-9)))
    bleach = float(np.exp(-p.bleach_beta_per_uj * d_relaxed))
    return SampleHistory(
        dose_uj=d_relaxed,
        delta_temp_k=dt_relaxed,
        bleach_factor=bleach,
        last_time_min=float(current_time_min),
    )


def update_sample_history(
    history: SampleHistory,
    sweep_dose_uj: float,
    current_time_min: float,
    microviscosity_cp: float,
    params: LongitudinalParams | None = None,
) -> SampleHistory:
    """Relax history to current_time_min and add cumulative pump dose from a sweep."""
    p = params or LongitudinalParams()
    relaxed = relax_sample_history(history, current_time_min, microviscosity_cp, p)
    d_new = relaxed.dose_uj + max(0.0, float(sweep_dose_uj))
    dt_new = relaxed.delta_temp_k + p.heating_chi_k_per_uj * max(0.0, float(sweep_dose_uj))
    bleach_new = float(np.exp(-p.bleach_beta_per_uj * d_new))
    return SampleHistory(
        dose_uj=d_new,
        delta_temp_k=dt_new,
        bleach_factor=bleach_new,
        last_time_min=float(current_time_min),
    )


def make_thermal_history_temperature_fn(
    history_events: list[SampleHistory],
    base_temp_k: float = 298.15,
    params: LongitudinalParams | None = None,
):
    """Build a piecewise-relaxing T(t) callable from longitudinal heating events for M_w Arrhenius coupling."""
    p = params or LongitudinalParams()
    events = sorted(history_events, key=lambda h: h.last_time_min)

    def temp_fn(t_min: float) -> float:
        t_val = float(t_min)
        dt_k = 0.0
        for ev in events:
            if ev.last_time_min <= t_val:
                dt_k = ev.delta_temp_k * float(np.exp(-(t_val - ev.last_time_min) / max(p.tau_th_min, 1e-9)))
            else:
                break
        return base_temp_k + dt_k

    return temp_fn
