"""Independent high-fidelity truth generators (isolated from solvers)."""

from .fp_cavity import FabryPerotConfig, FabryPerotTruthSimulator, ObservationBundle
from .photophysics import (
    KineticTrajectory,
    TruthSampleState,
    effective_gain_and_emission,
    simulate_secondary_nucleation_kinetics,
    spectral_reabsorption_cm1,
    wavelength_dependent_scattering,
)
from .proxy_map import LatentProxyVector, map_latents_to_truth_state

__all__ = [
    "FabryPerotConfig",
    "FabryPerotTruthSimulator",
    "KineticTrajectory",
    "LatentProxyVector",
    "ObservationBundle",
    "TruthSampleState",
    "effective_gain_and_emission",
    "map_latents_to_truth_state",
    "simulate_secondary_nucleation_kinetics",
    "spectral_reabsorption_cm1",
    "wavelength_dependent_scattering",
]
