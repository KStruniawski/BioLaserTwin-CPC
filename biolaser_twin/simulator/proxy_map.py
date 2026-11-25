"""Truth-side proxy mapping from dimensionless latent variables z to physical units.

Strictly isolated inside simulator/ to prevent inverse-crime sharing with R_phys.
"""

from dataclasses import dataclass
from .photophysics import TruthSampleState


@dataclass(frozen=True)
class LatentProxyVector:
    """Dimensionless latent coordinates for synthetic population sampling."""

    z1_microviscosity: float  # normalized [0, 1] -> [1.0, 50.0] cP
    z2_bound_fraction: float  # [0, 1]
    z3_refractive_index: float  # [1.333, 1.380]
    z4_scattering_cm1: float  # [0.05, 15.0] cm^-1
    z5_cavity_um: float  # [10.0, 50.0] um
    tht_conc_mm: float = 10.0


def map_latents_to_truth_state(z: LatentProxyVector) -> TruthSampleState:
    """Map latent coordinates to physical TruthSampleState using truth-only scaling."""
    visc_cp = 1.0 + 49.0 * (max(0.0, min(1.0, z.z1_microviscosity)) ** 1.4)
    return TruthSampleState(
        refractive_index=z.z3_refractive_index,
        cavity_length_um=z.z5_cavity_um,
        microviscosity_cp=visc_cp,
        bound_fraction=max(0.0, min(1.0, z.z2_bound_fraction)),
        scattering_loss_cm1=max(0.01, z.z4_scattering_cm1),
        tht_conc_mm=max(0.0, z.tht_conc_mm),
    )
