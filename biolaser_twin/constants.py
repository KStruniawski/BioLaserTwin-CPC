"""Separate nominal constants registry for solvers/identifiability vs. truth generator."""

from dataclasses import dataclass
import numpy as np


LATENT_NAMES: tuple[str, ...] = ("nL_um", "z1_visc", "z2_bound", "z4_scat")
LATENT_PRIOR_SCALES: np.ndarray = np.array([35.0, 0.30, 0.30, 4.0], dtype=float)


@dataclass(frozen=True)
class NominalOpticalConstants:
    """Nominal optical and dye constants exposed to inverse solvers and FIM/DoE.

    Strictly separated from truth-generator parameter draws to prevent inverse crime.
    gain_scale_kappa is calibrated strictly on Level 0 / dev data only.
    """

    n_water_nominal: float = 1.333
    mirror_reflectivity_nominal: float = 0.99
    tht_molar_absorptivity_412: float = 36000.0  # M^-1 cm^-1
    spec_resolution_nm_nominal: float = 0.10
    nominal_tht_conc_mm: float = 10.0
    gain_scale_kappa: float = 1.0
    scat_scale_nominal: float = 0.02
