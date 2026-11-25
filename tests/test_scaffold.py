"""Verification tests for the BioLaserTwin package scaffold."""

import numpy as np

from biolaser_twin.extractor import extract_spectral_features
from biolaser_twin.identifiability import compute_fim, cramer_rao_sd
from biolaser_twin.simulator import FabryPerotConfig, FabryPerotTruthSimulator
from biolaser_twin.solvers import PhotonicPIELMSolver


def test_airy_fsr_matches_passive_theory() -> None:
    """Verify Airy peak spacing matches theoretical FSR = lambda^2 / (2 n L)."""
    cfg = FabryPerotConfig(cavity_length_um=30.0, refractive_index=1.345, mirror_reflectivity=0.95)
    sim = FabryPerotTruthSimulator(cfg)
    wl = sim.wavelength_grid_nm()
    spectrum = sim.airy_transmission(wl)
    feats = extract_spectral_features(wl, spectrum)

    expected_fsr = sim.passive_fsr_nm(center_lambda_nm=485.0)
    assert feats.estimated_fsr_nm is not None
    assert abs(feats.estimated_fsr_nm - expected_fsr) < 0.15


def test_fim_and_cramer_rao_shapes() -> None:
    """Verify FIM and CRB standard deviations are finite and positive."""
    jac = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    cov_inv = np.eye(3)
    fim = compute_fim(jac, cov_inv)
    sd = cramer_rao_sd(fim)
    assert sd.shape == (2,)
    assert np.all(sd > 0.0)


def test_photonic_pielm_fit_predict() -> None:
    """Verify PhotonicPIELMSolver fits and predicts with expected shapes."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6))
    y = rng.normal(size=(40, 2))
    solver = PhotonicPIELMSolver(hidden_dim=32, seed=0).fit(x, y)
    pred = solver.predict(x)
    assert pred.shape == (40, 2)
