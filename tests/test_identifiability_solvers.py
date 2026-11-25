"""Unit tests for identifiability diagnostics, conformal calibration, and solver isolation."""

import importlib
import sys
import numpy as np

from biolaser_twin.calibration import SplitConformalCalibrator
from biolaser_twin.constants import LATENT_PRIOR_SCALES, NominalOpticalConstants
from biolaser_twin.identifiability import (
    DesignConfig,
    check_structural_n_l_rank,
    cramer_rao_sd,
    reduced_forward_observables,
    select_optimal_designs,
)
from biolaser_twin.metrics import (
    gaussian_crps,
    gaussian_nll,
    interval_coverage_and_width,
    normalized_rmse,
)
from biolaser_twin.observations import (
    ObservationRecord,
    build_observation_record,
    calibrate_nominal_constants_from_dev,
    estimate_noise_cov,
)
from biolaser_twin.solvers import (
    ClassicalNLSSolver,
    FrameCNNSolver,
    FrameViTSolver,
    HybridCNNFeaturesSolver,
    PhotonicPIELMSolver,
    PlainELMSolver,
    TabularMLPSolver,
)


def test_structural_n_l_rank_deficiency_and_eigen_crb() -> None:
    """Verify S1 structural n vs L rank deficiency (<5) and inf CRB on null directions."""
    rank, full_dim = check_structural_n_l_rank()
    assert rank < full_dim

    # Rank-deficient FIM with zero sensitivity on coordinate 1 must return np.inf
    fim = np.diag([10.0, 0.0, 5.0, 2.0])
    crb = cramer_rao_sd(fim)
    assert np.isinf(crb[1])
    assert np.isfinite(crb[0]) and np.isfinite(crb[2]) and np.isfinite(crb[3])


def test_classical_nls_recovers_clean_nominal_latents_within_1pct() -> None:
    """Verify whitened ClassicalNLSSolver recovers noise-free nominal latents cleanly."""
    design = DesignConfig(nominal_cavity_um=30.0, excitations_nm=(405.0, 440.0))
    consts = NominalOpticalConstants()
    true_latents = np.array([50.0, 0.65, 0.30, 4.5], dtype=float)
    obs = reduced_forward_observables(true_latents, design, consts)
    noise_cov = estimate_noise_cov([])

    nls = ClassicalNLSSolver(constants=consts, noise_cov=noise_cov, prior_std=np.full(4, 1e3))
    pred_mean, pred_std = nls.predict_with_uncertainty(obs[None, :], design=design)
    rel_err = np.abs(pred_mean[0] - true_latents) / np.maximum(true_latents, 0.1)
    assert np.all(rel_err < 0.01)
    assert np.all(pred_std > 0.0)


def test_photonic_pielm_coupled_gn_and_unlabeled_collocation() -> None:
    """Verify PhotonicPIELM GN iterations change output and support arbitrary collocation sizes."""
    rng = np.random.default_rng(17)
    design = DesignConfig(nominal_cavity_um=30.0, excitations_nm=(405.0, 440.0))
    consts = NominalOpticalConstants()
    noise_cov = estimate_noise_cov([])

    def _gen_set(n: int) -> tuple[np.ndarray, np.ndarray, list[ObservationRecord]]:
        latents = np.column_stack(
            [
                rng.uniform(25.0, 60.0, size=n),
                rng.uniform(0.1, 0.9, size=n),
                rng.uniform(0.1, 0.9, size=n),
                rng.uniform(0.5, 8.0, size=n),
            ]
        )
        obs = np.stack([reduced_forward_observables(z, design, consts) for z in latents], axis=0)
        obs += rng.normal(0.0, np.sqrt(np.diag(noise_cov)) * 0.1, size=obs.shape)
        recs = [
            ObservationRecord(values=o, mask=np.ones(6, dtype=bool), design=design, lasing_405=True, lasing_440=True)
            for o in obs
        ]
        return obs, latents, recs

    x_tr, y_tr, _ = _gen_set(24)
    x_col, _, recs_col = _gen_set(37)  # different row count from x_tr!
    x_te, y_te, _ = _gen_set(20)

    p0 = PhotonicPIELMSolver(hidden_dim=32, phys_lambda=0.0, n_gn_iters=0, seed=1).fit(x_tr, y_tr)
    p1 = PhotonicPIELMSolver(hidden_dim=32, phys_lambda=0.5, n_gn_iters=1, noise_cov=noise_cov, seed=1).fit(
        x_tr, y_tr, collocation_features=x_col, collocation_records=recs_col
    )
    p4 = PhotonicPIELMSolver(hidden_dim=32, phys_lambda=0.5, n_gn_iters=4, noise_cov=noise_cov, seed=1).fit(
        x_tr, y_tr, collocation_features=x_col, collocation_records=recs_col
    )

    m0, _ = p0.predict_with_uncertainty(x_te)
    m1, _ = p1.predict_with_uncertainty(x_te)
    m4, s4 = p4.predict_with_uncertainty(x_te)
    assert np.max(np.abs(m4 - m1)) > 1e-6  # GN loop actually updates weights
    assert np.mean(normalized_rmse(y_te, m4, LATENT_PRIOR_SCALES)) < np.mean(
        normalized_rmse(y_te, m0, LATENT_PRIOR_SCALES)
    )
    assert np.all(s4 > 0.0)


def test_budget_matched_doe_and_baselines_with_conformal() -> None:
    """Verify Bayesian DoE ranking, MLP/CNN/ViT solvers, per-latent metrics, and conformal calibration."""
    ref_grid = np.array([[35.0, 0.3, 0.5, 2.0], [50.0, 0.7, 0.7, 5.0]])
    cov = estimate_noise_cov([])
    c_single = DesignConfig(nominal_cavity_um=30.0, spec_psf_fwhm_nm=0.10, excitations_nm=(440.0,))
    c_dual = DesignConfig(nominal_cavity_um=30.0, spec_psf_fwhm_nm=0.10, excitations_nm=(405.0, 440.0))
    ranked = select_optimal_designs([c_single, c_dual], ref_grid, noise_cov=cov, measurement_budget_shots=2)
    assert len(ranked) == 2 and np.isfinite(ranked[0].a_opt_norm_trace)

    rng = np.random.default_rng(5)
    x_tr = rng.normal(size=(24, 6))
    y_tr = rng.uniform(0.1, 1.0, size=(24, 4))
    x_va = rng.normal(size=(12, 6))
    y_va = rng.uniform(0.1, 1.0, size=(12, 4))

    mlp = TabularMLPSolver(hidden_dim=16, epochs=10, n_ensemble=2, seed=1).fit(x_tr, y_tr)
    vm, vs = mlp.predict_with_uncertainty(x_va)
    cal = SplitConformalCalibrator().fit(y_va, vm, vs)
    vs_cal = cal.calibrate_std(vs)
    cov_lat, width_lat = interval_coverage_and_width(y_va, vm, vs_cal)
    assert cov_lat.shape == (4,) and width_lat.shape == (4,)
    assert gaussian_nll(y_va, vm, vs_cal).shape == (4,)
    assert gaussian_crps(y_va, vm, vs_cal).shape == (4,)

    frames = rng.uniform(0.0, 50.0, size=(16, 4, 16, 64))
    cnn = FrameCNNSolver(epochs=4, seed=1).fit(frames, y_tr[:16])
    cm, cs = cnn.predict_with_uncertainty(frames[:4])
    assert cm.shape == (4, 4) and np.all(cs > 0.0)

    vit = FrameViTSolver(patch_size=8, epochs=4, seed=1).fit(frames, y_tr[:16])
    vm1, vs1 = vit.predict_with_uncertainty(frames[:4])
    vm2, vs2 = vit.predict_with_uncertainty(frames[:4])
    assert np.allclose(vm1, vm2, atol=0.0) and np.allclose(vs1, vs2, atol=0.0)

    hyb = HybridCNNFeaturesSolver(epochs=4, seed=1).fit(frames, x_tr[:16], y_tr[:16])
    hm, hs = hyb.predict_with_uncertainty(frames[:4], x_tr[:4])
    assert hm.shape == (4, 4) and np.all(hs > 0.0)


def test_masked_collocation_has_zero_drift_and_budget_sensitivity() -> None:
    """Verify masked collocation produces zero GN drift and DoE distinguishes budgets 0, 1, 2."""
    rng = np.random.default_rng(99)
    design = DesignConfig(nominal_cavity_um=30.0, excitations_nm=(405.0, 440.0))
    x = rng.normal(size=(20, 6))
    y = rng.uniform(0.2, 1.0, size=(20, 4))
    masked_recs = [
        ObservationRecord(values=np.zeros(6), mask=np.zeros(6, dtype=bool), design=design, lasing_405=False, lasing_440=False)
        for _ in range(15)
    ]
    plain = PlainELMSolver(hidden_dim=24, seed=3).fit(x, y)
    pielm_masked = PhotonicPIELMSolver(hidden_dim=24, phys_lambda=1.0, n_gn_iters=5, seed=3).fit(
        x, y, collocation_features=x[:15], collocation_records=masked_recs
    )
    m_plain, _ = plain.predict_with_uncertainty(x)
    m_masked, _ = pielm_masked.predict_with_uncertainty(x)
    assert np.allclose(m_plain, m_masked, atol=1e-6)

    ref_grid = np.array([[40.0, 0.4, 0.5, 2.0]])
    cov = estimate_noise_cov([])
    s0 = select_optimal_designs([design], ref_grid, cov, measurement_budget_shots=0)[0]
    s1 = select_optimal_designs([design], ref_grid, cov, measurement_budget_shots=1)[0]
    s2 = select_optimal_designs([design], ref_grid, cov, measurement_budget_shots=2)[0]
    assert np.isinf(s0.a_opt_norm_trace)
    assert s2.d_opt_logdet > s1.d_opt_logdet


def test_level0_extractor_to_nls_end_to_end() -> None:
    """Verify Level 0 simulator -> extractor -> dev calibration -> ClassicalNLS recovers nL cleanly."""
    from biolaser_twin.extractor import extract_sweep_features
    from biolaser_twin.simulator import FabryPerotConfig, FabryPerotTruthSimulator, LatentProxyVector, map_latents_to_truth_state

    l_um, n_val = 30.0, 1.345
    cfg = FabryPerotConfig(cavity_length_um=l_um, level=0, n_spatial_px=16, seed=10)
    sim = FabryPerotTruthSimulator(cfg)
    design = DesignConfig(nominal_cavity_um=l_um, excitations_nm=(405.0, 440.0))
    energies = np.linspace(2.0, 50.0, 10)

    dev_recs = []
    dev_lats = []
    for z2 in (0.3, 0.7):
        st = map_latents_to_truth_state(LatentProxyVector(0.3, z2, n_val, 0.5, l_um, 10.0))
        b405 = sim.render_pump_sweep(st, 405.0, energies)
        b440 = sim.render_pump_sweep(st, 440.0, energies)
        f405 = extract_sweep_features(b405.frames, b405.wavelengths_nm, b405.pump_energies_uj, dark_frames=b405.dark_frames)
        f440 = extract_sweep_features(b440.frames, b440.wavelengths_nm, b440.pump_energies_uj, dark_frames=b440.dark_frames)
        dev_recs.append(build_observation_record(f405, f440, design, max_pump_uj=50.0))
        dev_lats.append([n_val * l_um, 0.3, z2, 0.5])

    cal_consts = calibrate_nominal_constants_from_dev(dev_recs, np.asarray(dev_lats))
    nls = ClassicalNLSSolver(constants=cal_consts, noise_cov=estimate_noise_cov([]))
    res = nls.predict_records(dev_recs)
    assert np.all(~res.fit_failed)
    assert np.all(np.abs(res.means[:, 0] - n_val * l_um) / (n_val * l_um) < 0.02)


def test_all_downstream_modules_isolated_from_simulator() -> None:
    """Verify identifiability, solvers, observations, calibration, and metrics never import simulator."""
    for mod in list(sys.modules):
        if mod.startswith("biolaser_twin"):
            del sys.modules[mod]
    for sub in ("identifiability", "solvers", "observations", "calibration", "metrics"):
        importlib.import_module(f"biolaser_twin.{sub}")
    assert not any(m.startswith("biolaser_twin.simulator") for m in sys.modules)
