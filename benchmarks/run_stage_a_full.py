"""Full Stage A Benchmark Runner for BioLaserTwin-CPC (S1, S5, S2 + Level 2 OOD).

Includes:
- 1024-pixel spectral grid + dual-excitation (405 + 440 nm) frames for all frame models
- True held-out extrapolation split L in [15, 20) U (40, 45] um (zero overlap with [20, 40])
- S1: 16 (design x latent) pairs with tau=0.15, per-design Sigma, and empirical profile/NLS spread
- S5: Per-design Sigma across (resolution x L x excitation) + held-out empirical variance reduction
- S2: Minibatch + early-stopping neural baselines, MeanOnly baseline, separate val_tune vs val_cal,
      per-fraction lambda tuning + exploratory selective R_phys (FSR + scattering),
      hierarchical per-seed bootstrap, and one-sided non-inferiority test (delta=0.05) with Holm.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr

from biolaser_twin.calibration import SplitConformalCalibrator
from biolaser_twin.constants import LATENT_NAMES, LATENT_PRIOR_SCALES
from biolaser_twin.extractor import extract_sweep_features
from biolaser_twin.identifiability import (
    DesignConfig,
    check_structural_n_l_rank,
    evaluate_design,
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
from biolaser_twin.simulator import (
    FabryPerotConfig,
    FabryPerotTruthSimulator,
    LatentProxyVector,
    map_latents_to_truth_state,
)
from biolaser_twin.solvers import (
    ClassicalNLSSolver,
    FrameCNNSolver,
    FrameViTSolver,
    HybridCNNFeaturesSolver,
    MeanOnlySolver,
    PhotonicPIELMSolver,
    PlainELMSolver,
    TabularMLPSolver,
)


def _sample_cavity_lengths(rng: np.random.Generator, n_exp: int, extrap: bool) -> np.ndarray:
    if not extrap:
        return rng.uniform(20.0, 40.0, size=n_exp)
    # Disjoint extrapolation bands [15, 20) U (40, 45]
    lo = rng.uniform(15.0, 19.9, size=n_exp)
    hi = rng.uniform(40.1, 45.0, size=n_exp)
    pick_hi = rng.random(size=n_exp) > 0.5
    return np.where(pick_hi, hi, lo)


def _generate_dataset(
    n_exp: int,
    level: int,
    seed: int,
    extrap: bool = False,
    psf_fwhm_nm: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ObservationRecord], float]:
    """Generate independent experiments on 1024-px grid with dual-excitation frames."""
    rng = np.random.default_rng(seed)
    energies = np.linspace(2.0, 48.0, 8)
    n_spectral_px = 1024
    l_vals = _sample_cavity_lengths(rng, n_exp, extrap=extrap)

    feats_list = []
    targets_list = []
    frames_list = []
    records: list[ObservationRecord] = []
    fit_failures = 0

    for idx in range(n_exp):
        l_um = float(l_vals[idx])
        n_val = float(rng.uniform(1.335, 1.365))
        z1 = float(rng.uniform(0.15, 0.85))
        z2 = float(rng.uniform(0.15, 0.85))
        z4 = float(rng.uniform(0.2, 6.0))

        cfg = FabryPerotConfig(
            cavity_length_um=l_um,
            refractive_index=n_val,
            n_spectral_px=n_spectral_px,
            n_spatial_px=16,
            spec_psf_fwhm_nm=psf_fwhm_nm,
            oversample=16,
            level=level,
            seed=seed * 10000 + idx,
        )
        sim = FabryPerotTruthSimulator(cfg)
        state = map_latents_to_truth_state(LatentProxyVector(z1, z2, n_val, z4, l_um, tht_conc_mm=10.0))

        b405 = sim.render_pump_sweep(state, 405.0, energies)
        b440 = sim.render_pump_sweep(state, 440.0, energies)
        f405 = extract_sweep_features(
            b405.frames, b405.wavelengths_nm, b405.pump_energies_uj,
            spec_psf_fwhm_nm=b405.spec_psf_fwhm_nm, dark_frames=b405.dark_frames,
            nominal_cavity_um=b405.nominal_cavity_um,
        )
        f440 = extract_sweep_features(
            b440.frames, b440.wavelengths_nm, b440.pump_energies_uj,
            spec_psf_fwhm_nm=b440.spec_psf_fwhm_nm, dark_frames=b440.dark_frames,
            nominal_cavity_um=b440.nominal_cavity_um,
        )
        # Outlier-aware failure check on FSR
        exp_fsr = (485.0**2) / (2.0 * 1.345 * l_um * 1000.0)
        fsr_bad = (
            f440.mean_fsr_nm is None
            or abs(f440.mean_fsr_nm - exp_fsr) / exp_fsr > 0.25
        )
        if f405.fit_failed or f440.fit_failed or fsr_bad:
            fit_failures += 1

        design = DesignConfig(
            nominal_cavity_um=b440.nominal_cavity_um,
            spec_psf_fwhm_nm=psf_fwhm_nm,
            excitations_nm=(405.0, 440.0),
        )
        rec = build_observation_record(f405, f440, design, max_pump_uj=float(energies[-1]))
        records.append(rec)

        x_tab = rec.values.copy()
        if not rec.mask[0]:
            x_tab[0] = (485.0**2) / (2.0 * 1.34 * design.nominal_cavity_um * 1000.0)
        if not rec.mask[1]:
            x_tab[1] = float(energies[-1]) * 1.1
        if not rec.mask[2]:
            x_tab[2] = float(energies[-1]) * 1.1
        feats_list.append(x_tab)
        targets_list.append([n_val * l_um, z1, z2, z4])
        # Stack BOTH 405 nm and 440 nm sweeps -> shape (2, n_sweeps, n_spatial, 1024)
        frames_list.append(np.stack([b405.frames, b440.frames], axis=0))

    return (
        np.asarray(feats_list, dtype=float),
        np.asarray(targets_list, dtype=float),
        np.asarray(frames_list, dtype=np.float32),
        records,
        fit_failures / max(1, n_exp),
    )


def _estimate_design_noise_cov(design: DesignConfig, level: int = 1, n_repeats: int = 8, seed: int = 900) -> np.ndarray:
    """Estimate empirical noise covariance Sigma for a specific hardware DesignConfig."""
    energies = np.linspace(2.0, 48.0, 8)
    l_um = design.nominal_cavity_um
    n_val = 1.345
    sim = FabryPerotTruthSimulator(
        FabryPerotConfig(
            cavity_length_um=l_um,
            refractive_index=n_val,
            n_spectral_px=1024,
            n_spatial_px=16,
            spec_psf_fwhm_nm=design.spec_psf_fwhm_nm,
            oversample=16,
            level=level,
            seed=seed,
        )
    )
    state = map_latents_to_truth_state(LatentProxyVector(0.5, 0.5, n_val, 2.0, l_um, tht_conc_mm=10.0))
    recs = []
    for _ in range(n_repeats):
        f405 = None
        f440 = None
        if 405.0 in design.excitations_nm:
            b405 = sim.render_pump_sweep(state, 405.0, energies)
            f405 = extract_sweep_features(
                b405.frames, b405.wavelengths_nm, b405.pump_energies_uj,
                spec_psf_fwhm_nm=b405.spec_psf_fwhm_nm, dark_frames=b405.dark_frames,
                nominal_cavity_um=b405.nominal_cavity_um,
            )
        if 440.0 in design.excitations_nm:
            b440 = sim.render_pump_sweep(state, 440.0, energies)
            f440 = extract_sweep_features(
                b440.frames, b440.wavelengths_nm, b440.pump_energies_uj,
                spec_psf_fwhm_nm=b440.spec_psf_fwhm_nm, dark_frames=b440.dark_frames,
                nominal_cavity_um=b440.nominal_cavity_um,
            )
        recs.append(build_observation_record(f405, f440, design, max_pump_uj=float(energies[-1])))
    return estimate_noise_cov(recs, min_std=1e-3)


def _eval_predictions(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    nrmse = normalized_rmse(y_true, mean, ref_scale=LATENT_PRIOR_SCALES)
    nll = gaussian_nll(y_true, mean, std)
    crps = gaussian_crps(y_true, mean, std)
    cov, width = interval_coverage_and_width(y_true, mean, std)
    return {
        "mean_nrmse": float(np.mean(nrmse)),
        "primary_nl_z4_nrmse": float(0.5 * (nrmse[0] + nrmse[3])),
        "per_latent": {
            name: {
                "nrmse": float(nrmse[i]),
                "nll": float(nll[i]),
                "crps": float(crps[i]),
                "cov90": float(cov[i]),
                "width90": float(width[i]),
            }
            for i, name in enumerate(LATENT_NAMES)
        },
    }


def _hierarchical_bootstrap_ni_holm(
    y_true: np.ndarray,
    seed_preds_by_solver: dict[str, list[np.ndarray]],
    ref_solver: str,
    margin_delta: float = 0.05,
    n_boot: int = 500,
    seed: int = 777,
) -> dict[str, dict]:
    """Hierarchical bootstrap over (seeds x test_experiments) with one-sided NI test & Holm."""
    rng = np.random.default_rng(seed)
    n_test = len(y_true)
    n_seeds = len(seed_preds_by_solver[ref_solver])

    def _mean_seed_primary(preds_list: list[np.ndarray], s_idx: np.ndarray, e_idx: np.ndarray) -> float:
        vals = []
        for s in s_idx:
            nr = normalized_rmse(y_true[e_idx], preds_list[int(s)][e_idx], ref_scale=LATENT_PRIOR_SCALES)
            vals.append(0.5 * (nr[0] + nr[3]))
        return float(np.mean(vals))

    all_s = np.arange(n_seeds)
    all_e = np.arange(n_test)
    ref_obs = _mean_seed_primary(seed_preds_by_solver[ref_solver], all_s, all_e)

    raw_p_ni: dict[str, float] = {}
    diffs_obs: dict[str, float] = {}
    ci_95: dict[str, list[float]] = {}

    for name, p_list in seed_preds_by_solver.items():
        if name == ref_solver:
            continue
        cur_obs = _mean_seed_primary(p_list, all_s, all_e)
        # Positive diff means ref_solver (PhotonicPIELM) has HIGHER error than comparator
        diffs_obs[name] = float(ref_obs - cur_obs)
        boot_diffs = np.zeros(n_boot, dtype=float)
        for b in range(n_boot):
            s_idx = rng.integers(0, n_seeds, size=n_seeds)
            e_idx = rng.integers(0, n_test, size=n_test)
            boot_diffs[b] = _mean_seed_primary(seed_preds_by_solver[ref_solver], s_idx, e_idx) - _mean_seed_primary(p_list, s_idx, e_idx)
        # One-sided non-inferiority H0: (err_ref - err_comp) >= delta
        p_ni = max(1.0 / (n_boot + 1), float(np.mean(boot_diffs >= margin_delta)))
        raw_p_ni[name] = p_ni
        ci_95[name] = [float(np.quantile(boot_diffs, 0.025)), float(np.quantile(boot_diffs, 0.975))]

    sorted_names = sorted(raw_p_ni, key=lambda k: raw_p_ni[k])
    m = len(sorted_names)
    holm_p: dict[str, float] = {}
    running_max = 0.0
    for rank_idx, k in enumerate(sorted_names):
        adj = min(1.0, raw_p_ni[k] * (m - rank_idx))
        running_max = max(running_max, adj)
        holm_p[k] = running_max

    return {
        k: {
            "err_ref_minus_comp": diffs_obs[k],
            "ci95": ci_95[k],
            "p_ni_raw": raw_p_ni[k],
            "p_ni_holm": holm_p[k],
            "non_inferior_at_0.05": bool(holm_p[k] < 0.05),
        }
        for k in sorted_names
    }


def main() -> None:
    t0 = time.perf_counter()
    print("=== Stage A Full Benchmark (1024px, S1, S5, S2) ===")

    # 1. Dev calibration
    _, y_dev, _, recs_dev, dev_fail = _generate_dataset(16, level=1, seed=1)
    cal_consts = calibrate_nominal_constants_from_dev(recs_dev, y_dev)

    # 2. S1 & S5 across 4 diverse hardware designs (resolution x L x excitation)
    rank5, dim5 = check_structural_n_l_rank(constants=cal_consts)
    s1_designs = [
        DesignConfig(nominal_cavity_um=20.0, spec_psf_fwhm_nm=0.40, excitations_nm=(405.0,)),
        DesignConfig(nominal_cavity_um=30.0, spec_psf_fwhm_nm=0.40, excitations_nm=(440.0,)),
        DesignConfig(nominal_cavity_um=30.0, spec_psf_fwhm_nm=0.10, excitations_nm=(440.0,)),
        DesignConfig(nominal_cavity_um=40.0, spec_psf_fwhm_nm=0.08, excitations_nm=(405.0, 440.0)),
    ]
    design_covs = [
        _estimate_design_noise_cov(d, level=1, n_repeats=6, seed=900 + i)
        for i, d in enumerate(s1_designs)
    ]

    # S1: 16 (design x latent) pairs comparing FIM CRB (tau=0.15) vs empirical NLS SD across repeated noise
    crb_vals = []
    emp_sd_vals = []
    s1_pairs = []
    tau_s1 = 0.15
    agreements = []
    for d, cov_d in zip(s1_designs, design_covs):
        ref_z = np.array([1.345 * d.nominal_cavity_um, 0.5, 0.5, 2.0])
        # Match 1-sweep-per-excitation empirical NLS budget (B_shots = len(d.excitations_nm))
        score_d = evaluate_design(
            d, ref_z[None, :], noise_cov=cov_d, constants=cal_consts,
            rel_sd_threshold=tau_s1, measurement_budget_shots=len(d.excitations_nm),
        )
        crb_rel = score_d.mean_crb_sd / LATENT_PRIOR_SCALES

        # Empirical NLS recovery spread over 8 repeated noisy sweeps at ref_z
        sim_d = FabryPerotTruthSimulator(
            FabryPerotConfig(
                cavity_length_um=d.nominal_cavity_um, refractive_index=1.345,
                n_spectral_px=1024, n_spatial_px=16, spec_psf_fwhm_nm=d.spec_psf_fwhm_nm, level=1, seed=800,
            )
        )
        st_d = map_latents_to_truth_state(LatentProxyVector(0.5, 0.5, 1.345, 2.0, d.nominal_cavity_um, 10.0))
        energies = np.linspace(2.0, 48.0, 8)
        recs_rep = []
        for _ in range(8):
            f405 = extract_sweep_features(
                *(lambda b: (b.frames, b.wavelengths_nm, b.pump_energies_uj, b.spec_psf_fwhm_nm, b.dark_frames, b.nominal_cavity_um))(
                    sim_d.render_pump_sweep(st_d, 405.0, energies)
                )
            ) if 405.0 in d.excitations_nm else None
            f440 = extract_sweep_features(
                *(lambda b: (b.frames, b.wavelengths_nm, b.pump_energies_uj, b.spec_psf_fwhm_nm, b.dark_frames, b.nominal_cavity_um))(
                    sim_d.render_pump_sweep(st_d, 440.0, energies)
                )
            ) if 440.0 in d.excitations_nm else None
            recs_rep.append(build_observation_record(f405, f440, d, max_pump_uj=48.0))
        nls_d = ClassicalNLSSolver(constants=cal_consts, noise_cov=cov_d).predict_records(recs_rep)
        emp_rel_sd = np.std(nls_d.means, axis=0) / LATENT_PRIOR_SCALES

        for j in range(4):
            if np.isfinite(crb_rel[j]):
                crb_vals.append(float(crb_rel[j]))
                emp_sd_vals.append(float(emp_rel_sd[j]))
                agreements.append(bool((crb_rel[j] < tau_s1) == (emp_rel_sd[j] < tau_s1)))
                s1_pairs.append({
                    "L_um": d.nominal_cavity_um,
                    "psf_nm": d.spec_psf_fwhm_nm,
                    "exc": list(d.excitations_nm),
                    "latent": LATENT_NAMES[j],
                    "crb_rel_sd": float(crb_rel[j]),
                    "emp_nls_rel_sd": float(emp_rel_sd[j]),
                    "agree_at_tau": bool((crb_rel[j] < tau_s1) == (emp_rel_sd[j] < tau_s1)),
                })

    rho_s1, p_s1 = spearmanr(crb_vals, emp_sd_vals)
    s1_agreement = float(np.mean(agreements))

    # S5: Budget-matched DoE across designs + empirical held-out posterior variance check
    ref_grid = np.array([[32.0, 0.3, 0.3, 1.5], [40.0, 0.5, 0.5, 3.0], [50.0, 0.7, 0.7, 4.5]])
    doe_scores = [
        evaluate_design(d, ref_grid, noise_cov=cov_d, constants=cal_consts, rel_sd_threshold=tau_s1, measurement_budget_shots=2)
        for d, cov_d in zip(s1_designs, design_covs)
    ]
    doe_ranked = sorted(doe_scores, key=lambda s: s.a_opt_norm_trace)
    a_opt_best = doe_ranked[0].a_opt_norm_trace
    a_opt_worst = doe_ranked[-1].a_opt_norm_trace
    doe_var_reduction_pct = 100.0 * (a_opt_worst - a_opt_best) / max(a_opt_worst, 1e-8)

    # Empirical S5 test on N=40 held-out simulated samples at matched 2-sweep budget
    def _eval_design_empirical(d_cfg: DesignConfig, cov_d: np.ndarray, n_eval: int = 40, seed_base: int = 950) -> dict:
        rng_e = np.random.default_rng(seed_base)
        energies_e = np.linspace(2.0, 48.0, 8)
        recs_e, y_e = [], []
        for idx_e in range(n_eval):
            n_v = float(rng_e.uniform(1.335, 1.365))
            z1_v = float(rng_e.uniform(0.15, 0.85))
            z2_v = float(rng_e.uniform(0.15, 0.85))
            z4_v = float(rng_e.uniform(0.2, 6.0))
            sim_e = FabryPerotTruthSimulator(
                FabryPerotConfig(
                    cavity_length_um=d_cfg.nominal_cavity_um, refractive_index=n_v,
                    n_spectral_px=1024, n_spatial_px=16, spec_psf_fwhm_nm=d_cfg.spec_psf_fwhm_nm,
                    level=1, seed=seed_base * 1000 + idx_e,
                )
            )
            st_e = map_latents_to_truth_state(LatentProxyVector(z1_v, z2_v, n_v, z4_v, d_cfg.nominal_cavity_um, 10.0))
            if len(d_cfg.excitations_nm) == 2:
                b405 = sim_e.render_pump_sweep(st_e, 405.0, energies_e)
                b440 = sim_e.render_pump_sweep(st_e, 440.0, energies_e)
                f405 = extract_sweep_features(b405.frames, b405.wavelengths_nm, b405.pump_energies_uj, spec_psf_fwhm_nm=b405.spec_psf_fwhm_nm, dark_frames=b405.dark_frames, nominal_cavity_um=b405.nominal_cavity_um)
                f440 = extract_sweep_features(b440.frames, b440.wavelengths_nm, b440.pump_energies_uj, spec_psf_fwhm_nm=b440.spec_psf_fwhm_nm, dark_frames=b440.dark_frames, nominal_cavity_um=b440.nominal_cavity_um)
            else:
                exc_w = d_cfg.excitations_nm[0]
                b1 = sim_e.render_pump_sweep(st_e, exc_w, energies_e)
                b2 = sim_e.render_pump_sweep(st_e, exc_w, energies_e)
                avg_f = 0.5 * (b1.frames + b2.frames)
                avg_d = 0.5 * (b1.dark_frames + b2.dark_frames)
                feat_s = extract_sweep_features(avg_f, b1.wavelengths_nm, b1.pump_energies_uj, spec_psf_fwhm_nm=b1.spec_psf_fwhm_nm, dark_frames=avg_d, nominal_cavity_um=b1.nominal_cavity_um)
                f405 = feat_s if exc_w < 430.0 else None
                f440 = feat_s if exc_w >= 430.0 else None
            recs_e.append(build_observation_record(f405, f440, d_cfg, max_pump_uj=48.0))
            y_e.append([n_v * d_cfg.nominal_cavity_um, z1_v, z2_v, z4_v])
        y_arr = np.asarray(y_e, dtype=float)
        nls_e = ClassicalNLSSolver(constants=cal_consts, noise_cov=cov_d / (2.0 / len(d_cfg.excitations_nm))).predict_records(recs_e)
        rel_err = (nls_e.means - y_arr) / LATENT_PRIOR_SCALES[None, :]
        mse_norm = float(np.mean(np.sum(rel_err**2, axis=1)))
        nrmse_per = normalized_rmse(y_arr, nls_e.means, LATENT_PRIOR_SCALES)
        return {"norm_mse_sum": mse_norm, "mean_nrmse": float(np.mean(nrmse_per)), "per_latent_nrmse": nrmse_per.tolist()}

    emp_doe_best = _eval_design_empirical(doe_ranked[0].design, design_covs[3])
    emp_doe_worst = _eval_design_empirical(doe_ranked[-1].design, design_covs[s1_designs.index(doe_ranked[-1].design)])
    emp_doe_reduction_pct = 100.0 * (emp_doe_worst["norm_mse_sum"] - emp_doe_best["norm_mse_sum"]) / max(emp_doe_worst["norm_mse_sum"], 1e-8)

    # 3. Generate full Level 1 & Level 2 OOD datasets (1024 px, disjoint extrap, separate val_tune vs val_cal)
    x_tr, y_tr, f_tr, recs_tr, fail_tr = _generate_dataset(300, level=1, seed=10)
    x_col, _, _, recs_col, _ = _generate_dataset(60, level=1, seed=11)
    x_va_tune, y_va_tune, f_va_tune, recs_va_tune, _ = _generate_dataset(60, level=1, seed=12)
    x_va_cal, y_va_cal, f_va_cal, recs_va_cal, _ = _generate_dataset(60, level=1, seed=16)
    x_te, y_te, f_te, recs_te, fail_te = _generate_dataset(200, level=1, seed=13)
    x_ex, y_ex, f_ex, recs_ex, fail_ex = _generate_dataset(100, level=1, seed=14, extrap=True)
    x_ood, y_ood, f_ood, recs_ood, fail_ood = _generate_dataset(100, level=2, seed=15)

    print(
        f"1024-px Data generated in {time.perf_counter() - t0:.1f}s | fit_fail: "
        f"tr={fail_tr:.2%} te={fail_te:.2%} ex={fail_ex:.2%} ood={fail_ood:.2%}"
    )

    noise_cov = design_covs[3]  # empirical dual-excitation covariance
    # Exploratory selective R_phys mask: FSR (idx 0) + scattering (idx 5)
    selective_mask = np.array([True, False, False, False, False, True], dtype=bool)

    fractions = {"10pct": 30, "25pct": 75, "50pct": 150, "100pct": 300}
    seeds = [1, 2, 3, 4, 5]
    solver_names = [
        "MeanOnly", "PlainELM", "PhotonicPIELM", "SelectivePIELM",
        "ClassicalNLS", "TabularMLP", "FrameCNN", "FrameViT", "HybridCNNFeatures",
    ]

    # ClassicalNLS evaluated once
    nls = ClassicalNLSSolver(constants=cal_consts, noise_cov=noise_cov).fit()
    nls_va = nls.predict_records(recs_va_cal)
    nls_te = nls.predict_records(recs_te)
    nls_ex = nls.predict_records(recs_ex)
    nls_ood = nls.predict_records(recs_ood)
    nls_cal = SplitConformalCalibrator(0.10).fit(y_va_cal, nls_va.means, nls_va.stds)
    nls_summary = {
        "level1_interp": _eval_predictions(y_te, nls_te.means, nls_cal.calibrate_std(nls_te.stds)),
        "level1_extrap": _eval_predictions(y_ex, nls_ex.means, nls_cal.calibrate_std(nls_ex.stds)),
        "level2_ood": _eval_predictions(y_ood, nls_ood.means, nls_cal.calibrate_std(nls_ood.stds)),
    }

    curve_results: dict[str, dict] = {}
    val_lambda_logs: dict[str, dict] = {}
    primary_25_seed_preds_te: dict[str, list[np.ndarray]] = {s: [] for s in solver_names if s != "ClassicalNLS"}
    lambda_grid = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1)
    solver_timings_100pct: dict[str, list[float]] = {s: [] for s in solver_names if s != "ClassicalNLS"}

    for frac_label, n_sub in fractions.items():
        frac_res: dict[str, list[dict]] = {s: [] for s in solver_names if s != "ClassicalNLS"}
        val_lambda_logs[frac_label] = {
            "PhotonicPIELM": {str(l): [] for l in lambda_grid},
            "SelectivePIELM": {str(l): [] for l in lambda_grid},
            "selected_full_per_seed": [],
            "selected_sel_per_seed": [],
        }

        def _tune_lambda_seed(xs_s: np.ndarray, ys_s: np.ndarray, sd_val: int, obs_mask: np.ndarray | None, log_key: str) -> float:
            best_l, best_v = 0.0, np.inf
            for lam in lambda_grid:
                m = PhotonicPIELMSolver(
                    hidden_dim=48, phys_lambda=lam, n_ensemble=3, n_gn_iters=3,
                    constants=cal_consts, noise_cov=noise_cov, active_obs_mask=obs_mask, seed=sd_val,
                ).fit(xs_s, ys_s, collocation_features=x_col, collocation_records=recs_col)
                verr = float(np.mean(normalized_rmse(y_va_tune, m.predict(x_va_tune), LATENT_PRIOR_SCALES)))
                val_lambda_logs[frac_label][log_key][str(lam)].append(verr)
                if verr < best_v:
                    best_v, best_l = verr, lam
            return best_l

        for sd in seeds:
            # Independent training subset permutation per seed
            perm = np.random.default_rng(1000 + sd).permutation(len(x_tr))[:n_sub]
            xs, ys, fs = x_tr[perm], y_tr[perm], f_tr[perm]

            lam_full = _tune_lambda_seed(xs, ys, sd, None, "PhotonicPIELM")
            lam_sel = _tune_lambda_seed(xs, ys, sd, selective_mask, "SelectivePIELM")
            val_lambda_logs[frac_label]["selected_full_per_seed"].append(lam_full)
            val_lambda_logs[frac_label]["selected_sel_per_seed"].append(lam_sel)

            models = {}
            fit_specs = [
                ("MeanOnly", lambda: MeanOnlySolver().fit(ys)),
                ("PlainELM", lambda: PlainELMSolver(hidden_dim=48, n_ensemble=3, seed=sd).fit(xs, ys)),
                ("PhotonicPIELM", lambda: PhotonicPIELMSolver(
                    hidden_dim=48, phys_lambda=lam_full, n_ensemble=3, n_gn_iters=3,
                    constants=cal_consts, noise_cov=noise_cov, seed=sd,
                ).fit(xs, ys, collocation_features=x_col, collocation_records=recs_col)),
                ("SelectivePIELM", lambda: PhotonicPIELMSolver(
                    hidden_dim=48, phys_lambda=lam_sel, n_ensemble=3, n_gn_iters=3,
                    constants=cal_consts, noise_cov=noise_cov, active_obs_mask=selective_mask, seed=sd,
                ).fit(xs, ys, collocation_features=x_col, collocation_records=recs_col)),
                ("TabularMLP", lambda: TabularMLPSolver(hidden_dim=64, epochs=80, n_ensemble=3, seed=sd).fit(xs, ys, x_va_tune, y_va_tune)),
                ("FrameCNN", lambda: FrameCNNSolver(epochs=80, lr=2e-3, seed=sd).fit(fs, ys, f_va_tune, y_va_tune)),
                ("FrameViT", lambda: FrameViTSolver(patch_size=16, epochs=120, lr=1e-3, seed=sd).fit(fs, ys, f_va_tune, y_va_tune)),
                ("HybridCNNFeatures", lambda: HybridCNNFeaturesSolver(epochs=80, lr=3e-3, seed=sd).fit(fs, xs, ys, f_va_tune, x_va_tune, y_va_tune)),
            ]
            for s_name, fn in fit_specs:
                ts_fit = time.perf_counter()
                models[s_name] = fn()
                if frac_label == "100pct":
                    solver_timings_100pct[s_name].append(float(time.perf_counter() - ts_fit))

            for s_name, model in models.items():
                if s_name == "MeanOnly":
                    vm, vs = model.predict_with_uncertainty(len(y_va_cal))
                    tm, ts = model.predict_with_uncertainty(len(y_te))
                    em, es = model.predict_with_uncertainty(len(y_ex))
                    om, os_ = model.predict_with_uncertainty(len(y_ood))
                elif s_name in ("FrameCNN", "FrameViT"):
                    vm, vs = model.predict_with_uncertainty(f_va_cal)
                    tm, ts = model.predict_with_uncertainty(f_te)
                    em, es = model.predict_with_uncertainty(f_ex)
                    om, os_ = model.predict_with_uncertainty(f_ood)
                elif s_name == "HybridCNNFeatures":
                    vm, vs = model.predict_with_uncertainty(f_va_cal, x_va_cal)
                    tm, ts = model.predict_with_uncertainty(f_te, x_te)
                    em, es = model.predict_with_uncertainty(f_ex, x_ex)
                    om, os_ = model.predict_with_uncertainty(f_ood, x_ood)
                else:
                    vm, vs = model.predict_with_uncertainty(x_va_cal)
                    tm, ts = model.predict_with_uncertainty(x_te)
                    em, es = model.predict_with_uncertainty(x_ex)
                    om, os_ = model.predict_with_uncertainty(x_ood)

                cal = SplitConformalCalibrator(0.10).fit(y_va_cal, vm, vs)
                frac_res[s_name].append(
                    {
                        "level1_interp": _eval_predictions(y_te, tm, cal.calibrate_std(ts)),
                        "level1_extrap": _eval_predictions(y_ex, em, cal.calibrate_std(es)),
                        "level2_ood": _eval_predictions(y_ood, om, cal.calibrate_std(os_)),
                    }
                )
                if frac_label == "25pct":
                    primary_25_seed_preds_te[s_name].append(tm)

        # Summarize per-seed lambda logs as mean across the 5 seeds
        for log_k in ("PhotonicPIELM", "SelectivePIELM"):
            val_lambda_logs[frac_label][f"{log_k}_mean"] = {
                str(l): float(np.mean(val_lambda_logs[frac_label][log_k][str(l)]))
                for l in lambda_grid
            }

        agg: dict[str, dict] = {"ClassicalNLS": nls_summary}
        for s_name, runs in frac_res.items():
            agg[s_name] = {}
            for split_key in ("level1_interp", "level1_extrap", "level2_ood"):
                agg[s_name][split_key] = {
                    "mean_nrmse": float(np.mean([r[split_key]["mean_nrmse"] for r in runs])),
                    "std_nrmse": float(np.std([r[split_key]["mean_nrmse"] for r in runs])),
                    "primary_nl_z4_nrmse": float(np.mean([r[split_key]["primary_nl_z4_nrmse"] for r in runs])),
                    "primary_nl_z4_std": float(np.std([r[split_key]["primary_nl_z4_nrmse"] for r in runs])),
                    "per_latent": {
                        lat: {
                            m_key: float(np.mean([r[split_key]["per_latent"][lat][m_key] for r in runs]))
                            for m_key in ("nrmse", "nll", "crps", "cov90", "width90")
                        } | {
                            "nrmse_std": float(np.std([r[split_key]["per_latent"][lat]["nrmse"] for r in runs]))
                        }
                        for lat in LATENT_NAMES
                    },
                }
        curve_results[frac_label] = agg

    primary_25_seed_preds_te["ClassicalNLS"] = [nls_te.means for _ in seeds]
    boot_25_05 = _hierarchical_bootstrap_ni_holm(
        y_te, primary_25_seed_preds_te, ref_solver="PhotonicPIELM", margin_delta=0.05, n_boot=5000
    )
    boot_25_015 = _hierarchical_bootstrap_ni_holm(
        y_te, primary_25_seed_preds_te, ref_solver="PhotonicPIELM", margin_delta=0.015, n_boot=5000
    )

    # Level 0 Matched-Physics Sanity Control (25% data = 75 train, 5 seeds)
    _, y_dev0, _, recs_dev0, _ = _generate_dataset(16, level=0, seed=501)
    cal_consts_l0 = calibrate_nominal_constants_from_dev(recs_dev0, y_dev0)
    cov_l0 = _estimate_design_noise_cov(s1_designs[3], level=0, n_repeats=6, seed=910)
    x_tr0, y_tr0, _, _, _ = _generate_dataset(75, level=0, seed=510)
    x_col0, _, _, recs_col0, _ = _generate_dataset(60, level=0, seed=511)
    x_va0, y_va0, _, _, _ = _generate_dataset(60, level=0, seed=512)
    x_te0, y_te0, _, _, _ = _generate_dataset(100, level=0, seed=513)
    l0_plain, l0_pielm, l0_lams = [], [], []
    for sd in seeds:
        m_plain0 = PlainELMSolver(hidden_dim=48, n_ensemble=3, seed=sd).fit(x_tr0, y_tr0)
        l0_plain.append(float(np.mean(normalized_rmse(y_te0, m_plain0.predict(x_te0), LATENT_PRIOR_SCALES))))
        best_l0, best_v0 = 0.0, np.inf
        for lam in lambda_grid:
            m_p0 = PhotonicPIELMSolver(
                hidden_dim=48, phys_lambda=lam, n_ensemble=3, n_gn_iters=3,
                constants=cal_consts_l0, noise_cov=cov_l0, seed=sd,
            ).fit(x_tr0, y_tr0, collocation_features=x_col0, collocation_records=recs_col0)
            v0 = float(np.mean(normalized_rmse(y_va0, m_p0.predict(x_va0), LATENT_PRIOR_SCALES)))
            if v0 < best_v0:
                best_v0, best_l0 = v0, lam
        m_best0 = PhotonicPIELMSolver(
            hidden_dim=48, phys_lambda=best_l0, n_ensemble=3, n_gn_iters=3,
            constants=cal_consts_l0, noise_cov=cov_l0, seed=sd,
        ).fit(x_tr0, y_tr0, collocation_features=x_col0, collocation_records=recs_col0)
        l0_pielm.append(float(np.mean(normalized_rmse(y_te0, m_best0.predict(x_te0), LATENT_PRIOR_SCALES))))
        l0_lams.append(best_l0)

    out = {
        "calibrated_kappa": cal_consts.gain_scale_kappa,
        "fit_failure_rates": {"dev": dev_fail, "train": fail_tr, "test": fail_te, "extrap": fail_ex, "ood": fail_ood},
        "s1_identifiability": {
            "structural_rank": rank5,
            "full_dim": dim5,
            "n_pairs": len(crb_vals),
            "tau": tau_s1,
            "agreement_rate": s1_agreement,
            "crb_vs_nls_spearman_rho": float(rho_s1) if np.isfinite(rho_s1) else None,
            "crb_vs_nls_p": float(p_s1) if np.isfinite(p_s1) else None,
            "pairs": s1_pairs,
        },
        "s5_doe": {
            "top_design": {
                "nominal_cavity_um": doe_ranked[0].design.nominal_cavity_um,
                "excitations_nm": list(doe_ranked[0].design.excitations_nm),
                "spec_psf_fwhm_nm": doe_ranked[0].design.spec_psf_fwhm_nm,
                "a_opt_norm_trace": a_opt_best,
            },
            "worst_design_a_opt": a_opt_worst,
            "theoretical_trace_reduction_pct": float(doe_var_reduction_pct),
            "empirical_heldout_n40": {
                "best_design": emp_doe_best,
                "worst_design": emp_doe_worst,
                "empirical_norm_mse_reduction_pct": float(emp_doe_reduction_pct),
            },
        },
        "level0_control": {
            "plain_elm_mean_nrmse": float(np.mean(l0_plain)),
            "plain_elm_std_nrmse": float(np.std(l0_plain)),
            "photonic_pielm_mean_nrmse": float(np.mean(l0_pielm)),
            "photonic_pielm_std_nrmse": float(np.std(l0_pielm)),
            "selected_lambdas_per_seed": l0_lams,
        },
        "val_lambda_logs_all_seeds": val_lambda_logs,
        "s2_curves": curve_results,
        "s2_primary_25pct_hierarchical_ni_bootstrap": boot_25_05,
        "s2_primary_25pct_hierarchical_ni_bootstrap_delta015": boot_25_015,
        "solver_timings_100pct_s": {k: float(np.mean(v)) for k, v in solver_timings_100pct.items()},
        "total_runtime_s": float(time.perf_counter() - t0),
    }

    out_path = Path(__file__).resolve().parent / "results" / "stage_a_full.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\n=== Completed in {out['total_runtime_s']:.1f}s | Saved to {out_path} ===")
    print(f"S1 (16 pairs): agreement={s1_agreement:.1%}, Spearman rho={out['s1_identifiability']['crb_vs_nls_spearman_rho']:.3f} (p={p_s1:.4f})")
    print(f"S5 DoE trace reduction={doe_var_reduction_pct:.1f}% (top L={doe_ranked[0].design.nominal_cavity_um}, exc={doe_ranked[0].design.excitations_nm}, PSF={doe_ranked[0].design.spec_psf_fwhm_nm})")
    print("\nS2 Primary Condition (Level 1, 25% data, 5-seed mean NRMSE):")
    for s_name in solver_names:
        m = curve_results["25pct"][s_name]["level1_interp"]
        p = m["per_latent"]
        print(
            f"  {s_name:18s} | mean={m['mean_nrmse']:.4f} | nL+z4={m['primary_nl_z4_nrmse']:.4f} | "
            f"nL={p['nL_um']['nrmse']:.3f} z1={p['z1_visc']['nrmse']:.3f} z2={p['z2_bound']['nrmse']:.3f} z4={p['z4_scat']['nrmse']:.3f}"
        )


if __name__ == "__main__":
    main()
