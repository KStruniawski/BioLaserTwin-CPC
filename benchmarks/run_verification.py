"""Quantitative Verification & Validation (V&V) suite for BioLaserTwin.

Runs:
1. Analytical passive-cavity FSR verification against lambda^2 / (2 n_g L) across L in [15, 45] um.
2. Analytical Airy finesse and FWHM linewidth verification against pi*sqrt(R)/(1-R) across R.
3. Oversampling convergence of 1D pixel-integrated spectra (oversample 1..32 vs 64x reference).
4. Literature-grounded physical regime validation against published ThT lasing studies:
   - Hanczyc & Fita 2021 (ACS Photonics, 10.1021/acsphotonics.1c00082)
   - Hanczyc 2024 (J. Phys. Chem. Lett., 10.1021/acs.jpclett.4c01709)
   - Rusakov et al. 2024 (J. Phys. Chem. Lett., 10.1021/acs.jpclett.4c00699)
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
from scipy.signal import find_peaks

from biolaser_twin.constants import LATENT_NAMES, LATENT_PRIOR_SCALES, NominalOpticalConstants
from biolaser_twin.extractor import extract_sweep_features
from biolaser_twin.identifiability import DesignConfig, reduced_forward_observables
from biolaser_twin.observations import (
    ObservationRecord,
    build_observation_record,
    calibrate_nominal_constants_from_dev,
    estimate_noise_cov,
)
from biolaser_twin.metrics import normalized_rmse
from biolaser_twin.simulator import (
    FabryPerotConfig,
    FabryPerotTruthSimulator,
    LatentProxyVector,
    map_latents_to_truth_state,
)
from biolaser_twin.simulator.photophysics import effective_gain_and_emission
from biolaser_twin.solvers import PhotonicPIELMSolver, PlainELMSolver


def verify_passive_fsr() -> dict:
    """Compare simulated Airy peak spacing against analytical lambda^2 / (2 * n_g * L)."""
    lengths_um = [15.0, 20.0, 30.0, 40.0, 45.0]
    n0 = 1.345
    records = []
    for level in (0, 1):
        n_g = n0 + (0.0194 if level >= 1 else 0.0)
        for l_um in lengths_um:
            cfg = FabryPerotConfig(
                cavity_length_um=l_um,
                refractive_index=n0,
                mirror_reflectivity=0.985,
                lambda_min_nm=480.0,
                lambda_max_nm=490.0,
                n_spectral_px=8192,
                level=level,
            )
            sim = FabryPerotTruthSimulator(cfg)
            wl = np.linspace(480.0, 490.0, 32768)
            trans = sim.airy_transmission(wl, state=None)
            peaks, _ = find_peaks(trans, height=0.5, distance=20)
            # Wavenumber-space FSR around 485 nm
            inv_lam = 1.0 / wl[peaks]
            fsr_sim = float((485.0**2) * np.median(np.abs(np.diff(inv_lam))))
            fsr_ana = float((485.0**2) / (2.0 * n_g * l_um * 1000.0))
            rel_err = abs(fsr_sim - fsr_ana) / fsr_ana
            records.append(
                {
                    "level": level,
                    "L_um": l_um,
                    "fsr_sim_nm": fsr_sim,
                    "fsr_analytical_nm": fsr_ana,
                    "rel_error": rel_err,
                }
            )
    max_rel_err = max(r["rel_error"] for r in records)
    return {"max_rel_error": max_rel_err, "records": records}


def verify_finesse_and_linewidth() -> dict:
    """Compare simulated passive Airy FWHM and finesse against analytical pi*sqrt(R)/(1-R)."""
    reflectivities = [0.90, 0.95, 0.98, 0.988, 0.995]
    l_um = 20.0
    n0 = 1.345
    fsr_ana = (485.0**2) / (2.0 * n0 * l_um * 1000.0)
    records = []
    for r_val in reflectivities:
        cfg = FabryPerotConfig(
            cavity_length_um=l_um,
            refractive_index=n0,
            mirror_reflectivity=r_val,
            level=0,
        )
        sim = FabryPerotTruthSimulator(cfg)
        # Fine grid centered around one resonance near 485 nm
        wl = np.linspace(482.0, 488.0, 200000)
        trans = sim.airy_transmission(wl, state=None)
        peaks, _ = find_peaks(trans, height=0.8)
        pk = peaks[len(peaks) // 2]
        half_max = 0.5 * trans[pk]
        left = pk
        while left > 0 and trans[left] > half_max:
            left -= 1
        right = pk
        while right < len(trans) - 1 and trans[right] > half_max:
            right += 1
        fwhm_sim = float(wl[right] - wl[left])
        finesse_ana = float(np.pi * np.sqrt(r_val) / (1.0 - r_val))
        fwhm_ana = float(fsr_ana / finesse_ana)
        finesse_sim = float(fsr_ana / fwhm_sim)
        records.append(
            {
                "R": r_val,
                "finesse_sim": finesse_sim,
                "finesse_analytical": finesse_ana,
                "fwhm_sim_nm": fwhm_sim,
                "fwhm_analytical_nm": fwhm_ana,
                "rel_error": abs(fwhm_sim - fwhm_ana) / fwhm_ana,
            }
        )
    max_rel_err = max(r["rel_error"] for r in records)
    return {"max_rel_error": max_rel_err, "records": records}


def verify_oversampling_convergence() -> dict:
    """Measure 1D pixel-integrated spectrum L2 error across oversample=1..32 vs 64x/128x."""
    state = map_latents_to_truth_state(LatentProxyVector(0.5, 0.6, 1.345, 1.5, 30.0, 10.0))
    ovs = [1, 2, 4, 8, 16, 32, 64]
    specs = {}
    for ov in ovs:
        cfg = FabryPerotConfig(
            cavity_length_um=30.0,
            refractive_index=1.345,
            n_spectral_px=1024,
            n_spatial_px=16,
            spec_psf_fwhm_nm=0.10,
            oversample=ov,
            level=0,
            seed=42,
        )
        sim = FabryPerotTruthSimulator(cfg)
        frame = sim.render_2d_frame(state, excitation_nm=440.0, pump_energy_uj=30.0)
        specs[ov] = frame[8, :]

    ref = specs[64]
    ref_norm = float(np.linalg.norm(ref))
    records = []
    for ov in [1, 2, 4, 8, 16, 32]:
        rel_l2 = float(np.linalg.norm(specs[ov] - ref) / max(ref_norm, 1e-12))
        records.append({"oversample": ov, "rel_l2_vs_64x": rel_l2})

    # High-finesse / long-cavity corner case (L=40 um, n=1.365, z4=0.2 cm^-1) vs 128x
    state_hf = map_latents_to_truth_state(LatentProxyVector(0.5, 0.6, 1.365, 0.2, 40.0, 10.0))
    specs_hf = {}
    for ov in [16, 32, 64, 128]:
        cfg_hf = FabryPerotConfig(
            cavity_length_um=40.0, refractive_index=1.365,
            n_spectral_px=1024, n_spatial_px=16, spec_psf_fwhm_nm=0.10,
            oversample=ov, level=0, seed=42,
        )
        specs_hf[ov] = FabryPerotTruthSimulator(cfg_hf).render_2d_frame(state_hf, 440.0, 30.0)[8, :]
    ref_hf = specs_hf[128]
    ref_hf_norm = float(np.linalg.norm(ref_hf))
    hf_records = [
        {"oversample": ov, "rel_l2_vs_128x": float(np.linalg.norm(specs_hf[ov] - ref_hf) / max(ref_hf_norm, 1e-12))}
        for ov in [16, 32, 64]
    ]

    # Downstream feature-extraction sensitivity on L=40 um high-finesse state (16x vs 32x vs 64x)
    energies = np.linspace(2.0, 48.0, 8)
    feat_sens = []
    for ov in (16, 32, 64):
        sim_s = FabryPerotTruthSimulator(
            FabryPerotConfig(
                cavity_length_um=40.0, refractive_index=1.365,
                n_spectral_px=1024, n_spatial_px=16, spec_psf_fwhm_nm=0.10,
                oversample=ov, level=0, seed=42,
            )
        )
        b440 = sim_s.render_pump_sweep(state_hf, 440.0, energies)
        f440 = extract_sweep_features(
            b440.frames, b440.wavelengths_nm, b440.pump_energies_uj,
            spec_psf_fwhm_nm=b440.spec_psf_fwhm_nm, dark_frames=b440.dark_frames,
            nominal_cavity_um=b440.nominal_cavity_um,
        )
        feat_sens.append({
            "oversample": ov,
            "fsr_nm": f440.mean_fsr_nm,
            "e_th_440_uj": f440.threshold_energy_uj,
            "com_440_nm": f440.center_of_mass_nm,
            "off_peak_baseline": f440.off_peak_baseline,
        })

    return {"records": records, "high_finesse_L40_records": hf_records, "feature_sensitivity_L40": feat_sens}


def verify_bae_pielm_and_s1_per_latent() -> dict:
    """Evaluate Bayesian Approximation Error (BAE) PhotonicPIELM and per-latent S1 CRB correlations."""
    from scipy.stats import spearmanr

    # 1. Per-latent S1 Spearman correlations from stage_a_full.json
    stage_a_path = Path(__file__).resolve().parent / "results" / "stage_a_full.json"
    s1_per_latent = {}
    if stage_a_path.exists():
        sa = json.loads(stage_a_path.read_text())
        pairs = sa["s1_identifiability"]["pairs"]
        for lat in LATENT_NAMES:
            sub = [p for p in pairs if p["latent"] == lat]
            r_lat, p_lat = spearmanr([p["crb_rel_sd"] for p in sub], [p["emp_nls_rel_sd"] for p in sub])
            s1_per_latent[lat] = {"n": len(sub), "rho": float(r_lat), "p": float(p_lat)}

    # 2. BAE-corrected PhotonicPIELM on Level 1 (25% data = 75 train, 5 seeds)
    def _gen_l1(n_exp: int, seed: int):
        rng = np.random.default_rng(seed)
        energies = np.linspace(2.0, 48.0, 8)
        feats, targets, recs = [], [], []
        for idx in range(n_exp):
            l_um = float(rng.uniform(20.0, 40.0))
            n_val = float(rng.uniform(1.335, 1.365))
            z1, z2, z4 = float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.2, 6.0))
            sim = FabryPerotTruthSimulator(
                FabryPerotConfig(cavity_length_um=l_um, refractive_index=n_val, n_spectral_px=1024, n_spatial_px=16, spec_psf_fwhm_nm=0.10, oversample=16, level=1, seed=seed * 10000 + idx)
            )
            st = map_latents_to_truth_state(LatentProxyVector(z1, z2, n_val, z4, l_um, 10.0))
            b405 = sim.render_pump_sweep(st, 405.0, energies)
            b440 = sim.render_pump_sweep(st, 440.0, energies)
            f405 = extract_sweep_features(b405.frames, b405.wavelengths_nm, b405.pump_energies_uj, spec_psf_fwhm_nm=0.10, dark_frames=b405.dark_frames, nominal_cavity_um=b405.nominal_cavity_um)
            f440 = extract_sweep_features(b440.frames, b440.wavelengths_nm, b440.pump_energies_uj, spec_psf_fwhm_nm=0.10, dark_frames=b440.dark_frames, nominal_cavity_um=b440.nominal_cavity_um)
            d_cfg = DesignConfig(nominal_cavity_um=b440.nominal_cavity_um, spec_psf_fwhm_nm=0.10, excitations_nm=(405.0, 440.0))
            rec = build_observation_record(f405, f440, d_cfg, max_pump_uj=48.0)
            recs.append(rec)
            x_tab = rec.values.copy()
            if not rec.mask[0]:
                x_tab[0] = (485.0**2) / (2.0 * 1.34 * d_cfg.nominal_cavity_um * 1000.0)
            if not rec.mask[1]:
                x_tab[1] = 48.0 * 1.1
            if not rec.mask[2]:
                x_tab[2] = 48.0 * 1.1
            feats.append(x_tab)
            targets.append([n_val * l_um, z1, z2, z4])
        return np.asarray(feats, dtype=float), np.asarray(targets, dtype=float), recs

    x_dev, y_dev, recs_dev = _gen_l1(16, seed=1)
    cal_consts = calibrate_nominal_constants_from_dev(recs_dev, y_dev)
    # Estimate masked BAE mean discrepancy & total discrepancy+noise covariance from dev (no double-counting)
    diffs = np.full((len(recs_dev), 6), np.nan, dtype=float)
    for idx_d, (rec, th) in enumerate(zip(recs_dev, y_dev)):
        y_nom = reduced_forward_observables(th, rec.design, cal_consts)
        diffs[idx_d, rec.mask] = rec.values[rec.mask] - y_nom[rec.mask]
    bae_mean = np.nanmean(diffs, axis=0)
    bae_mean = np.where(np.isfinite(bae_mean), bae_mean, 0.0)
    # Empirical variance per observable over valid dev entries (avoids zero-filling bias and double-counting)
    bae_vars = np.zeros(6, dtype=float)
    for j in range(6):
        valid = diffs[np.isfinite(diffs[:, j]), j]
        bae_vars[j] = float(np.var(valid, ddof=1)) if len(valid) >= 3 else float(np.var(valid)) if len(valid) > 0 else 1.0
    bae_cov = np.diag(np.maximum(bae_vars, 1e-6))

    x_tr, y_tr, _ = _gen_l1(75, seed=710)
    x_col, _, recs_col = _gen_l1(60, seed=711)
    x_va, y_va, _ = _gen_l1(60, seed=712)
    x_te, y_te, _ = _gen_l1(100, seed=713)

    # Bias-corrected collocation records
    recs_col_bae = [
        ObservationRecord(
            values=np.where(r.mask, r.values - bae_mean, r.values),
            mask=r.mask, design=r.design, lasing_405=r.lasing_405, lasing_440=r.lasing_440,
        )
        for r in recs_col
    ]

    plain_scores, bae_scores, bae_lams = [], [], []
    for sd in (1, 2, 3, 4, 5):
        mp = PlainELMSolver(hidden_dim=48, n_ensemble=3, seed=sd).fit(x_tr, y_tr)
        nr_p = normalized_rmse(y_te, mp.predict(x_te), LATENT_PRIOR_SCALES)
        plain_scores.append(float(0.5 * (nr_p[0] + nr_p[3])))

        best_l, best_v = 0.0, np.inf
        for lam in (0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5):
            mb = PhotonicPIELMSolver(
                hidden_dim=48, phys_lambda=lam, n_ensemble=3, n_gn_iters=3,
                constants=cal_consts, noise_cov=bae_cov, seed=sd,
            ).fit(x_tr, y_tr, collocation_features=x_col, collocation_records=recs_col_bae)
            verr = float(np.mean(normalized_rmse(y_va, mb.predict(x_va), LATENT_PRIOR_SCALES)))
            if verr < best_v:
                best_v, best_l = verr, lam
        mb_best = PhotonicPIELMSolver(
            hidden_dim=48, phys_lambda=best_l, n_ensemble=3, n_gn_iters=3,
            constants=cal_consts, noise_cov=bae_cov, seed=sd,
        ).fit(x_tr, y_tr, collocation_features=x_col, collocation_records=recs_col_bae)
        nr_b = normalized_rmse(y_te, mb_best.predict(x_te), LATENT_PRIOR_SCALES)
        bae_scores.append(float(0.5 * (nr_b[0] + nr_b[3])))
        bae_lams.append(best_l)

    return {
        "s1_per_latent": s1_per_latent,
        "bae_pielm_25pct": {
            "plain_elm_nl_z4_mean": float(np.mean(plain_scores)),
            "plain_elm_nl_z4_std": float(np.std(plain_scores)),
            "bae_pielm_nl_z4_mean": float(np.mean(bae_scores)),
            "bae_pielm_nl_z4_std": float(np.std(bae_scores)),
            "selected_lambdas": bae_lams,
        },
    }


def verify_matched_physics_ladder() -> dict:
    """Positive control y = f_nom(theta) + eps and controlled misspecification ladder (5 seeds)."""
    consts = NominalOpticalConstants()
    design = DesignConfig(nominal_cavity_um=30.0, spec_psf_fwhm_nm=0.10, excitations_nm=(405.0, 440.0))
    noise_stds = np.array([0.04, 0.8, 0.8, 0.15, 0.15, 0.005], dtype=float)
    noise_cov = np.diag(noise_stds**2)

    def _sample_thetas(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.column_stack([
            rng.uniform(27.0, 54.0, n),
            rng.uniform(0.15, 0.85, n),
            rng.uniform(0.15, 0.85, n),
            rng.uniform(0.2, 6.0, n),
        ])

    def _obs_with_mismatch(thetas: np.ndarray, alpha_mis: float, rng: np.random.Generator) -> tuple[np.ndarray, list[ObservationRecord]]:
        obs = np.zeros((len(thetas), 6), dtype=float)
        recs = []
        for i, th in enumerate(thetas):
            y_nom = reduced_forward_observables(th, design, consts)
            # Controlled systematic bias proportional to alpha_mis (mimicking unmodeled GT/viscosity/reabs)
            bias = alpha_mis * np.array([0.04 * y_nom[0], 0.35 * y_nom[1] * (1.0 - th[1]), 0.15 * y_nom[2], 1.2 * (1.0 - th[2]), 0.5, 0.002 * th[3]])
            y_obs = y_nom + bias + rng.normal(0.0, noise_stds)
            obs[i] = y_obs
            recs.append(ObservationRecord(values=y_obs, mask=np.ones(6, dtype=bool), design=design, lasing_405=True, lasing_440=True))
        return obs, recs

    rng_data = np.random.default_rng(2026)
    th_tr = _sample_thetas(rng_data, 30)
    th_col = _sample_thetas(rng_data, 60)
    th_te = _sample_thetas(rng_data, 100)

    ladder = []
    for alpha_mis in (0.0, 0.25, 0.5, 1.0):
        x_tr, _ = _obs_with_mismatch(th_tr, alpha_mis, rng_data)
        x_col, recs_col = _obs_with_mismatch(th_col, alpha_mis, rng_data)
        x_te, _ = _obs_with_mismatch(th_te, alpha_mis, rng_data)

        plain_errs, pielm_errs = [], []
        for sd in (1, 2, 3, 4, 5):
            m_plain = PlainELMSolver(hidden_dim=48, n_ensemble=3, seed=sd).fit(x_tr, th_tr)
            m_pielm = PhotonicPIELMSolver(
                hidden_dim=48, phys_lambda=0.05, n_ensemble=3, n_gn_iters=5,
                constants=consts, noise_cov=noise_cov, seed=sd,
            ).fit(x_tr, th_tr, collocation_features=x_col, collocation_records=recs_col)
            plain_errs.append(float(np.mean(normalized_rmse(th_te, m_plain.predict(x_te), LATENT_PRIOR_SCALES))))
            pielm_errs.append(float(np.mean(normalized_rmse(th_te, m_pielm.predict(x_te), LATENT_PRIOR_SCALES))))
        ladder.append({
            "alpha_mis": alpha_mis,
            "plain_elm_mean": float(np.mean(plain_errs)),
            "plain_elm_std": float(np.std(plain_errs)),
            "photonic_pielm_mean": float(np.mean(pielm_errs)),
            "photonic_pielm_std": float(np.std(pielm_errs)),
        })
    return {"ladder": ladder}


def validate_literature_regimes() -> dict:
    """Evaluate threshold & spectral trends across viscosity, binding, and scattering regimes."""
    wl = np.linspace(465.0, 505.0, 1024)

    # 1. Viscosity sensitivity at 405 nm vs 440 nm (cf. Rusakov et al. 2024, 10.1021/acs.jpclett.4c00699)
    visc_grid = np.linspace(0.0, 1.0, 11)
    visc_records = []
    for z1 in visc_grid:
        st = map_latents_to_truth_state(LatentProxyVector(float(z1), 0.25, 1.345, 1.0, 30.0, 10.0))
        _, _, eth_405 = effective_gain_and_emission(st, 405.0, 20.0, wl)
        _, _, eth_440 = effective_gain_and_emission(st, 440.0, 20.0, wl)
        visc_records.append(
            {
                "z1": float(z1),
                "viscosity_cp": st.microviscosity_cp,
                "e_th_405_uj": eth_405,
                "e_th_440_uj": eth_440,
            }
        )

    # 2. Aggregation trajectory proxy: monomer -> oligomer (high bound, low scat) -> fibril (high scat)
    # (cf. Hanczyc & Fita 2021, 10.1021/acsphotonics.1c00082; Hanczyc 2024, 10.1021/acs.jpclett.4c01709)
    stages = [
        ("Monomer", 0.05, 0.05, 0.5),
        ("Early Oligomer", 0.45, 0.55, 1.0),
        ("Protofibril", 0.65, 0.80, 3.0),
        ("Mature Fibril", 0.70, 0.85, 8.0),
    ]
    agg_records = []
    for label, z1, z2, z4 in stages:
        st = map_latents_to_truth_state(LatentProxyVector(z1, z2, 1.345, z4, 30.0, 10.0))
        _, _, eth_430 = effective_gain_and_emission(st, 430.0, 20.0, wl)
        _, _, eth_445 = effective_gain_and_emission(st, 445.0, 20.0, wl)
        agg_records.append(
            {
                "stage": label,
                "z1": z1,
                "z2": z2,
                "z4_cm1": z4,
                "e_th_430_uj": eth_430,
                "e_th_445_uj": eth_445,
            }
        )

    return {"viscosity_sweep": visc_records, "aggregation_stages": agg_records}


def main() -> None:
    t0 = time.perf_counter()
    fsr_res = verify_passive_fsr()
    fin_res = verify_finesse_and_linewidth()
    ov_res = verify_oversampling_convergence()
    lit_res = validate_literature_regimes()
    mis_res = verify_matched_physics_ladder()
    bae_s1_res = verify_bae_pielm_and_s1_per_latent()

    out = {
        "passive_fsr": fsr_res,
        "finesse_linewidth": fin_res,
        "oversampling_convergence": ov_res,
        "literature_regimes": lit_res,
        "matched_physics_ladder": mis_res,
        "bae_and_s1_per_latent": bae_s1_res,
        "runtime_s": float(time.perf_counter() - t0),
    }
    out_path = Path(__file__).resolve().parent / "results" / "verification.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Verification complete in {out['runtime_s']:.2f}s -> {out_path}")
    print(f"  Max passive FSR rel error: {fsr_res['max_rel_error']:.3e}")
    print(f"  Max Airy FWHM rel error:   {fin_res['max_rel_error']:.3e}")
    for r in ov_res["records"]:
        print(f"  Oversample {r['oversample']:2d}x -> rel L2 vs 64x: {r['rel_l2_vs_64x']:.3e}")


if __name__ == "__main__":
    main()
