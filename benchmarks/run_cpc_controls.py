"""Paired BAE vs TabularMLP control, L_nom metadata control, and exact S1 permutation p-values."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from biolaser_twin.extractor import extract_sweep_features
from biolaser_twin.identifiability import DesignConfig, reduced_forward_observables
from biolaser_twin.metrics import normalized_rmse
from biolaser_twin.observations import ObservationRecord, build_observation_record, calibrate_nominal_constants_from_dev, estimate_noise_cov
from biolaser_twin.simulator import FabryPerotConfig, FabryPerotTruthSimulator, LatentProxyVector, map_latents_to_truth_state
from biolaser_twin.solvers import PhotonicPIELMSolver, PlainELMSolver, TabularMLPSolver
from run_stage_a_full import LATENT_NAMES, LATENT_PRIOR_SCALES


def _gen_l1(n_exp: int, seed: int, use_true_l_nom: bool = True):
    rng = np.random.default_rng(seed)
    energies = np.linspace(2.0, 48.0, 8)
    feats, targets, recs, l_noms = [], [], [], []
    for idx in range(n_exp):
        l_um = float(rng.uniform(20.0, 40.0))
        n_val = float(rng.uniform(1.335, 1.365))
        z1, z2, z4 = float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.2, 6.0))
        sim = FabryPerotTruthSimulator(
            FabryPerotConfig(
                cavity_length_um=l_um, refractive_index=n_val, n_spectral_px=1024,
                n_spatial_px=16, spec_psf_fwhm_nm=0.10, oversample=16, level=1,
                seed=seed * 10000 + idx,
            )
        )
        st = map_latents_to_truth_state(LatentProxyVector(z1, z2, n_val, z4, l_um, 10.0))
        b405 = sim.render_pump_sweep(st, 405.0, energies)
        b440 = sim.render_pump_sweep(st, 440.0, energies)
        l_hint = b440.nominal_cavity_um if use_true_l_nom else 30.0
        f405 = extract_sweep_features(
            b405.frames, b405.wavelengths_nm, b405.pump_energies_uj,
            spec_psf_fwhm_nm=0.10, dark_frames=b405.dark_frames, nominal_cavity_um=l_hint,
        )
        f440 = extract_sweep_features(
            b440.frames, b440.wavelengths_nm, b440.pump_energies_uj,
            spec_psf_fwhm_nm=0.10, dark_frames=b440.dark_frames, nominal_cavity_um=l_hint,
        )
        d_cfg = DesignConfig(nominal_cavity_um=l_hint, spec_psf_fwhm_nm=0.10, excitations_nm=(405.0, 440.0))
        rec = build_observation_record(f405, f440, d_cfg, max_pump_uj=48.0)
        recs.append(rec)
        l_noms.append(float(b440.nominal_cavity_um))
        x_tab = rec.values.copy()
        if not rec.mask[0]:
            x_tab[0] = (485.0**2) / (2.0 * 1.34 * l_hint * 1000.0)
        if not rec.mask[1]:
            x_tab[1] = 48.0 * 1.1
        if not rec.mask[2]:
            x_tab[2] = 48.0 * 1.1
        feats.append(x_tab)
        targets.append([n_val * l_um, z1, z2, z4])
    return np.asarray(feats, dtype=float), np.asarray(targets, dtype=float), recs, np.asarray(l_noms, dtype=float)


def _exact_spearman_perm_p(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr
    obs = abs(float(spearmanr(x, y).statistic))
    n = len(x)
    ge = 0
    total = 0
    for perm in itertools.permutations(y):
        total += 1
        if abs(float(spearmanr(x, perm).statistic)) >= obs - 1e-12:
            ge += 1
    return float(ge / total)


def run_paired_bae_control() -> dict:
    x_dev, y_dev, recs_dev, _ = _gen_l1(16, seed=1)
    cal_consts = calibrate_nominal_constants_from_dev(recs_dev, y_dev)

    diffs = np.full((len(recs_dev), 6), np.nan, dtype=float)
    for idx_d, (rec, th) in enumerate(zip(recs_dev, y_dev)):
        y_nom = reduced_forward_observables(th, rec.design, cal_consts)
        diffs[idx_d, rec.mask] = rec.values[rec.mask] - y_nom[rec.mask]
    bae_mean = np.where(np.isfinite(np.nanmean(diffs, axis=0)), np.nanmean(diffs, axis=0), 0.0)
    bae_vars = np.zeros(6, dtype=float)
    for j in range(6):
        valid = diffs[np.isfinite(diffs[:, j]), j]
        bae_vars[j] = float(np.var(valid, ddof=1)) if len(valid) >= 3 else 1.0
    bae_cov = np.diag(np.maximum(bae_vars, 1e-6))

    x_tr, y_tr, _, _ = _gen_l1(75, seed=710)
    x_col, _, recs_col, _ = _gen_l1(60, seed=711)
    x_va, y_va, _, _ = _gen_l1(60, seed=712)
    x_te, y_te, _, _ = _gen_l1(100, seed=713)

    x_tr91 = np.vstack([x_tr, x_dev])
    y_tr91 = np.vstack([y_tr, y_dev])

    base_noise_cov = estimate_noise_cov(recs_dev)
    recs_col_bae = [
        ObservationRecord(
            values=np.where(r.mask, r.values - bae_mean, np.nan),
            mask=r.mask,
            design=r.design,
            lasing_405=r.lasing_405,
            lasing_440=r.lasing_440,
        )
        for r in recs_col
    ]
    lam_grid = [0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5]

    out = {}
    for label, fn in [
        ("PlainELM_75", lambda s: PlainELMSolver(hidden_dim=48, n_ensemble=3, seed=s).fit(x_tr, y_tr)),
        ("PlainELM_91", lambda s: PlainELMSolver(hidden_dim=48, n_ensemble=3, seed=s).fit(x_tr91, y_tr91)),
        ("TabularMLP_75", lambda s: TabularMLPSolver(hidden_dim=64, n_ensemble=3, epochs=120, seed=s).fit(x_tr, y_tr, val_features=x_va, val_targets=y_va)),
        ("TabularMLP_91", lambda s: TabularMLPSolver(hidden_dim=64, n_ensemble=3, epochs=120, seed=s).fit(x_tr91, y_tr91, val_features=x_va, val_targets=y_va)),
    ]:
        nlz4, all4 = [], []
        for s in range(1, 6):
            m = fn(s)
            mu, _ = m.predict_with_uncertainty(x_te)
            nr = normalized_rmse(y_te, mu, ref_scale=LATENT_PRIOR_SCALES)
            nlz4.append(float(0.5 * (nr[0] + nr[3])))
            all4.append(float(np.mean(nr)))
        out[label] = {"nl_z4_mean": float(np.mean(nlz4)), "nl_z4_std": float(np.std(nlz4)), "all4_mean": float(np.mean(all4))}

    for label, col_recs, cov_mat in [
        ("PhotonicPIELM_uncorr_75", recs_col, base_noise_cov),
        ("BAE_mean_only_75", recs_col_bae, base_noise_cov),
        ("BAE_PhotonicPIELM_75", recs_col_bae, bae_cov),
    ]:
        nlz4, all4, lams = [], [], []
        for s in range(1, 6):
            best_m, best_l, best_v = None, 0.0, float("inf")
            for lam in lam_grid:
                m = PhotonicPIELMSolver(
                    hidden_dim=48, n_ensemble=3, phys_lambda=lam, n_gn_iters=3,
                    constants=cal_consts, noise_cov=cov_mat, seed=s,
                )
                m.fit(x_tr, y_tr, collocation_features=x_col, collocation_records=col_recs)
                mv, _ = m.predict_with_uncertainty(x_va)
                val_nr = normalized_rmse(y_va, mv, ref_scale=LATENT_PRIOR_SCALES)
                val_err = float(0.5 * (val_nr[0] + val_nr[3]))
                if val_err < best_v:
                    best_v, best_m, best_l = val_err, m, lam
            mu, _ = best_m.predict_with_uncertainty(x_te)
            nr = normalized_rmse(y_te, mu, ref_scale=LATENT_PRIOR_SCALES)
            nlz4.append(float(0.5 * (nr[0] + nr[3])))
            all4.append(float(np.mean(nr)))
            lams.append(float(best_l))
        out[label] = {
            "nl_z4_mean": float(np.mean(nlz4)), "nl_z4_std": float(np.std(nlz4)),
            "all4_mean": float(np.mean(all4)), "selected_lambdas": lams,
        }
    return out


def run_metadata_control() -> dict:
    # Check L_nom metadata-only baseline vs TabularMLP with/without L_nom hint on L~U[20,40]
    x_tr_h, y_tr, _, _ = _gen_l1(150, seed=310, use_true_l_nom=True)
    x_va_h, y_va, _, _ = _gen_l1(50, seed=311, use_true_l_nom=True)
    x_te_h, y_te, _, l_noms_te = _gen_l1(200, seed=312, use_true_l_nom=True)

    x_tr_nh, _, _, _ = _gen_l1(150, seed=310, use_true_l_nom=False)
    x_va_nh, _, _, _ = _gen_l1(50, seed=311, use_true_l_nom=False)
    x_te_nh, _, _, _ = _gen_l1(200, seed=312, use_true_l_nom=False)

    # Trivial metadata-only estimator: nL = 1.35 * L_nom, prior mean for z1,z2,z4
    pred_meta = np.tile(np.mean(y_tr, axis=0), (len(y_te), 1))
    pred_meta[:, 0] = 1.35 * l_noms_te
    nr_meta = normalized_rmse(y_te, pred_meta, ref_scale=LATENT_PRIOR_SCALES)

    with_hint, no_hint = [], []
    for s in range(5):
        m1 = TabularMLPSolver(hidden_dim=64, n_ensemble=3, epochs=120, seed=s).fit(x_tr_h, y_tr, val_features=x_va_h, val_targets=y_va)
        nr1 = normalized_rmse(y_te, m1.predict_with_uncertainty(x_te_h)[0], ref_scale=LATENT_PRIOR_SCALES)
        with_hint.append((float(nr1[0]), float(nr1[3]), float(0.5 * (nr1[0] + nr1[3]))))

        m2 = TabularMLPSolver(hidden_dim=64, n_ensemble=3, epochs=120, seed=s).fit(x_tr_nh, y_tr, val_features=x_va_nh, val_targets=y_va)
        nr2 = normalized_rmse(y_te, m2.predict_with_uncertainty(x_te_nh)[0], ref_scale=LATENT_PRIOR_SCALES)
        no_hint.append((float(nr2[0]), float(nr2[3]), float(0.5 * (nr2[0] + nr2[3]))))

    return {
        "MetadataOnly_1.35_Lnom": {"nL_nrmse": float(nr_meta[0]), "z4_nrmse": float(nr_meta[3]), "nl_z4_nrmse": float(0.5 * (nr_meta[0] + nr_meta[3]))},
        "TabularMLP_with_Lnom_hint": {
            "nL_mean": float(np.mean([r[0] for r in with_hint])),
            "z4_mean": float(np.mean([r[1] for r in with_hint])),
            "nl_z4_mean": float(np.mean([r[2] for r in with_hint])),
            "nl_z4_std": float(np.std([r[2] for r in with_hint])),
        },
        "TabularMLP_no_Lnom_hint": {
            "nL_mean": float(np.mean([r[0] for r in no_hint])),
            "z4_mean": float(np.mean([r[1] for r in no_hint])),
            "nl_z4_mean": float(np.mean([r[2] for r in no_hint])),
            "nl_z4_std": float(np.std([r[2] for r in no_hint])),
        },
    }


def main() -> None:
    sa = json.loads((ROOT / "benchmarks" / "results" / "stage_a_full.json").read_text())
    pairs = sa["s1_identifiability"]["pairs"]
    exact_s1 = {}
    for lat in LATENT_NAMES:
        sub = [p for p in pairs if p["latent"] == lat]
        crb = [p["crb_rel_sd"] for p in sub]
        emp = [p["emp_nls_rel_sd"] for p in sub]
        from scipy.stats import spearmanr
        rho = float(spearmanr(crb, emp).statistic)
        exact_s1[lat] = {"n": len(sub), "rho": rho, "exact_two_sided_p": _exact_spearman_perm_p(crb, emp)}

    paired_bae = run_paired_bae_control()
    meta_ctrl = run_metadata_control()

    out = {"exact_s1_permutation_p": exact_s1, "paired_bae_control": paired_bae, "metadata_Lnom_control": meta_ctrl}
    out_path = ROOT / "benchmarks" / "results" / "cpc_controls.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("Paired BAE control:", json.dumps(paired_bae, indent=2))
    print("Metadata control:", json.dumps(meta_ctrl, indent=2))
    print("Exact S1 p-values:", json.dumps(exact_s1, indent=2))


if __name__ == "__main__":
    main()
