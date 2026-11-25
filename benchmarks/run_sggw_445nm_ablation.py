"""SGGW Hardware-Aligned Ablation: Single-Excitation 445 nm (470-530 nm) vs Dual 405+445 nm.

Directly evaluates the impact of SGGW's confirmed 445 nm ns-laser / 470-530 nm emission
window (P. Hanczyc 2026-11-16) relative to dual-excitation (405 + 445 nm) on latent recovery.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np

from biolaser_twin.calibration import SplitConformalCalibrator
from biolaser_twin.constants import LATENT_NAMES, LATENT_PRIOR_SCALES
from biolaser_twin.extractor import extract_sweep_features
from biolaser_twin.identifiability import DesignConfig, evaluate_design
from biolaser_twin.metrics import normalized_rmse
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
    MeanOnlySolver,
    PhotonicPIELMSolver,
    PlainELMSolver,
    TabularMLPSolver,
)


def _generate_sggw_split(
    n_exp: int,
    dual_exc: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ObservationRecord]]:
    rng = np.random.default_rng(seed)
    energies = np.linspace(2.0, 48.0, 8)
    feats, targets, frames, recs = [], [], [], []
    exc_tuple = (405.0, 445.0) if dual_exc else (445.0,)

    for idx in range(n_exp):
        l_um = float(rng.uniform(20.0, 40.0))
        n_val = float(rng.uniform(1.335, 1.365))
        z1 = float(rng.uniform(0.15, 0.85))
        z2 = float(rng.uniform(0.15, 0.85))
        z4 = float(rng.uniform(0.2, 6.0))

        cfg = FabryPerotConfig(
            cavity_length_um=l_um,
            refractive_index=n_val,
            lambda_min_nm=470.0,
            lambda_max_nm=530.0,
            n_spectral_px=1024,
            n_spatial_px=16,
            spec_psf_fwhm_nm=0.10,
            oversample=16,
            level=1,
            seed=seed * 10000 + idx,
        )
        sim = FabryPerotTruthSimulator(cfg)
        state = map_latents_to_truth_state(LatentProxyVector(z1, z2, n_val, z4, l_um, tht_conc_mm=10.0))

        b445 = sim.render_pump_sweep(state, 445.0, energies)
        f445 = extract_sweep_features(
            b445.frames, b445.wavelengths_nm, b445.pump_energies_uj,
            spec_psf_fwhm_nm=b445.spec_psf_fwhm_nm, dark_frames=b445.dark_frames,
            nominal_cavity_um=b445.nominal_cavity_um,
        )
        if dual_exc:
            b405 = sim.render_pump_sweep(state, 405.0, energies)
            f405 = extract_sweep_features(
                b405.frames, b405.wavelengths_nm, b405.pump_energies_uj,
                spec_psf_fwhm_nm=b405.spec_psf_fwhm_nm, dark_frames=b405.dark_frames,
                nominal_cavity_um=b405.nominal_cavity_um,
            )
            fr_stack = np.stack([b405.frames, b445.frames], axis=0)
        else:
            # Shot-budget matched: 2nd 445 nm sweep averaged with 1st (2 total sweeps vs 1+1 in dual)
            b445_2 = sim.render_pump_sweep(state, 445.0, energies)
            avg_frames = 0.5 * (b445.frames + b445_2.frames)
            avg_dark = 0.5 * (b445.dark_frames + b445_2.dark_frames)
            f445 = extract_sweep_features(
                avg_frames, b445.wavelengths_nm, b445.pump_energies_uj,
                spec_psf_fwhm_nm=b445.spec_psf_fwhm_nm, dark_frames=avg_dark,
                nominal_cavity_um=b445.nominal_cavity_um,
            )
            f405 = None
            fr_stack = avg_frames[None, ...]

        design = DesignConfig(nominal_cavity_um=b445.nominal_cavity_um, spec_psf_fwhm_nm=0.10, excitations_nm=exc_tuple)
        rec = build_observation_record(f405, f445, design, max_pump_uj=float(energies[-1]))
        recs.append(rec)

        x_tab = rec.values.copy()
        if not rec.mask[0]:
            x_tab[0] = (485.0**2) / (2.0 * 1.34 * design.nominal_cavity_um * 1000.0)
        if not rec.mask[1]:
            x_tab[1] = float(energies[-1]) * 1.1
        if not rec.mask[2]:
            x_tab[2] = float(energies[-1]) * 1.1
        feats.append(x_tab)
        targets.append([n_val * l_um, z1, z2, z4])
        frames.append(fr_stack)

    return (
        np.asarray(feats, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(frames, dtype=np.float32),
        recs,
    )


def _estimate_single_sweep_noise_cov(dual_exc: bool, n_repeats: int = 8, seed: int = 905) -> np.ndarray:
    """Estimate repeated-measurement noise covariance Sigma from a single fixed reference sample."""
    energies = np.linspace(2.0, 48.0, 8)
    sim = FabryPerotTruthSimulator(
        FabryPerotConfig(
            cavity_length_um=30.0, refractive_index=1.345,
            lambda_min_nm=470.0, lambda_max_nm=530.0,
            n_spectral_px=1024, n_spatial_px=16, spec_psf_fwhm_nm=0.10,
            oversample=16, level=1, seed=seed,
        )
    )
    st = map_latents_to_truth_state(LatentProxyVector(0.5, 0.5, 1.345, 2.0, 30.0, 10.0))
    exc_tuple = (405.0, 445.0) if dual_exc else (445.0,)
    d_cfg = DesignConfig(nominal_cavity_um=30.0, spec_psf_fwhm_nm=0.10, excitations_nm=exc_tuple)
    recs = []
    for _ in range(n_repeats):
        b445 = sim.render_pump_sweep(st, 445.0, energies)
        f445 = extract_sweep_features(
            b445.frames, b445.wavelengths_nm, b445.pump_energies_uj,
            spec_psf_fwhm_nm=b445.spec_psf_fwhm_nm, dark_frames=b445.dark_frames,
            nominal_cavity_um=b445.nominal_cavity_um,
        )
        f405 = None
        if dual_exc:
            b405 = sim.render_pump_sweep(st, 405.0, energies)
            f405 = extract_sweep_features(
                b405.frames, b405.wavelengths_nm, b405.pump_energies_uj,
                spec_psf_fwhm_nm=b405.spec_psf_fwhm_nm, dark_frames=b405.dark_frames,
                nominal_cavity_um=b405.nominal_cavity_um,
            )
        recs.append(build_observation_record(f405, f445, d_cfg, max_pump_uj=48.0))
    return estimate_noise_cov(recs, min_std=1e-3)


def main() -> None:
    t0 = time.perf_counter()
    print("=== SGGW 445 nm (470-530 nm) vs Dual 405+445 nm Ablation ===")
    results: dict[str, dict] = {}

    for mode_name, dual_exc in (("SGGW_445nm_only", False), ("Dual_405_445nm", True)):
        x_dev, y_dev, _, recs_dev = _generate_sggw_split(16, dual_exc=dual_exc, seed=201)
        cal_consts = calibrate_nominal_constants_from_dev(recs_dev, y_dev)
        cov = _estimate_single_sweep_noise_cov(dual_exc=dual_exc, n_repeats=8, seed=905)

        x_tr, y_tr, f_tr, _ = _generate_sggw_split(150, dual_exc=dual_exc, seed=210)
        x_va, y_va, f_va, _ = _generate_sggw_split(50, dual_exc=dual_exc, seed=212)
        x_te, y_te, f_te, recs_te = _generate_sggw_split(100, dual_exc=dual_exc, seed=213)

        # Theoretical FIM score (exact 445 nm wavelength)
        d_cfg = DesignConfig(
            nominal_cavity_um=30.0,
            spec_psf_fwhm_nm=0.10,
            excitations_nm=(405.0, 445.0) if dual_exc else (445.0,),
        )
        ref_grid = np.array([[35.0, 0.3, 0.3, 1.5], [45.0, 0.7, 0.7, 4.0]])
        fim_score = evaluate_design(d_cfg, ref_grid, noise_cov=cov, constants=cal_consts, measurement_budget_shots=2)

        seeds = [1, 2, 3, 4, 5]
        per_solver: dict[str, list[np.ndarray]] = {
            "MeanOnly": [], "PlainELM": [], "TabularMLP": [], "FrameCNN": [],
        }
        for sd in seeds:
            models = {
                "MeanOnly": MeanOnlySolver().fit(y_tr),
                "PlainELM": PlainELMSolver(hidden_dim=48, n_ensemble=3, seed=sd).fit(x_tr, y_tr),
                "TabularMLP": TabularMLPSolver(hidden_dim=64, epochs=80, n_ensemble=3, seed=sd).fit(x_tr, y_tr, x_va, y_va),
                "FrameCNN": FrameCNNSolver(epochs=50, lr=2e-3, seed=sd).fit(f_tr, y_tr, f_va, y_va),
            }
            for s_name, m in models.items():
                if s_name == "MeanOnly":
                    tm, _ = m.predict_with_uncertainty(len(y_te))
                elif s_name == "FrameCNN":
                    tm, _ = m.predict_with_uncertainty(f_te)
                else:
                    tm, _ = m.predict_with_uncertainty(x_te)
                per_solver[s_name].append(normalized_rmse(y_te, tm, LATENT_PRIOR_SCALES))

        nls = ClassicalNLSSolver(constants=cal_consts, noise_cov=cov).predict_records(recs_te)
        nls_nrmse = normalized_rmse(y_te, nls.means, LATENT_PRIOR_SCALES)

        summary = {
            "fim_a_opt_norm_trace": fim_score.a_opt_norm_trace,
            "fim_d_opt_logdet": fim_score.d_opt_logdet,
            "solvers": {
                s_name: {
                    "mean_nrmse": float(np.mean(runs)),
                    "std_nrmse": float(np.std([np.mean(r) for r in runs])),
                    "primary_nl_z4": float(np.mean([0.5 * (r[0] + r[3]) for r in runs])),
                    "primary_nl_z4_std": float(np.std([0.5 * (r[0] + r[3]) for r in runs])),
                    "per_latent": {lat: float(np.mean([r[i] for r in runs])) for i, lat in enumerate(LATENT_NAMES)},
                    "per_latent_std": {lat: float(np.std([r[i] for r in runs])) for i, lat in enumerate(LATENT_NAMES)},
                }
                for s_name, runs in per_solver.items()
            },
        }
        summary["solvers"]["ClassicalNLS"] = {
            "mean_nrmse": float(np.mean(nls_nrmse)),
            "primary_nl_z4": float(0.5 * (nls_nrmse[0] + nls_nrmse[3])),
            "per_latent": {lat: float(nls_nrmse[i]) for i, lat in enumerate(LATENT_NAMES)},
        }
        results[mode_name] = summary

    out = {"results": results, "runtime_s": float(time.perf_counter() - t0)}
    out_path = Path(__file__).resolve().parent / "results" / "sggw_445nm_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Completed in {out['runtime_s']:.1f}s -> {out_path}")
    for mode, res in results.items():
        print(f"\n[{mode}] (FIM A-opt={res['fim_a_opt_norm_trace']:.2f}):")
        for s_name, m in res["solvers"].items():
            p = m["per_latent"]
            print(
                f"  {s_name:12s} | mean={m['mean_nrmse']:.4f} | nL+z4={m['primary_nl_z4']:.4f} | "
                f"nL={p['nL_um']:.3f} z1={p['z1_visc']:.3f} z2={p['z2_bound']:.3f} z4={p['z4_scat']:.3f}"
            )


if __name__ == "__main__":
    main()
