"""DoE and inversion benchmark grounded on the three Andor Kymera 328i diffraction gratings."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from biolaser_twin.identifiability import DesignConfig, evaluate_design
from biolaser_twin.metrics import normalized_rmse
from biolaser_twin.observations import calibrate_nominal_constants_from_dev
from biolaser_twin.solvers import ClassicalNLSSolver, TabularMLPSolver
from run_stage_a_full import (
    LATENT_PRIOR_SCALES,
    _estimate_design_noise_cov,
    _generate_dataset,
)

GRATINGS = [
    {"name": "300 l/mm", "range_nm": [0.31, 0.44], "psf_fwhm_nm": 0.375},
    {"name": "600 l/mm", "range_nm": [0.15, 0.21], "psf_fwhm_nm": 0.180},
    {"name": "1200 l/mm", "range_nm": [0.07, 0.10], "psf_fwhm_nm": 0.085},
]


def main() -> None:
    _, y_dev, _, recs_dev, _ = _generate_dataset(16, level=1, seed=1, psf_fwhm_nm=0.10)
    cal_consts = calibrate_nominal_constants_from_dev(recs_dev, y_dev)

    ref_grid = np.array([
        [32.0, 0.3, 0.3, 1.5],
        [40.0, 0.5, 0.5, 3.0],
        [50.0, 0.7, 0.7, 4.5],
    ])
    results = []
    for i, g in enumerate(GRATINGS):
        psf = g["psf_fwhm_nm"]
        design = DesignConfig(nominal_cavity_um=30.0, spec_psf_fwhm_nm=psf, excitations_nm=(405.0, 440.0))
        cov = _estimate_design_noise_cov(design, level=1, n_repeats=6, seed=1200 + i)
        score = evaluate_design(
            design, ref_grid, noise_cov=cov, constants=cal_consts,
            rel_sd_threshold=0.15, measurement_budget_shots=2,
        )

        x_tr, y_tr, _, recs_tr, fail_tr = _generate_dataset(150, level=1, seed=2100 + i, psf_fwhm_nm=psf)
        x_val, y_val, _, _, fail_val = _generate_dataset(50, level=1, seed=2200 + i, psf_fwhm_nm=psf)
        x_te, y_te, _, recs_te, fail_te = _generate_dataset(100, level=1, seed=2300 + i, psf_fwhm_nm=psf)

        mlp_nl_z4, mlp_all = [], []
        for seed in range(5):
            mlp = TabularMLPSolver(hidden_dim=64, n_ensemble=3, epochs=120, seed=seed)
            mlp.fit(x_tr, y_tr, val_features=x_val, val_targets=y_val)
            mu, _ = mlp.predict_with_uncertainty(x_te)
            nrmse = normalized_rmse(y_te, mu, ref_scale=LATENT_PRIOR_SCALES)
            mlp_nl_z4.append(float(0.5 * (nrmse[0] + nrmse[3])))
            mlp_all.append(float(np.mean(nrmse)))

        nls_pred = ClassicalNLSSolver(constants=cal_consts, noise_cov=cov).predict_records(recs_te)
        nrmse_nls = normalized_rmse(y_te, nls_pred.means, ref_scale=LATENT_PRIOR_SCALES)

        entry = {
            "grating": g["name"],
            "range_nm": g["range_nm"],
            "psf_fwhm_nm": psf,
            "a_opt_norm_trace": float(score.a_opt_norm_trace),
            "d_opt_logdet": float(score.d_opt_logdet),
            "cov_diag": [float(v) for v in np.diag(cov)],
            "fit_failure_rate_test": float(fail_te),
            "tabular_mlp_nl_z4_mean": float(np.mean(mlp_nl_z4)),
            "tabular_mlp_nl_z4_std": float(np.std(mlp_nl_z4)),
            "tabular_mlp_all_mean": float(np.mean(mlp_all)),
            "classical_nls_nl_z4": float(0.5 * (nrmse_nls[0] + nrmse_nls[3])),
            "classical_nls_all": float(np.mean(nrmse_nls)),
        }
        results.append(entry)
        print(
            f"[{g['name']} | PSF={psf:.3f} nm] "
            f"Psi_A={entry['a_opt_norm_trace']:.3f}, Psi_D={entry['d_opt_logdet']:.2f}, "
            f"MLP (nL,z4)={entry['tabular_mlp_nl_z4_mean']:.4f}+/-{entry['tabular_mlp_nl_z4_std']:.4f}, "
            f"NLS (nL,z4)={entry['classical_nls_nl_z4']:.4f}, fail={fail_te*100:.1f}%"
        )

    out_path = ROOT / "benchmarks" / "results" / "kymera_gratings_doe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"gratings": results}, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
