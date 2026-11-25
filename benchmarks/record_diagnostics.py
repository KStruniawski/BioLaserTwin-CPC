"""Record simulator throughput, d_GT(lambda) FSR shift, active PIELM runtime, and sync exact S1 p-values."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from biolaser_twin.observations import calibrate_nominal_constants_from_dev, estimate_noise_cov
from biolaser_twin.simulator import FabryPerotConfig, FabryPerotTruthSimulator, LatentProxyVector, map_latents_to_truth_state
from biolaser_twin.solvers import PhotonicPIELMSolver
from run_cpc_controls import _gen_l1


def main() -> None:
    results_dir = Path(__file__).resolve().parent / "results"

    # 1. Simulator throughput (N_y=16 rows)
    cfg = FabryPerotConfig(
        cavity_length_um=30.0, refractive_index=1.35, n_spectral_px=1024,
        n_spatial_px=16, spec_psf_fwhm_nm=0.10, oversample=16, level=1, seed=0,
    )
    sim = FabryPerotTruthSimulator(cfg)
    st = map_latents_to_truth_state(LatentProxyVector(0.5, 0.5, 1.35, 1.0, 30.0, 10.0))
    energies_100 = np.full(100, 40.0)
    _ = sim.render_pump_sweep(st, 440.0, np.array([40.0]))
    t0 = time.perf_counter()
    _ = sim.render_pump_sweep(st, 440.0, energies_100)
    dt_sim = time.perf_counter() - t0
    throughput_fps = 100.0 / dt_sim

    # 2. Exact d_GT(lambda) FSR shift with full state vs dispersion-only kernel
    wl = np.linspace(475.0, 515.0, 4096)
    fsr_shifts = {}
    for l_um in (15.0, 30.0, 45.0):
        sim_l = FabryPerotTruthSimulator(
            FabryPerotConfig(cavity_length_um=l_um, refractive_index=1.35, n_spectral_px=4096, oversample=16, level=1, seed=0)
        )
        st_l = map_latents_to_truth_state(LatentProxyVector(0.5, 0.5, 1.35, 0.5, l_um, 10.0))
        t_full = sim_l.airy_transmission(wl, state=st_l)
        peaks, _ = find_peaks(t_full, height=0.5 * np.max(t_full), distance=10)
        fsr_meas = float(np.median(np.diff(wl[peaks])))
        lam_c = float(np.mean(wl[peaks]))
        n_g = 1.35 + 0.0194
        fsr_ana = (lam_c ** 2) / (2.0 * n_g * l_um * 1000.0)
        rel_dev = (fsr_meas - fsr_ana) / fsr_ana
        fsr_shifts[f"L_{int(l_um)}um"] = {"fsr_meas_nm": fsr_meas, "fsr_ana_nm": fsr_ana, "rel_dev": rel_dev}

    # 3. Active PhotonicPIELM fit() runtime at N_train=300 (n_gn=3)
    x_tr, y_tr, recs, _ = _gen_l1(300, seed=123)
    cal_consts = calibrate_nominal_constants_from_dev(recs[:16], y_tr[:16])
    ncov = estimate_noise_cov(recs[:16])
    active_times = {}
    for lam in (0.1, 0.25):
        m = PhotonicPIELMSolver(hidden_dim=48, n_ensemble=3, phys_lambda=lam, n_gn_iters=3, constants=cal_consts, noise_cov=ncov, seed=1)
        t1 = time.perf_counter()
        m.fit(x_tr, y_tr, collocation_features=x_tr[:60], collocation_records=recs[:60])
        active_times[f"lambda_{lam}"] = time.perf_counter() - t1

    # 4. Aggregate core benchmark runtimes & sync verification.json s1_per_latent
    verif_path = results_dir / "verification.json"
    verif = json.loads(verif_path.read_text())
    cpc = json.loads((results_dir / "cpc_controls.json").read_text())
    stage_a = json.loads((results_dir / "stage_a_full.json").read_text())
    sggw = json.loads((results_dir / "sggw_445nm_ablation.json").read_text())

    s1_lat = verif.get("bae_and_s1_per_latent", {}).get("s1_per_latent", {})
    for k, ex in cpc["exact_s1_permutation_p"].items():
        if k in s1_lat:
            old_p = s1_lat[k].pop("p", s1_lat[k].get("asymptotic_p"))
            s1_lat[k]["asymptotic_p"] = old_p
            s1_lat[k]["exact_permutation_p"] = ex["exact_two_sided_p"]
    verif_path.write_text(json.dumps(verif, indent=2) + "\n")

    core_runtimes_s = {
        "verification": verif.get("runtime_s"),
        "stage_a_full": stage_a.get("total_runtime_s", stage_a.get("runtime_s")),
        "sggw_445nm_ablation": sggw.get("runtime_s"),
    }
    total_core_min = sum(v for v in core_runtimes_s.values() if v is not None) / 60.0

    diag = {
        "simulator_throughput_fps": throughput_fps,
        "n_frames_timed": 100,
        "d_gt_fsr_shifts": fsr_shifts,
        "active_pielm_fit_s": active_times,
        "core_benchmark_runtimes_s": core_runtimes_s,
        "total_core_runtime_min": total_core_min,
    }
    out_path = results_dir / "diagnostics.json"
    out_path.write_text(json.dumps(diag, indent=2) + "\n")
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
