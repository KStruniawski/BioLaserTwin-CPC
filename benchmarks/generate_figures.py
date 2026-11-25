"""Generate publication figures for BioLaserTwin-CPC directly from verified JSON artifacts."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FULL_JSON = RESULTS_DIR / "stage_a_full.json"
SGGW_JSON = RESULTS_DIR / "sggw_445nm_ablation.json"
VERIF_JSON = RESULTS_DIR / "verification.json"
FIG_DIR = PKG_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    full = json.loads(FULL_JSON.read_text())
    sggw = json.loads(SGGW_JSON.read_text())
    verif = json.loads(VERIF_JSON.read_text())

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

    # Figure 1: Data-efficiency curves (10% -> 100%) + Extrapolation & Level 2 OOD
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fracs = ["10pct", "25pct", "50pct", "100pct"]
    x_pct = [10, 25, 50, 100]
    models_to_plot = [
        ("HybridCNNFeatures", "-o"),
        ("TabularMLP", "-s"),
        ("PlainELM", "-^"),
        ("ClassicalNLS", "--d"),
        ("FrameCNN", "-v"),
        ("FrameViT", "-<"),
        ("MeanOnly", ":k"),
    ]
    for name, fmt in models_to_plot:
        ys = [full["s2_curves"][f][name]["level1_interp"]["primary_nl_z4_nrmse"] for f in fracs]
        ax1.plot(x_pct, ys, fmt, label=name, linewidth=1.6, markersize=5)
    ax1.set_xlabel("Training Data Fraction (% of N=300)")
    ax1.set_ylabel(r"Primary Endpoint NRMSE ($nL, z_4$)")
    ax1.set_title("(a) Level 1 Data-Efficiency Curves")
    ax1.set_xticks(x_pct)
    ax1.legend(fontsize=7.5, loc="upper right", ncol=2)

    # Panel (b): 100% Interp vs Extrap vs Level 2 OOD
    bar_models = ["TabularMLP", "HybridCNNFeatures", "PlainELM", "ClassicalNLS", "FrameCNN", "FrameViT"]
    x_idx = np.arange(len(bar_models))
    w = 0.26
    for offset, (split_key, label) in enumerate(
        [("level1_interp", "L1 Interp"), ("level1_extrap", "L1 Extrap"), ("level2_ood", "L2 OOD")]
    ):
        vals = [full["s2_curves"]["100pct"][m][split_key]["primary_nl_z4_nrmse"] for m in bar_models]
        ax2.bar(x_idx + (offset - 1) * w, vals, width=w, label=label)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(["MLP", "Hybrid", "ELM", "NLS", "CNN", "ViT"], rotation=15)
    ax2.set_ylabel(r"Primary Endpoint NRMSE ($nL, z_4$)")
    ax2.set_title("(b) Generalization at 100% Data")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_learning_and_generalization.pdf")
    plt.close(fig)

    # Figure 2: Per-latent NRMSE (25% data) + Validation lambda sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    latents = ["nL_um", "z1_visc", "z2_bound", "z4_scat"]
    lat_labels = [r"$nL$", r"$z_1$ (visc)", r"$z_2$ (bound)", r"$z_4$ (scat)"]
    comp_models = ["HybridCNNFeatures", "TabularMLP", "PlainELM", "ClassicalNLS", "FrameCNN"]
    x_l = np.arange(len(latents))
    bw = 0.16
    for i, m in enumerate(comp_models):
        vals = [full["s2_curves"]["25pct"][m]["level1_interp"]["per_latent"][lat]["nrmse"] for lat in latents]
        ax1.bar(x_l + (i - 2) * bw, vals, width=bw, label=m)
    ax1.set_xticks(x_l)
    ax1.set_xticklabels(lat_labels)
    ax1.set_ylabel("Normalized RMSE (25% Data)")
    ax1.set_title("(a) Per-Latent Error Breakdown (Level 1, 25%)")
    ax1.legend(fontsize=7.5)

    lams = ["0.0", "1e-06", "1e-05", "0.0001", "0.001", "0.01", "0.05", "0.1"]
    lam_labels = ["0", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "5e-2", "0.1"]
    for frac, ls in [("10pct", "--"), ("25pct", "-"), ("100pct", ":")]:
        y_full = [full["val_lambda_logs_all_seeds"][frac]["PhotonicPIELM_mean"][l] for l in lams]
        y_sel = [full["val_lambda_logs_all_seeds"][frac]["SelectivePIELM_mean"][l] for l in lams]
        ax2.plot(lam_labels, y_full, f"o{ls}", label=f"Full ({frac})", markersize=4)
        ax2.plot(lam_labels, y_sel, f"s{ls}", label=f"Selective ({frac})", markersize=4)
    ax2.set_xlabel(r"Physics Penalty Weight ($\lambda_{\mathrm{phys}}$)")
    ax2.set_ylabel("Validation Mean NRMSE (5-seed mean)")
    ax2.set_title("(b) Validation Loss vs. Physics Weight")
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_per_latent_and_lambda.pdf")
    plt.close(fig)

    # Figure 3: 445 nm (470-530 nm) vs Dual 405+445 nm ablation (with 5-seed std error bars)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x_l = np.arange(len(latents))
    bw = 0.19
    series = [
        ("SGGW_445nm_only", "TabularMLP", "445 nm Only (MLP)"),
        ("Dual_405_445nm", "TabularMLP", "405+445 nm (MLP)"),
        ("SGGW_445nm_only", "PlainELM", "445 nm Only (ELM)"),
        ("Dual_405_445nm", "PlainELM", "405+445 nm (ELM)"),
    ]
    for i, (mode, solver, lbl) in enumerate(series):
        vals = [sggw["results"][mode]["solvers"][solver]["per_latent"][lat] for lat in latents]
        errs = [sggw["results"][mode]["solvers"][solver]["per_latent_std"][lat] for lat in latents]
        ax.bar(x_l + (i - 1.5) * bw, vals, yerr=errs, capsize=2.5, width=bw, label=lbl)
    ax.set_xticks(x_l)
    ax.set_xticklabels(lat_labels)
    ax.set_ylabel("Normalized RMSE (470–530 nm Window)")
    ax.set_title("Single-Excitation (445 nm) vs. Dual-Excitation (405+445 nm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_sggw_445nm_vs_dual.pdf")
    plt.close(fig)

    # Figure 4: Numerical Verification & Literature-Grounded Photophysical Regimes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ovs = [r["oversample"] for r in verif["oversampling_convergence"]["records"]]
    l2_errs = [r["rel_l2_vs_64x"] for r in verif["oversampling_convergence"]["records"]]
    ax1.semilogy(ovs, l2_errs, "-o", linewidth=1.8, markersize=5, color="tab:blue")
    ax1.axvline(16, color="tab:red", linestyle="--", linewidth=1.2, label=r"Default $16\times$ (rel. $L_2 = 7.4\times 10^{-3}$)")
    ax1.set_xlabel("Subpixel Oversampling Factor")
    ax1.set_ylabel(r"Relative $L_2$ Error vs. $64\times$ Reference")
    ax1.set_title("(a) Spectral Discretization Convergence")
    ax1.set_xticks(ovs)
    ax1.legend(fontsize=8)

    visc_recs = verif["literature_regimes"]["viscosity_sweep"]
    etas = [r["viscosity_cp"] for r in visc_recs]
    e405 = [r["e_th_405_uj"] for r in visc_recs]
    e440 = [r["e_th_440_uj"] for r in visc_recs]
    ax2.plot(etas, e405, "-o", label="405 nm (free-ThT / viscosity probe)", markersize=4)
    ax2.plot(etas, e440, "-s", label="440 nm (bound-ThT probe)", markersize=4)
    ax2.set_xlabel(r"Microviscosity $\eta$ (cP)")
    ax2.set_ylabel(r"Lasing Threshold $E_{\mathrm{th}}$ ($\mu$J)")
    ax2.set_title("(b) Dual-Excitation Threshold vs. Microviscosity")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_verification_and_regimes.pdf")
    plt.close(fig)

    print(f"Generated 4 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
