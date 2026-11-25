"""Generate architecture pipeline diagram and BAE/Kymera summary figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.2), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    boxes = [
        (0.3, 1.15, 2.3, 1.3, "#e8f0fe", "#1a73e8", "Truth Simulator\n(simulator/)", "16x oversampled Airy\nDBR, bleaching, PSF,\nCCD shot/read noise"),
        (3.2, 1.15, 2.2, 1.3, "#fef7e0", "#ea8600", "Observable Interface\n(ObservationBundle)", "2D frames, dark frames,\npump energies E_p,\nnoisy L_nom (no truth z)"),
        (6.0, 1.15, 2.3, 1.3, "#e6f4ea", "#1e8e3e", "Feature Extractor\n(extractor/)", "Central-quarter ROI\n6D masked schema y,\nempirical Sigma"),
        (8.9, 1.15, 2.8, 1.3, "#f3e8fd", "#8430ce", "Inverse & UQ Layer\n(solvers/, calibration.py)", "ELM / PIELM + BAE,\nTabularMLP, NLS, CNN/ViT,\nSplitConformalCalibrator"),
    ]

    for x, y, w, h, fc, ec, title, body in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.8
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color="#202124")
        ax.text(x + w / 2, y + 0.38, body, ha="center", va="center", fontsize=7.6, color="#3c4043")

    arrow_kw = dict(arrowstyle="-|>,head_width=0.28,head_length=0.45", lw=1.8, color="#3c4043")
    ax.annotate("", xy=(3.15, 1.8), xytext=(2.68, 1.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.95, 1.8), xytext=(5.48, 1.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.85, 1.8), xytext=(8.38, 1.8), arrowprops=arrow_kw)

    # Runtime import-isolation boundary before downstream solvers/identifiability
    ax.plot([5.70, 5.70], [0.25, 3.05], color="#d93025", linestyle="--", linewidth=2.0)
    ax.text(
        5.70, 3.16,
        "Runtime Import-Isolation Boundary (identifiability/ & solvers/ via sys.modules)",
        ha="center", va="bottom", fontsize=8.4, fontweight="bold", color="#c5221f",
    )
    ax.text(
        8.9, 0.45,
        "Downstream identifiability/ and solvers/ are tested not to import simulator/",
        ha="center", va="center", fontsize=8.0, style="italic", color="#5f6368",
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_architecture.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_architecture.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def make_bae_and_kymera_figure() -> None:
    verif = json.loads((ROOT / "benchmarks" / "results" / "verification.json").read_text())
    stage_a = json.loads((ROOT / "benchmarks" / "results" / "stage_a_full.json").read_text())
    kymera = json.loads((ROOT / "benchmarks" / "results" / "kymera_gratings_doe.json").read_text())["gratings"]

    matched_row = next(r for r in verif["matched_physics_ladder"]["ladder"] if abs(r["alpha_mis"] - 0.0) < 1e-9)
    l0 = stage_a["level0_control"]
    bae = verif["bae_and_s1_per_latent"]["bae_pielm_25pct"]

    plain_means = [matched_row["plain_elm_mean"], l0["plain_elm_mean_nrmse"], bae["plain_elm_nl_z4_mean"]]
    plain_stds = [matched_row["plain_elm_std"], l0["plain_elm_std_nrmse"], bae["plain_elm_nl_z4_std"]]
    pielm_means = [matched_row["photonic_pielm_mean"], l0["photonic_pielm_mean_nrmse"], bae["bae_pielm_nl_z4_mean"]]
    pielm_stds = [matched_row["photonic_pielm_std"], l0["photonic_pielm_std_nrmse"], bae["bae_pielm_nl_z4_std"]]

    pct_matched = 100.0 * (plain_means[0] - pielm_means[0]) / plain_means[0]
    pct_bae = 100.0 * (plain_means[2] - pielm_means[2]) / plain_means[2]
    pct_bae_sd = 100.0 * (plain_stds[2] - pielm_stds[2]) / plain_stds[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.2), dpi=300)

    regimes = [
        "Matched Physics\n" + r"($\alpha_{\mathrm{mis}}=0$, 4-param)",
        "Level 0 Frame\n(Noiseless, 4-param)",
        "Level 1 Post-Hoc\n" + r"(BAE Split, $(nL, z_4)$)",
    ]
    deltas = [
        f"-{pct_matched:.1f}%\n" + r"($\lambda=0.05$)",
        r"$0.0\%$" + "\n" + r"($\lambda=0.0$)",
        f"-{pct_bae:.1f}% / -{pct_bae_sd:.1f}% SD\n" + r"($\lambda\in\{0.1,0.25\}$)",
    ]

    x = np.arange(len(regimes))
    w = 0.34
    ax1.bar(x - w / 2, plain_means, w, yerr=plain_stds, capsize=4, color="#7f8c8d", label="PlainELM")
    ax1.bar(x + w / 2, pielm_means, w, yerr=pielm_stds, capsize=4, color="#2980b9", label="PhotonicPIELM / BAE")
    for idx, d in enumerate(deltas):
        top = max(plain_means[idx] + plain_stds[idx], pielm_means[idx] + pielm_stds[idx])
        ax1.text(x[idx], top + 0.012, d, ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#2c3e50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, fontsize=8.1)
    ax1.set_ylabel("Mean NRMSE (5-seed mean ± SD)")
    ax1.set_ylim(0, 0.365)
    ax1.set_title("(a) PIELM vs. PlainELM across discrepancy regimes")
    ax1.legend(loc="upper right", fontsize=8.3)
    ax1.grid(axis="y", alpha=0.25)

    # Panel (b): PSF-width sweep at Kymera 328i grating-representative resolutions
    labels = [f"{g['grating']}\n({g['psf_fwhm_nm']:.3f} nm)" for g in kymera]
    mlp_m = [g["tabular_mlp_nl_z4_mean"] for g in kymera]
    mlp_s = [g["tabular_mlp_nl_z4_std"] for g in kymera]
    nls_m = [g["classical_nls_nl_z4"] for g in kymera]
    psi_d = [g["d_opt_logdet"] for g in kymera]

    xk = np.arange(len(labels))
    ax2.errorbar(xk, mlp_m, yerr=mlp_s, marker="o", capsize=4, linewidth=2.0, color="#27ae60", label=r"TabularMLP $(nL, z_4)$ NRMSE")
    ax2.plot(xk, nls_m, marker="s", linestyle="--", linewidth=1.8, color="#e67e22", label=r"ClassicalNLS $(nL, z_4)$ NRMSE")
    ax2.set_xticks(xk)
    ax2.set_xticklabels(labels, fontsize=8.3)
    ax2.set_ylabel(r"$(nL, z_4)$ Mean NRMSE")
    ax2.set_ylim(0.04, 0.135)
    ax2.grid(alpha=0.25)

    ax2_r = ax2.twinx()
    ax2_r.plot(xk, psi_d, marker="^", linestyle=":", linewidth=1.8, color="#8e44ad", label=r"D-optimality $\Psi_D$ (nats)")
    ax2_r.set_ylabel(r"D-optimality $\Psi_D$ (nats)", color="#8e44ad")
    ax2_r.tick_params(axis="y", labelcolor="#8e44ad")
    ax2_r.set_ylim(25.5, 34.5)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=7.8, frameon=True)
    ax2.set_title(r"(b) PSF sweep at Kymera 328i resolutions ($L \sim \mathcal{U}[20,40]\,\mu\mathrm{m}$)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_bae_and_kymera.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_bae_and_kymera.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    make_architecture_figure()
    make_bae_and_kymera_figure()
    print("Regenerated fig_architecture.pdf and fig_bae_and_kymera.pdf from JSON artifacts")
