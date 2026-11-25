"""Generate an illustrative 2D spatial-spectral 1024x1024 CCD frame preview figure."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biolaser_twin.simulator import FabryPerotConfig, FabryPerotTruthSimulator, TruthSampleState


def main() -> None:
    cfg = FabryPerotConfig(
        cavity_length_um=30.0,
        refractive_index=1.350,
        n_spectral_px=1024,
        n_spatial_px=1024,
        spec_psf_fwhm_nm=0.10,
        oversample=16,
        level=1,
        seed=42,
    )
    sim = FabryPerotTruthSimulator(cfg)
    eta = 1.0 + 49.0 * (0.45 ** 1.4)
    state = TruthSampleState(1.350, 30.0, eta, 0.65, 1.2, 10.0)
    energies = np.array([10.0, 40.0])  # below and above threshold
    bundle = sim.render_pump_sweep(state, excitation_nm=440.0, pump_energies_uj=energies)

    lam = bundle.wavelengths_nm
    frame_below = bundle.frames[0]
    frame_above = bundle.frames[1]

    roi_lo = (3 * cfg.n_spatial_px) // 8
    roi_hi = (5 * cfg.n_spatial_px) // 8
    spec_below = frame_below[roi_lo:roi_hi].mean(axis=0)
    spec_above = frame_above[roi_lo:roi_hi].mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=300, gridspec_kw={"width_ratios": [1.15, 1.0]})

    im = ax1.imshow(
        frame_above,
        aspect="auto",
        origin="lower",
        extent=[lam[0], lam[-1], 0, cfg.n_spatial_px],
        cmap="inferno",
        vmin=100,
        vmax=np.percentile(frame_above, 99.7),
    )
    ax1.axhline(roi_lo, color="cyan", linestyle="--", linewidth=1.2, label=r"Central ROI $[3N_y/8, 5N_y/8)$")
    ax1.axhline(roi_hi, color="cyan", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Spatial detector row (px)")
    ax1.set_title(r"(a) Simulated $1024\times 1024$ CCD frame ($40\,\mu\mathrm{J}$, Level 1)")
    ax1.legend(loc="upper right", fontsize=8, frameon=True, facecolor="black", labelcolor="white")
    cbar = fig.colorbar(im, ax=ax1, pad=0.02)
    cbar.set_label("Counts (ADU)")

    ax2.plot(lam, spec_above, color="#c0392b", linewidth=1.1, label=r"Above threshold ($40\,\mu\mathrm{J}$)")
    ax2.plot(lam, spec_below, color="#2980b9", linewidth=1.1, label=r"Below threshold ($10\,\mu\mathrm{J}$)")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("ROI-averaged intensity (ADU)")
    ax2.set_title("(b) Extracted 1D ROI spectra across threshold")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fig0_frame_preview.pdf"
    out_png = out_dir / "fig0_frame_preview.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved {out_pdf} (frame shape={frame_above.shape}, max={frame_above.max()})")


if __name__ == "__main__":
    main()
