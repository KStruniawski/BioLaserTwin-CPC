# BioLaserTwin (CPC Release v0.1.0)

Standalone software release accompanying the *Computer Physics Communications* (CPiP) manuscript:
**"BioLaserTwin: Simulation and Inverse Modeling of Fabry–Pérot Cavity Lasing with Model Discrepancy in Thioflavin T-Based Amyloid Analysis"**.

## Installation & Verification

```bash
pip install -e ".[dev]"
pytest -q
```

The automated test suite (`21` tests across 4 files) verifies runtime `sys.modules` import isolation between `simulator/` and downstream `identifiability/` & `solvers/`, analytical passive-cavity FSR and Airy linewidth limits, feature extraction and masking, and solver/conformal calibration interfaces.

## Minimal Runnable Quickstart

See Supporting Information Listing S1 for the self-contained end-to-end simulation, feature extraction, active `PhotonicPIELMSolver` fitting, and `SplitConformalCalibrator` workflow.

## Reproducing Manuscript Results & Figures

| Manuscript Item | Script | Output Artifact |
|---|---|---|
| Analytical limits, bias ladder, BAE split, Fig. 5 | `python benchmarks/run_verification.py` | `benchmarks/results/verification.json` |
| Main 9-solver benchmark, Level 0, UQ, Figs. 3–4 | `python benchmarks/run_stage_a_full.py` | `benchmarks/results/stage_a_full.json` |
| Single 445 nm vs. dual 405+445 nm ablation, Fig. 6 | `python benchmarks/run_sggw_445nm_ablation.py` | `benchmarks/results/sggw_445nm_ablation.json` |
| Kymera 328i PSF-width sweep, Table S3, Fig. 7(b) | `python benchmarks/run_kymera_gratings_doe.py` | `benchmarks/results/kymera_gratings_doe.json` |
| Paired BAE, $\hat{L}_{\text{nom}}$ controls, S1 exact p-values, Tables S4–S5 | `python benchmarks/run_cpc_controls.py` | `benchmarks/results/cpc_controls.json` |
| Publication figures | `python benchmarks/generate_figures.py` | `figures/` |

## License & Archiving

Released under the MIT License (`LICENSE`). Packaged for archival deposition in the *CPC* Program Library (Mendeley Data).
