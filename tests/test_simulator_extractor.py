"""Non-circular simulator and extractor tests verifying oversampling, active-cavity sensitivity, and invariants."""

import importlib
import sys
import numpy as np

from biolaser_twin.extractor import extract_sweep_features
from biolaser_twin.simulator import (
    FabryPerotConfig,
    FabryPerotTruthSimulator,
    LatentProxyVector,
    effective_gain_and_emission,
    map_latents_to_truth_state,
    simulate_secondary_nucleation_kinetics,
)


def test_oversampled_rendering_is_smooth_in_cavity_length() -> None:
    """Verify oversampling + pixel integration eliminates subpixel aliasing jitter in L."""
    peaks = []
    for l_um in np.linspace(30.000, 30.010, 6):
        cfg = FabryPerotConfig(cavity_length_um=float(l_um), level=0, n_spatial_px=8)
        sim = FabryPerotTruthSimulator(cfg)
        state = map_latents_to_truth_state(
            LatentProxyVector(0.3, 0.7, 1.345, 0.5, float(l_um), 10.0)
        )
        frame = sim.render_2d_frame(state, excitation_nm=440.0, pump_energy_uj=30.0)
        peaks.append(float(np.sum(frame)))

    rel_span = (max(peaks) - min(peaks)) / np.mean(peaks)
    assert rel_span < 0.05  # smooth integrated output (no 3.2x aliasing jitter)


def test_fsr_accuracy_across_cavity_lengths() -> None:
    """Verify extracted wavenumber-based FSR matches theory (<1% Level 0, <3.5% Level 1 with GT shift)."""
    for level, tol in ((0, 0.01), (1, 0.035)):
        for l_um in (20.0, 30.0, 45.0):
            cfg = FabryPerotConfig(cavity_length_um=l_um, level=level, n_spatial_px=16, seed=123)
            sim = FabryPerotTruthSimulator(cfg)
            state = map_latents_to_truth_state(
                LatentProxyVector(0.3, 0.7, 1.345, 0.5, l_um, 10.0)
            )
            energies = np.linspace(2.0, 45.0, 10)
            bundle = sim.render_pump_sweep(state, excitation_nm=440.0, pump_energies_uj=energies)
            feats = extract_sweep_features(
                bundle.frames,
                bundle.wavelengths_nm,
                bundle.pump_energies_uj,
                spec_psf_fwhm_nm=bundle.spec_psf_fwhm_nm,
                dark_frames=bundle.dark_frames,
                nominal_cavity_um=bundle.nominal_cavity_um,
            )
            assert feats.lasing_detected is True
            assert feats.mean_fsr_nm is not None
            expected_fsr = sim.passive_fsr_nm(485.0)
            assert abs(feats.mean_fsr_nm - expected_fsr) / expected_fsr < tol


def test_sweep_order_invariance_and_shot_independence() -> None:
    """Verify reversed pump sweeps yield identical features and repeated sweeps have independent noise."""
    cfg = FabryPerotConfig(cavity_length_um=30.0, level=1, n_spatial_px=16, seed=7)
    sim = FabryPerotTruthSimulator(cfg)
    state = map_latents_to_truth_state(LatentProxyVector(0.3, 0.7, 1.345, 0.5, 30.0, 10.0))
    energies = np.linspace(2.0, 45.0, 8)
    b1 = sim.render_pump_sweep(state, 440.0, energies)
    b2 = sim.render_pump_sweep(state, 440.0, energies)
    assert not np.array_equal(b1.frames, b2.frames)  # independent RNG streams

    f_asc = extract_sweep_features(
        b1.frames, b1.wavelengths_nm, b1.pump_energies_uj, dark_frames=b1.dark_frames
    )
    f_desc = extract_sweep_features(
        b1.frames[::-1], b1.wavelengths_nm, b1.pump_energies_uj[::-1], dark_frames=b1.dark_frames
    )
    assert f_asc.threshold_energy_uj == f_desc.threshold_energy_uj
    assert f_asc.mean_fsr_nm == f_desc.mean_fsr_nm


def test_threshold_sensitivity_to_reflectivity_and_scattering() -> None:
    """Verify E_th depends on mirror_reflectivity and scattering_loss_cm1."""
    wl = np.linspace(465.0, 505.0, 128)
    s_low = map_latents_to_truth_state(LatentProxyVector(0.3, 0.7, 1.345, 0.2, 30.0, 10.0))
    s_high = map_latents_to_truth_state(LatentProxyVector(0.3, 0.7, 1.345, 8.0, 30.0, 10.0))
    _, _, eth_r95 = effective_gain_and_emission(s_low, 440.0, 20.0, wl, mirror_reflectivity=0.95)
    _, _, eth_r99 = effective_gain_and_emission(s_low, 440.0, 20.0, wl, mirror_reflectivity=0.99)
    _, _, eth_scat = effective_gain_and_emission(s_high, 440.0, 20.0, wl, mirror_reflectivity=0.99)
    assert eth_r95 > eth_r99
    assert eth_scat > eth_r99


def test_kinetics_respects_nonuniform_time_grid() -> None:
    """Verify non-uniform time grid integrates actual dt rather than median step."""
    t_short = simulate_secondary_nucleation_kinetics(np.array([0.0, 1.0, 2.0, 3.0]))
    t_long = simulate_secondary_nucleation_kinetics(np.array([0.0, 1.0, 2.0, 100.0]))
    assert t_long.fibril_frac[-1] > t_short.fibril_frac[-1]


def test_solvers_and_identifiability_never_import_simulator() -> None:
    """Enforce anti-inverse-crime isolation for both solvers and identifiability."""
    for mod in list(sys.modules):
        if mod.startswith("biolaser_twin"):
            del sys.modules[mod]
    importlib.import_module("biolaser_twin.solvers")
    importlib.import_module("biolaser_twin.identifiability")
    assert not any(m.startswith("biolaser_twin.simulator") for m in sys.modules)
