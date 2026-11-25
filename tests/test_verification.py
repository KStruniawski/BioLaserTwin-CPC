"""Verification tests for analytical cavity limits and 445 nm DesignConfig support."""

import numpy as np
from scipy.signal import find_peaks

from biolaser_twin.identifiability import DesignConfig, active_observable_mask, reduced_forward_observables
from biolaser_twin.simulator import FabryPerotConfig, FabryPerotTruthSimulator


def test_passive_fsr_matches_analytical():
    """Passive Airy peak spacing matches lambda^2 / (2 * n_g * L) within 0.1%."""
    n0 = 1.345
    l_um = 25.0
    for level in (0, 1):
        n_g = n0 + (0.0194 if level >= 1 else 0.0)
        sim = FabryPerotTruthSimulator(
            FabryPerotConfig(cavity_length_um=l_um, refractive_index=n0, mirror_reflectivity=0.985, level=level)
        )
        wl = np.linspace(480.0, 490.0, 32768)
        trans = sim.airy_transmission(wl, state=None)
        peaks, _ = find_peaks(trans, height=0.5, distance=20)
        fsr_sim = float((485.0**2) * np.median(np.abs(np.diff(1.0 / wl[peaks]))))
        fsr_ana = float((485.0**2) / (2.0 * n_g * l_um * 1000.0))
        assert abs(fsr_sim - fsr_ana) / fsr_ana < 1e-3


def test_passive_airy_fwhm_matches_finesse():
    """Passive Airy linewidth matches FSR / (pi*sqrt(R)/(1-R)) within 1%."""
    n0, l_um, r_val = 1.345, 20.0, 0.988
    sim = FabryPerotTruthSimulator(
        FabryPerotConfig(cavity_length_um=l_um, refractive_index=n0, mirror_reflectivity=r_val, level=0)
    )
    wl = np.linspace(482.0, 488.0, 150000)
    trans = sim.airy_transmission(wl, state=None)
    peaks, _ = find_peaks(trans, height=0.8)
    pk = peaks[len(peaks) // 2]
    half = 0.5 * trans[pk]
    left = pk
    while left > 0 and trans[left] > half:
        left -= 1
    right = pk
    while right < len(trans) - 1 and trans[right] > half:
        right += 1
    fwhm_sim = float(wl[right] - wl[left])
    fsr_ana = (485.0**2) / (2.0 * n0 * l_um * 1000.0)
    fwhm_ana = fsr_ana / (np.pi * np.sqrt(r_val) / (1.0 - r_val))
    assert abs(fwhm_sim - fwhm_ana) / fwhm_ana < 0.01


def test_445nm_design_config_mask_and_forward():
    """445 nm excitation is properly recognized in active_observable_mask and reduced_forward_observables."""
    d_single = DesignConfig(excitations_nm=(445.0,))
    mask_single = active_observable_mask(d_single)
    assert mask_single.tolist() == [True, False, True, False, True, True]

    d_dual = DesignConfig(excitations_nm=(405.0, 445.0))
    mask_dual = active_observable_mask(d_dual)
    assert mask_dual.tolist() == [True, True, True, True, True, True]

    obs_440 = reduced_forward_observables(np.array([40.0, 0.5, 0.5, 2.0]), DesignConfig(excitations_nm=(440.0,)))
    obs_445 = reduced_forward_observables(np.array([40.0, 0.5, 0.5, 2.0]), d_single)
    assert obs_445[2] != obs_440[2]  # 445 nm has distinct excitation efficiency from 440 nm


def test_two_peak_fsr_uses_485nm_reference():
    """Two-peak spectra must use the same 485 nm inverse-wavelength FSR convention as >=3 peaks."""
    from biolaser_twin.extractor.frame_features import extract_sweep_features

    wl = np.linspace(495.0, 510.0, 3001)
    spec = 0.05 + np.exp(-0.5 * ((wl - 500.0) / 0.1) ** 2) + np.exp(-0.5 * ((wl - 503.0) / 0.1) ** 2)
    frames = np.tile(spec[None, None, :], (2, 4, 1))
    feats = extract_sweep_features(frames, wl, np.array([10.0, 50.0]), spec_psf_fwhm_nm=0.2)
    expected = (485.0**2) * abs(1.0 / 503.0 - 1.0 / 500.0)
    assert feats.mean_fsr_nm is not None
    assert abs(feats.mean_fsr_nm - expected) < 0.02


def test_estimate_kinetic_noise_std_supports_four_channels():
    """estimate_kinetic_noise_std must handle 4-channel series without broadcasting errors."""
    from biolaser_twin.observations import estimate_kinetic_noise_std

    rep1 = np.ones((5, 4))
    rep2 = np.ones((5, 4)) * 1.1
    stds = estimate_kinetic_noise_std([rep1, rep2])
    assert stds.shape == (5, 4)
