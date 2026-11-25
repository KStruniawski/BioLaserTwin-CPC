"""Physical feature extraction from 1D spectra and 2D spatial-spectral frames."""

from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class ExtractedFeatures:
    """Observable physical summary vector extracted from a frame or spectrum."""

    center_of_mass_nm: float
    spectral_variance_nm2: float
    peak_wavelength_nm: float
    peak_intensity: float
    estimated_fsr_nm: float | None
    apparent_fwhm_nm: float | None
    spectrometer_limited_q_flag: bool
    fit_failed: bool


@dataclass(frozen=True)
class SweepFeatures:
    """Observable summary vector extracted from a full pump-energy sweep."""

    lasing_detected: bool
    threshold_energy_uj: float | None
    slope_efficiency: float
    mean_fsr_nm: float | None
    apparent_fwhm_nm: float | None
    center_of_mass_nm: float
    off_peak_baseline: float
    spatial_width_px: float
    spectrometer_limited_q_flag: bool
    saturated: bool
    fit_failed: bool


def _estimate_noise_std(spectra_1d: np.ndarray, dark_frames: np.ndarray | None) -> tuple[float, float]:
    """Estimate background offset and noise std from dark frames (matching central-quarter ROI) or MAD."""
    if dark_frames is not None and dark_frames.size > 0:
        if dark_frames.ndim == 3:
            n_y = dark_frames.shape[1]
            y_lo, y_hi = max(0, (3 * n_y) // 8), min(n_y, max(1, (5 * n_y) // 8))
            dark_1d = np.mean(dark_frames[:, y_lo:y_hi, :], axis=1)
        else:
            dark_1d = np.mean(dark_frames, axis=1)
        return float(np.mean(dark_1d)), max(1e-6, float(np.std(dark_1d)))

    # Fallback: MAD of second-difference high-pass residual on lowest-energy spectrum
    spec = spectra_1d[0]
    diff2 = np.diff(spec, n=2)
    mad = float(np.median(np.abs(diff2 - np.median(diff2))))
    sigma = max(1e-6, mad / (0.6745 * np.sqrt(6.0)))
    return float(np.percentile(spec, 5)), sigma


def _interpolated_fwhm_nm(wavelengths_nm: np.ndarray, spectrum: np.ndarray, bg: float) -> float | None:
    """Estimate subpixel-interpolated FWHM around the highest peak; return None if truncated."""
    if len(spectrum) < 5:
        return None
    peak_idx = int(np.argmax(spectrum))
    peak_val = float(spectrum[peak_idx])
    if peak_val <= bg:
        return None
    half_val = bg + 0.5 * (peak_val - bg)

    left_idx = peak_idx
    while left_idx > 0 and spectrum[left_idx] > half_val:
        left_idx -= 1
    right_idx = peak_idx
    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_val:
        right_idx += 1

    if left_idx == 0 and spectrum[0] > half_val:
        return None
    if right_idx == len(spectrum) - 1 and spectrum[-1] > half_val:
        return None
    if left_idx == peak_idx or right_idx == peak_idx:
        return None

    # Linear subpixel interpolation on both sides
    dl = (half_val - spectrum[left_idx]) / max(spectrum[left_idx + 1] - spectrum[left_idx], 1e-12)
    wl_left = wavelengths_nm[left_idx] + dl * (wavelengths_nm[left_idx + 1] - wavelengths_nm[left_idx])

    dr = (spectrum[right_idx - 1] - half_val) / max(spectrum[right_idx - 1] - spectrum[right_idx], 1e-12)
    wl_right = wavelengths_nm[right_idx - 1] + dr * (wavelengths_nm[right_idx] - wavelengths_nm[right_idx - 1])
    return float(max(0.0, wl_right - wl_left))


def _fit_hinge_threshold(energies: np.ndarray, integrated_sig: np.ndarray) -> tuple[float | None, float, bool]:
    """Fit piecewise-linear hinge model y = a + s0*E + s_excess*max(0, E - E_th)."""
    if len(energies) < 3:
        slope = float(np.polyfit(energies, integrated_sig, 1)[0]) if len(energies) >= 2 else 0.0
        return None, slope, False

    best_sse = np.inf
    best_e_th = None
    best_excess_slope = 0.0

    # Linear no-threshold null model
    p_lin = np.polyfit(energies, integrated_sig, 1)
    sse_null = float(np.sum((integrated_sig - np.polyval(p_lin, energies)) ** 2))

    candidates = np.linspace(float(energies[0]), float(energies[-2]), 25)
    for e_cand in candidates:
        hinge = np.maximum(0.0, energies - e_cand)
        design = np.column_stack([np.ones_like(energies), energies, hinge])
        coef, _, _, _ = np.linalg.lstsq(design, integrated_sig, rcond=None)
        if coef[2] <= 0.0:
            continue
        resid = integrated_sig - design @ coef
        sse = float(np.sum(resid**2))
        if sse < best_sse:
            best_sse = sse
            best_e_th = float(e_cand)
            best_excess_slope = float(coef[2])

    # Require genuine kink improvement over linear fluorescence
    lasing_detected = bool(best_e_th is not None and best_sse < 0.70 * sse_null and best_excess_slope > 0.0)
    if not lasing_detected:
        return None, float(p_lin[0]), False
    return best_e_th, best_excess_slope, True


def extract_spectral_features(
    wavelengths_nm: np.ndarray,
    spectrum: np.ndarray,
    spec_psf_fwhm_nm: float = 0.10,
    noise_std: float | None = None,
    nominal_cavity_um: float = 30.0,
) -> ExtractedFeatures:
    """Extract moments, wavenumber-based FSR, interpolated FWHM, and resolution flag."""
    bg = float(np.percentile(spectrum, 5))
    weights = np.clip(spectrum - bg, a_min=0.0, a_max=None)
    total = float(np.sum(weights))
    if total <= 0.0:
        com = float(np.mean(wavelengths_nm))
        var = 0.0
    else:
        com = float(np.sum(wavelengths_nm * weights) / total)
        var = float(np.sum(((wavelengths_nm - com) ** 2) * weights) / total)

    peak_idx = int(np.argmax(spectrum))
    peak_wl = float(wavelengths_nm[peak_idx])
    peak_val = float(spectrum[peak_idx])

    if noise_std is None:
        diff2 = np.diff(spectrum, n=2)
        noise_std = max(1e-6, float(np.median(np.abs(diff2))) / (0.6745 * np.sqrt(6.0)))

    d_wl = float(abs(wavelengths_nm[1] - wavelengths_nm[0]))
    fsr_nom = (485.0**2) / (2.0 * 1.34 * max(nominal_cavity_um, 1.0) * 1000.0)
    min_dist_px = max(2, int(0.55 * fsr_nom / d_wl))
    prominence = max(6.0 * noise_std, 0.08 * max(peak_val - bg, 0.0))

    peaks, _ = find_peaks(spectrum - bg, prominence=prominence, distance=min_dist_px)
    if len(peaks) >= 2:
        # Median of consecutive inverse-wavelength spacings (referenced to 485 nm)
        inv_lam = 1.0 / wavelengths_nm[peaks]
        diffs = np.diff(inv_lam)
        step_inv = float(np.median(np.abs(diffs)))
        fsr_nm = float((485.0**2) * step_inv) if step_inv > 0 else None
        fit_failed = fsr_nm is None
    else:
        fsr_nm = None
        fit_failed = True

    fwhm = _interpolated_fwhm_nm(wavelengths_nm, spectrum, bg)
    spec_limited = bool(fwhm is not None and fwhm <= 1.25 * spec_psf_fwhm_nm)

    return ExtractedFeatures(
        center_of_mass_nm=com,
        spectral_variance_nm2=var,
        peak_wavelength_nm=peak_wl,
        peak_intensity=peak_val,
        estimated_fsr_nm=fsr_nm,
        apparent_fwhm_nm=fwhm,
        spectrometer_limited_q_flag=spec_limited,
        fit_failed=fit_failed,
    )


def extract_sweep_features(
    frames: np.ndarray,
    wavelengths_nm: np.ndarray,
    pump_energies_uj: np.ndarray,
    spec_psf_fwhm_nm: float = 0.10,
    dark_frames: np.ndarray | None = None,
    nominal_cavity_um: float = 30.0,
    adc_max_counts: int = 65535,
) -> SweepFeatures:
    """Extract order-invariant threshold, slope, modal FSR, and flags from a 2D sweep."""
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError("frames must have shape (n_sweeps, n_spatial_px, n_spectral_px)")

    # Sort sweep by ascending pump energy for strict order invariance
    order = np.argsort(pump_energies_uj)
    energies = np.asarray(pump_energies_uj, dtype=float)[order]
    frames_sorted = frames[order]

    # Use central 25% of spatial rows to avoid slit-tilt broadening
    n_y = frames_sorted.shape[1]
    y_lo, y_hi = max(0, (3 * n_y) // 8), min(n_y, max(1, (5 * n_y) // 8))
    spectra_1d = np.mean(frames_sorted[:, y_lo:y_hi, :], axis=1)

    bg_offset, noise_std = _estimate_noise_std(spectra_1d, dark_frames)
    integrated_sig = np.sum(np.clip(spectra_1d - bg_offset, a_min=0.0, a_max=None), axis=1)

    e_th, slope, lasing_detected = _fit_hinge_threshold(energies, integrated_sig)
    top_feats = extract_spectral_features(
        wavelengths_nm,
        spectra_1d[-1],
        spec_psf_fwhm_nm=spec_psf_fwhm_nm,
        noise_std=noise_std,
        nominal_cavity_um=nominal_cavity_um,
    )

    spatial_profile = np.mean(frames_sorted[-1], axis=1)
    y_idx = np.arange(len(spatial_profile), dtype=float)
    sp_w = np.clip(spatial_profile - np.min(spatial_profile), a_min=0.0, a_max=None)
    sp_sum = float(np.sum(sp_w))
    if sp_sum > 0.0:
        y_mean = float(np.sum(y_idx * sp_w) / sp_sum)
        spatial_width = float(np.sqrt(np.sum(((y_idx - y_mean) ** 2) * sp_w) / sp_sum))
    else:
        spatial_width = 0.0

    saturated = bool(np.max(frames_sorted) >= adc_max_counts)

    return SweepFeatures(
        lasing_detected=lasing_detected,
        threshold_energy_uj=e_th,
        slope_efficiency=slope,
        mean_fsr_nm=top_feats.estimated_fsr_nm,
        apparent_fwhm_nm=top_feats.apparent_fwhm_nm,
        center_of_mass_nm=top_feats.center_of_mass_nm,
        off_peak_baseline=float(np.percentile(spectra_1d[-1], 5) - bg_offset),
        spatial_width_px=spatial_width,
        spectrometer_limited_q_flag=top_feats.spectrometer_limited_q_flag,
        saturated=saturated,
        fit_failed=top_feats.fit_failed,
    )
