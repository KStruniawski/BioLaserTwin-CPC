"""High-fidelity Fabry-Perot spatial-spectral frame truth simulator (oversampled + pixel-integrated)."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .longitudinal import LongitudinalParams, SampleHistory, relax_sample_history, update_sample_history
from .photophysics import (
    TruthSampleState,
    effective_gain_and_emission,
    wavelength_dependent_scattering,
)


@lru_cache(maxsize=1)
def load_sggw_dbr_mirror_spectra() -> dict[str, np.ndarray]:
    """Load digitized SGGW dielectric DBR mirror spectra (353-698 nm, 1 nm grid)."""
    path = Path(__file__).resolve().parent / "sggw_dbr_mirror_spectra.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "wavelength_nm": np.asarray(raw["wavelength_nm"], dtype=float),
        "experiment_T_0deg": np.asarray(raw["experiment_T_0deg"], dtype=float),
        "experiment_R_20deg": np.asarray(raw["experiment_R_20deg"], dtype=float),
        "experiment_R_0deg_inferred": np.asarray(raw["experiment_R_0deg_inferred"], dtype=float),
        "tmm_digitized_T_0deg": np.asarray(raw["tmm_digitized_T_0deg"], dtype=float),
        "tmm_digitized_R_0deg": np.asarray(raw["tmm_digitized_R_0deg"], dtype=float),
    }


def sggw_dbr_pump_transmittance(
    excitation_nm: float,
    dbr_shift_nm: float = 0.0,
    use_exp: bool = True,
) -> float:
    """Return 0-deg DBR transmittance T(lambda_pump - dbr_shift_nm) for longitudinal pumping."""
    spec = load_sggw_dbr_mirror_spectra()
    key = "experiment_T_0deg" if use_exp else "tmm_digitized_T_0deg"
    t_val = float(np.interp(excitation_nm - float(dbr_shift_nm), spec["wavelength_nm"], spec[key]))
    return float(np.clip(t_val, 0.005, 0.99))


@dataclass(frozen=True)
class FabryPerotConfig:
    """Configuration for rendering a 2D spatial-spectral Fabry-Perot frame."""

    cavity_length_um: float = 30.0
    refractive_index: float = 1.345
    mirror_reflectivity: float = 0.988
    lambda_min_nm: float = 465.0
    lambda_max_nm: float = 505.0
    n_spectral_px: int = 1024
    n_spatial_px: int = 64
    spec_psf_fwhm_nm: float = 0.10
    oversample: int = 16
    level: int = 1  # 0: matched sanity, 1: high-fidelity truth, 2: OOD/cross-instrument
    seed: int = 42
    use_sggw_dbr: bool = False
    dbr_curve: str = "experiment_R_0deg_inferred"
    dbr_shift_nm: float = 0.0
    pump_jitter_rel: float | None = None
    pump_coupling_eta: float = 1.0


@dataclass(frozen=True)
class ObservationBundle:
    """Experiment-observable output passed to extractor/solvers (no hidden truth)."""

    frames: np.ndarray  # shape: (n_sweeps, n_spatial_px, n_spectral_px)
    dark_frames: np.ndarray  # shape: (n_dark, n_spatial_px, n_spectral_px)
    wavelengths_nm: np.ndarray
    pump_energies_uj: np.ndarray  # measured/logged energies
    excitation_nm: float
    nominal_cavity_um: float  # true L + spacer tolerance offset (never exact L)
    spec_psf_fwhm_nm: float
    adc_max_counts: int = 65535


class FabryPerotTruthSimulator:
    """Independent truth generator for Fabry-Perot lasing spectra and 2D frames."""

    def __init__(self, config: FabryPerotConfig | None = None) -> None:
        self.config = config or FabryPerotConfig()
        self._seed_seq = np.random.SeedSequence(self.config.seed)

    def wavelength_grid_nm(self) -> np.ndarray:
        """Return the spectrometer pixel-center wavelength axis in nanometers."""
        return np.linspace(
            self.config.lambda_min_nm,
            self.config.lambda_max_nm,
            self.config.n_spectral_px,
        )

    def _water_dispersion_n(self, base_n: float, wavelengths_nm: np.ndarray) -> np.ndarray:
        """Cauchy-like water dispersion n(lambda) around 485 nm (n_g - n ~ 0.02)."""
        if self.config.level == 0:
            return np.full_like(wavelengths_nm, base_n)
        # dn/dlambda ~ -4e-5 nm^-1 -> n_g = n - lambda * dn/dlambda ~ n + 0.0194
        return base_n - 4.0e-5 * (wavelengths_nm - 485.0)

    def passive_fsr_nm(self, center_lambda_nm: float = 485.0) -> float:
        """Compute theoretical passive FSR = lambda^2 / (2 * n_g * L)."""
        l_nm = self.config.cavity_length_um * 1000.0
        n_g = self.config.refractive_index + (0.0194 if self.config.level >= 1 else 0.0)
        return (center_lambda_nm**2) / (2.0 * n_g * l_nm)

    def _effective_optical_length_nm(
        self,
        state: TruthSampleState,
        wavelengths_nm: np.ndarray,
        temp_offset_k: float = 0.0,
    ) -> np.ndarray:
        """Include water dispersion, thermo-optic shift, and physical-sign DBR penetration."""
        l_geom_nm = state.cavity_length_um * 1000.0
        n_lam = self._water_dispersion_n(state.refractive_index, wavelengths_nm)
        if self.config.level == 0:
            return n_lam * l_geom_nm

        dn_dt = -1.2e-4 * ((state.temperature_k + temp_offset_k) - 298.15)
        # Positive group-delay penetration in DBR mirrors (~180 nm)
        gt_penetration_nm = 180.0 - 0.6 * (wavelengths_nm - 485.0)
        return (n_lam + dn_dt) * (l_geom_nm + gt_penetration_nm)

    def _mirror_reflectivity_curve(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        if self.config.use_sggw_dbr:
            spec = load_sggw_dbr_mirror_spectra()
            r_interp = np.interp(
                wavelengths_nm - float(self.config.dbr_shift_nm),
                spec["wavelength_nm"],
                spec[self.config.dbr_curve],
            )
            return np.clip(r_interp, 0.05, 0.999)
        r = self.config.mirror_reflectivity
        if self.config.level >= 1:
            return np.clip(r - 0.004 * ((wavelengths_nm - 485.0) / 20.0) ** 2, 0.80, 0.999)
        return np.full_like(wavelengths_nm, r)

    def airy_transmission(
        self,
        wavelengths_nm: np.ndarray,
        state: TruthSampleState | None = None,
    ) -> np.ndarray:
        """Compute passive Airy cavity transmission on an arbitrary wavelength grid."""
        r_lam = self._mirror_reflectivity_curve(wavelengths_nm)
        finesse_coeff = 4.0 * r_lam / ((1.0 - r_lam) ** 2)
        if state is None:
            l_nm = self.config.cavity_length_um * 1000.0
            n_lam = self._water_dispersion_n(self.config.refractive_index, wavelengths_nm)
            opt_len = n_lam * l_nm
        else:
            opt_len = self._effective_optical_length_nm(state, wavelengths_nm)
        phase = 2.0 * np.pi * opt_len / wavelengths_nm
        return 1.0 / (1.0 + finesse_coeff * (np.sin(phase) ** 2))

    def render_2d_frame(
        self,
        state: TruthSampleState,
        excitation_nm: float,
        pump_energy_uj: float,
        rng: np.random.Generator | None = None,
        bleach_factor: float = 1.0,
        temp_offset_k: float = 0.0,
    ) -> np.ndarray:
        """Render a 2D spatial-spectral frame via oversampled physics + PSF + pixel integration."""
        rng = rng or np.random.default_rng(self._seed_seq.spawn(1)[0])
        n_px = self.config.n_spectral_px
        ov = max(1, self.config.oversample)
        wl_px = self.wavelength_grid_nm()
        d_wl_px = float(wl_px[1] - wl_px[0])

        # Oversampled grid covering exact pixel bin edges
        wl_fine = np.linspace(
            self.config.lambda_min_nm - 0.5 * d_wl_px,
            self.config.lambda_max_nm + 0.5 * d_wl_px,
            n_px * ov,
            endpoint=False,
        ) + 0.5 * (d_wl_px / ov)

        r_lam = self._mirror_reflectivity_curve(wl_fine)
        r_for_gain = r_lam if self.config.use_sggw_dbr else self.config.mirror_reflectivity
        source_spec, net_gain_cm1, _ = effective_gain_and_emission(
            state,
            excitation_nm,
            pump_energy_uj,
            wl_fine,
            mirror_reflectivity=r_for_gain,
            bleach_factor=bleach_factor,
        )

        # Gain-loaded Airy cavity: G_rt = R(lambda) * exp((g_clamp - alpha_scat - alpha_reabs) * L)
        l_cm = max(state.cavity_length_um * 1e-4, 1e-5)
        if self.config.use_sggw_dbr:
            mirror_loss_ref = -np.log(np.clip(r_lam, 0.5, 0.9999) ** 2) / (2.0 * l_cm)
        else:
            r0 = min(max(self.config.mirror_reflectivity, 0.5), 0.9999)
            # net_gain_cm1 includes mirror_loss(R0); subtract mirror_loss(R0) so R(lambda) is not canceled
            mirror_loss_ref = -np.log(r0**2) / (2.0 * l_cm)
        g_rt = np.clip(r_lam * np.exp((net_gain_cm1 + mirror_loss_ref) * l_cm), 0.05, 0.996)
        opt_len = self._effective_optical_length_nm(state, wl_fine, temp_offset_k=temp_offset_k)
        phase = 2.0 * np.pi * opt_len / wl_fine
        cavity_loaded = ((1.0 - r_lam) ** 2) / (((1.0 - g_rt) ** 2) + 4.0 * g_rt * (np.sin(phase) ** 2))

        # Add elastic off-peak Mie/RGD scattering halo proportional to z4 / RGD spectrum
        alpha_halo = (
            np.interp(wl_fine, state.scattering_wavelengths_nm, state.scattering_loss_spectrum_cm1)
            if (state.scattering_wavelengths_nm is not None and state.scattering_loss_spectrum_cm1 is not None)
            else wavelength_dependent_scattering(state.scattering_loss_cm1, wl_fine)
        )
        scat_halo = 0.015 * alpha_halo * (pump_energy_uj / 20.0)
        ideal_fine = source_spec * cavity_loaded + scat_halo

        # Convolve on oversampled grid with spectrometer PSF before pixel integration
        d_wl_fine = float(wl_fine[1] - wl_fine[0])
        psf_fwhm = self.config.spec_psf_fwhm_nm * (2.0 if self.config.level == 2 else 1.0)
        sigma_fine = (psf_fwhm / 2.355) / d_wl_fine
        if sigma_fine > 0.2:
            ideal_fine = gaussian_filter1d(ideal_fine, sigma=sigma_fine)

        # Integrate oversampled bins into exact detector pixels
        spec_1d = np.mean(ideal_fine.reshape(n_px, ov), axis=1)

        # Spatial profile along slit + subpixel-exact tilt in Level 1/2
        y = np.linspace(-1.0, 1.0, self.config.n_spatial_px)
        spatial_env = np.exp(-0.5 * (y / 0.45) ** 2)
        frame = np.outer(spatial_env, spec_1d)

        if self.config.level >= 1:
            tilt_nm = 0.06 if self.config.level == 1 else 0.22
            for row_idx, yi in enumerate(y):
                shift_px = (tilt_nm * yi) / d_wl_px
                if abs(shift_px) > 1e-4:
                    frame[row_idx] = np.interp(
                        np.arange(n_px) - shift_px,
                        np.arange(n_px),
                        frame[row_idx],
                        left=0.0,
                        right=0.0,
                    )

        if self.config.level >= 1:
            # Shot noise + CCD readout noise + dark offset + integer ADC quantization
            electrons = np.clip(frame * 800.0, 0.0, 65000.0)
            noisy = rng.poisson(electrons).astype(float) + rng.normal(100.0, 8.0, size=frame.shape)
            frame = np.rint(np.clip(noisy, 0.0, 65535.0))

        return frame

    def render_pump_sweep(
        self,
        state: TruthSampleState,
        excitation_nm: float,
        pump_energies_uj: np.ndarray,
        n_dark: int = 2,
        history: SampleHistory | None = None,
        current_time_min: float = 0.0,
        longitudinal_params: LongitudinalParams | None = None,
    ) -> ObservationBundle:
        """Render a pump-energy sweep; history=None is bitwise-identical to v0.1."""
        rng = np.random.default_rng(self._seed_seq.spawn(1)[0])
        frames_list = []
        if history is None:
            cum_dose_uj = 0.0
            temp_base_k = 0.0
            beta = 0.0005
            chi = 0.004
        else:
            lp = longitudinal_params or LongitudinalParams()
            relaxed = relax_sample_history(history, current_time_min, state.microviscosity_cp, lp)
            cum_dose_uj = relaxed.dose_uj
            temp_base_k = relaxed.delta_temp_k
            beta = lp.bleach_beta_per_uj
            chi = lp.heating_chi_k_per_uj

        sweep_dose_uj = 0.0
        for e_nom in pump_energies_uj:
            if self.config.level >= 1:
                j_std = self.config.pump_jitter_rel if self.config.pump_jitter_rel is not None else (0.02 if self.config.level == 1 else 0.08)
                jitter = rng.normal(1.0, j_std)
                e_actual = max(0.05, float(e_nom) * self.config.pump_coupling_eta * jitter)
            else:
                e_actual = float(e_nom) * self.config.pump_coupling_eta

            cum_dose_uj += e_actual
            sweep_dose_uj += e_actual
            bleach = float(np.exp(-beta * cum_dose_uj)) if self.config.level >= 1 else 1.0
            temp_dt = (temp_base_k + chi * sweep_dose_uj) if self.config.level >= 1 else 0.0

            frames_list.append(
                self.render_2d_frame(
                    state,
                    excitation_nm,
                    e_actual,
                    rng=rng,
                    bleach_factor=bleach,
                    temp_offset_k=temp_dt,
                )
            )

        # Render dark frames for data-driven noise estimation
        if self.config.level >= 1:
            dark = np.rint(
                np.clip(
                    rng.normal(100.0, 8.0, size=(n_dark, self.config.n_spatial_px, self.config.n_spectral_px)),
                    0.0,
                    65535.0,
                )
            )
        else:
            dark = np.zeros((n_dark, self.config.n_spatial_px, self.config.n_spectral_px), dtype=float)

        # Spacer tolerance offset prevents exact L leakage to solvers
        l_offset = float(rng.normal(0.0, 0.8)) if self.config.level >= 1 else 0.0

        return ObservationBundle(
            frames=np.stack(frames_list, axis=0),
            dark_frames=dark,
            wavelengths_nm=self.wavelength_grid_nm(),
            pump_energies_uj=np.asarray(pump_energies_uj, dtype=float),
            excitation_nm=excitation_nm,
            nominal_cavity_um=float(state.cavity_length_um + l_offset),
            spec_psf_fwhm_nm=self.config.spec_psf_fwhm_nm,
        )

    def render_pump_sweep_with_history(
        self,
        state: TruthSampleState,
        excitation_nm: float,
        pump_energies_uj: np.ndarray,
        n_dark: int = 2,
        history: SampleHistory | None = None,
        current_time_min: float = 0.0,
        longitudinal_params: LongitudinalParams | None = None,
    ) -> tuple[ObservationBundle, SampleHistory]:
        """Render a longitudinal pump sweep and return both ObservationBundle and updated SampleHistory."""
        rng = np.random.default_rng(self._seed_seq.spawn(1)[0])
        frames_list = []
        lp = longitudinal_params or LongitudinalParams()
        base_hist = history or SampleHistory(last_time_min=float(current_time_min))
        relaxed = relax_sample_history(base_hist, current_time_min, state.microviscosity_cp, lp)
        cum_dose_uj = relaxed.dose_uj
        temp_base_k = relaxed.delta_temp_k

        sweep_dose_uj = 0.0
        for e_nom in pump_energies_uj:
            if self.config.level >= 1:
                j_std = self.config.pump_jitter_rel if self.config.pump_jitter_rel is not None else (0.02 if self.config.level == 1 else 0.08)
                jitter = rng.normal(1.0, j_std)
                e_actual = max(0.05, float(e_nom) * self.config.pump_coupling_eta * jitter)
            else:
                e_actual = float(e_nom) * self.config.pump_coupling_eta

            cum_dose_uj += e_actual
            sweep_dose_uj += e_actual
            bleach = float(np.exp(-lp.bleach_beta_per_uj * cum_dose_uj)) if self.config.level >= 1 else 1.0
            temp_dt = (temp_base_k + lp.heating_chi_k_per_uj * sweep_dose_uj) if self.config.level >= 1 else 0.0

            frames_list.append(
                self.render_2d_frame(
                    state,
                    excitation_nm,
                    e_actual,
                    rng=rng,
                    bleach_factor=bleach,
                    temp_offset_k=temp_dt,
                )
            )

        if self.config.level >= 1:
            dark = np.rint(
                np.clip(
                    rng.normal(100.0, 8.0, size=(n_dark, self.config.n_spatial_px, self.config.n_spectral_px)),
                    0.0,
                    65535.0,
                )
            )
        else:
            dark = np.zeros((n_dark, self.config.n_spatial_px, self.config.n_spectral_px), dtype=float)

        l_offset = float(rng.normal(0.0, 0.8)) if self.config.level >= 1 else 0.0
        bundle = ObservationBundle(
            frames=np.stack(frames_list, axis=0),
            dark_frames=dark,
            wavelengths_nm=self.wavelength_grid_nm(),
            pump_energies_uj=np.asarray(pump_energies_uj, dtype=float),
            excitation_nm=excitation_nm,
            nominal_cavity_um=float(state.cavity_length_um + l_offset),
            spec_psf_fwhm_nm=self.config.spec_psf_fwhm_nm,
        )
        updated_hist = update_sample_history(
            base_hist,
            sweep_dose_uj=sweep_dose_uj,
            current_time_min=current_time_min,
            microviscosity_cp=state.microviscosity_cp,
            params=lp,
        )
        return bundle, updated_hist
