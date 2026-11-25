"""Reduced-order nominal forward model, structural rank check, and eigen-based FIM/CRB."""

from dataclasses import dataclass
import numpy as np
from biolaser_twin.constants import NominalOpticalConstants


@dataclass(frozen=True)
class DesignConfig:
    """Candidate experimental design configuration (controllable hardware settings)."""

    nominal_cavity_um: float = 30.0
    spec_psf_fwhm_nm: float = 0.10
    excitations_nm: tuple[float, ...] = (405.0, 440.0)


def reduced_forward_observables(
    latents: np.ndarray,
    design: DesignConfig,
    constants: NominalOpticalConstants | None = None,
) -> np.ndarray:
    """Reduced-order nominal map from [nL_um, z1_visc, z2_bound, z4_scat] to 6D schema.

    Schema: [fsr_nm, e_th_405_uj, e_th_440_uj, com_405_nm, com_440_nm, scat_norm].
    Uses only NominalOpticalConstants (isolated from truth simulator).
    """
    c = constants or NominalOpticalConstants()
    nl_um, z1_visc, z2_bound, z4_scat = [float(v) for v in latents]

    # FSR = lambda^2 / (2 * nL)
    fsr_nm = (485.0**2) / (2.0 * max(nl_um, 1.0) * 1000.0)

    # Nominal cavity loss depends on L_est = nL / n_water_nominal (so dE_th/dnL != 0)
    l_est_cm = max((nl_um / c.n_water_nominal) * 1e-4, 1e-5)
    mirror_loss = -np.log(c.mirror_reflectivity_nominal**2) / (2.0 * l_est_cm)
    total_loss = mirror_loss + max(z4_scat, 0.01)

    def _e_th_and_com(exc_nm: float) -> tuple[float, float]:
        w_free = float(np.exp(-0.5 * ((exc_nm - 408.0) / 18.0) ** 2))
        w_bound = float(np.exp(-0.5 * ((exc_nm - 442.0) / 16.0) ** 2))
        visc_proxy = 0.12 * (1.0 + 4.0 * max(0.0, z1_visc))
        gain_proxy = (
            c.gain_scale_kappa
            * c.nominal_tht_conc_mm
            * ((1.0 - z2_bound) * w_free * visc_proxy + z2_bound * w_bound)
        )
        e_th = total_loss / max(0.04 * gain_proxy, 1e-4)
        # Excitation-dependent COM shift
        bound_weight = (z2_bound * w_bound) / max((1.0 - z2_bound) * w_free * visc_proxy + z2_bound * w_bound, 1e-6)
        com = 478.0 + 8.0 * bound_weight
        return e_th, com

    exc_blue = next((e for e in design.excitations_nm if e < 430.0), 405.0)
    exc_long = next((e for e in design.excitations_nm if e >= 430.0), 440.0)
    e_405, com_405 = _e_th_and_com(exc_blue)
    e_440, com_440 = _e_th_and_com(exc_long)
    scat_norm = c.scat_scale_nominal * max(z4_scat, 0.0)
    return np.array([fsr_nm, e_405, e_440, com_405, com_440, scat_norm], dtype=float)


def active_observable_mask(design: DesignConfig) -> np.ndarray:
    """Return boolean mask of schema indices active under the given DesignConfig."""
    has_blue = any(e < 430.0 for e in design.excitations_nm)
    has_long = any(e >= 430.0 for e in design.excitations_nm)
    return np.array([True, has_blue, has_long, has_blue, has_long, True], dtype=bool)


def numerical_jacobian(
    latents: np.ndarray,
    design: DesignConfig,
    constants: NominalOpticalConstants | None = None,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute central finite-difference Jacobian J = d(observables)/d(latents)."""
    x0 = np.asarray(latents, dtype=float)
    y0 = reduced_forward_observables(x0, design, constants)
    jac = np.zeros((len(y0), len(x0)), dtype=float)
    for j in range(len(x0)):
        step = eps * max(1.0, abs(x0[j]))
        dx = np.zeros_like(x0)
        dx[j] = step
        yp = reduced_forward_observables(x0 + dx, design, constants)
        ym = reduced_forward_observables(x0 - dx, design, constants)
        jac[:, j] = (yp - ym) / (2.0 * step)
    return jac


def check_structural_n_l_rank(
    n_val: float = 1.345,
    l_um: float = 30.0,
    z1: float = 0.3,
    z2: float = 0.7,
    z4: float = 0.5,
    design: DesignConfig | None = None,
    constants: NominalOpticalConstants | None = None,
) -> tuple[int, int]:
    """Verify S1 structural unidentifiability: [n, L, z1, z2, z4] Jacobian is rank-deficient (<5)."""
    d = design or DesignConfig()
    # Parameterize via (n, L) -> nL = n * L and evaluate observables dependent on nL
    x5 = np.array([n_val, l_um, z1, z2, z4], dtype=float)
    eps = 1e-4

    def _obs_nl(p: np.ndarray) -> np.ndarray:
        nl_um = p[0] * p[1]
        fsr = (485.0**2) / (2.0 * nl_um * 1000.0)
        base = reduced_forward_observables(np.array([nl_um, p[2], p[3], p[4]]), d, constants)
        base[0] = fsr
        return base

    jac5 = np.zeros((6, 5), dtype=float)
    for j in range(5):
        step = eps * max(1.0, abs(x5[j]))
        dx = np.zeros(5)
        dx[j] = step
        jac5[:, j] = (_obs_nl(x5 + dx) - _obs_nl(x5 - dx)) / (2.0 * step)
    rank = int(np.linalg.matrix_rank(jac5, tol=1e-8))
    return rank, 5


def compute_fim(jacobian: np.ndarray, noise_cov_inv: np.ndarray) -> np.ndarray:
    """Compute Fisher Information Matrix F = J^T Sigma^{-1} J."""
    return jacobian.T @ noise_cov_inv @ jacobian


def cramer_rao_sd(fim: np.ndarray, eig_tol: float = 1e-10) -> np.ndarray:
    """Eigen-based CRB SD: returns np.inf on null/unidentifiable eigendirections."""
    eigvals, eigvecs = np.linalg.eigh(0.5 * (fim + fim.T))
    max_eig = float(np.max(eigvals)) if len(eigvals) > 0 else 0.0
    cutoff = max(eig_tol, 1e-12 * max_eig)

    variances = np.zeros(fim.shape[0], dtype=float)
    for k, lam in enumerate(eigvals):
        v = eigvecs[:, k]
        if lam <= cutoff:
            # Any coordinate with projection onto null eigenvector is unidentifiable
            variances = np.where(np.abs(v) > 1e-8, np.inf, variances)
        else:
            variances += (v**2) / lam
    return np.sqrt(variances)
