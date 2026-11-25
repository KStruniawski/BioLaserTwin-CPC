"""Bayesian prior-averaged, unit-normalized, budget-matched A/D-optimal DoE."""

from dataclasses import dataclass
import numpy as np
from biolaser_twin.constants import LATENT_PRIOR_SCALES, NominalOpticalConstants
from .fisher import (
    DesignConfig,
    active_observable_mask,
    compute_fim,
    cramer_rao_sd,
    numerical_jacobian,
)


@dataclass(frozen=True)
class DesignScore:
    """Identifiability score for a candidate experimental design."""

    design: DesignConfig
    a_opt_norm_trace: float  # unit-normalized trace(S^{-1} F^{-1} S^{-1}) (lower is better)
    d_opt_logdet: float  # log det of scaled FIM (higher is better)
    mean_crb_sd: np.ndarray
    identifiable_mask: np.ndarray


def evaluate_design(
    design: DesignConfig,
    reference_latents_grid: np.ndarray,
    noise_cov: np.ndarray,
    constants: NominalOpticalConstants | None = None,
    rel_sd_threshold: float = 0.25,
    measurement_budget_shots: int = 2,
) -> DesignScore:
    """Score a candidate design averaged over reference latents at matched shot budget.

    Dual excitation (2 wavelengths) uses half as many repeats per excitation as single
    excitation under a fixed total shot budget.
    """
    mask = active_observable_mask(design)
    n_exc = max(1, len(design.excitations_nm))
    if measurement_budget_shots <= 0:
        return DesignScore(
            design=design,
            a_opt_norm_trace=np.inf,
            d_opt_logdet=-np.inf,
            mean_crb_sd=np.full(len(LATENT_PRIOR_SCALES), np.inf),
            identifiable_mask=np.zeros(len(LATENT_PRIOR_SCALES), dtype=bool),
        )
    repeats = float(measurement_budget_shots) / float(n_exc)

    sub_cov = noise_cov[np.ix_(mask, mask)] / repeats
    sub_cov_inv = np.linalg.pinv(sub_cov)

    scale = LATENT_PRIOR_SCALES
    scale_mat = np.diag(scale)

    a_traces = []
    d_logdets = []
    crbs = []
    for ref in reference_latents_grid:
        jac_full = numerical_jacobian(ref, design, constants)
        jac = jac_full[mask, :]
        fim = compute_fim(jac, sub_cov_inv)
        # Unit-normalized FIM: S^T F S
        fim_norm = scale_mat @ fim @ scale_mat
        crb_norm = cramer_rao_sd(fim_norm)
        crb = crb_norm * scale
        crbs.append(crb)

        eigvals = np.linalg.eigvalsh(0.5 * (fim_norm + fim_norm.T))
        pos = eigvals[eigvals > 1e-10]
        if len(pos) == len(scale):
            a_traces.append(float(np.sum(1.0 / pos)))
            d_logdets.append(float(np.sum(np.log(pos))))
        else:
            a_traces.append(np.inf)
            d_logdets.append(-np.inf)

    mean_crb = np.mean(np.stack(crbs, axis=0), axis=0)
    ident_mask = (mean_crb / scale) < rel_sd_threshold
    return DesignScore(
        design=design,
        a_opt_norm_trace=float(np.mean(a_traces)),
        d_opt_logdet=float(np.mean(d_logdets)),
        mean_crb_sd=mean_crb,
        identifiable_mask=ident_mask,
    )


def select_optimal_designs(
    candidates: list[DesignConfig],
    reference_latents_grid: np.ndarray,
    noise_cov: np.ndarray,
    criterion: str = "d_opt",
    constants: NominalOpticalConstants | None = None,
    measurement_budget_shots: int = 2,
) -> list[DesignScore]:
    """Rank candidate experimental designs by budget-matched Bayesian A- or D-optimality."""
    scores = [
        evaluate_design(
            d,
            reference_latents_grid,
            noise_cov=noise_cov,
            constants=constants,
            measurement_budget_shots=measurement_budget_shots,
        )
        for d in candidates
    ]
    if criterion == "a_opt":
        return sorted(scores, key=lambda s: s.a_opt_norm_trace)
    return sorted(scores, key=lambda s: s.d_opt_logdet, reverse=True)
