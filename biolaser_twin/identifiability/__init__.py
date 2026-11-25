"""Fisher Information Matrix (FIM), structural rank check, CRB, and Bayesian DoE."""

from .doe import DesignScore, evaluate_design, select_optimal_designs
from .fisher import (
    DesignConfig,
    active_observable_mask,
    check_structural_n_l_rank,
    compute_fim,
    cramer_rao_sd,
    numerical_jacobian,
    reduced_forward_observables,
)

__all__ = [
    "DesignConfig",
    "DesignScore",
    "active_observable_mask",
    "check_structural_n_l_rank",
    "compute_fim",
    "cramer_rao_sd",
    "evaluate_design",
    "numerical_jacobian",
    "reduced_forward_observables",
    "select_optimal_designs",
]
