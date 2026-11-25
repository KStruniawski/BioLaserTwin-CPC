"""Per-latent regression, probabilistic UQ (NLL, CRPS, coverage+width), and fixed-denominator NRMSE."""

import math
import numpy as np
from scipy.special import erf


def normalized_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ref_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-latent RMSE normalized by fixed reference/training scale."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    scale = ref_scale if ref_scale is not None else np.std(y_true, axis=0)
    return rmse / np.maximum(scale, 1e-8)


def gaussian_nll(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    """Compute per-latent mean Gaussian negative log-likelihood."""
    var = np.maximum(y_std**2, 1e-12)
    return np.mean(0.5 * (np.log(2.0 * math.pi * var) + ((y_true - y_mean) ** 2) / var), axis=0)


def gaussian_crps(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    """Compute per-latent Continuous Ranked Probability Score for Gaussian predictions."""
    sigma = np.maximum(y_std, 1e-8)
    z = (y_true - y_mean) / sigma
    phi = np.exp(-0.5 * (z**2)) / np.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / np.sqrt(math.pi))
    return np.mean(crps, axis=0)


def interval_coverage_and_width(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    z_score: float = 1.6448536,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-latent 90% prediction-interval coverage and mean interval width."""
    half = z_score * np.maximum(y_std, 1e-8)
    covered = (y_true >= (y_mean - half)) & (y_true <= (y_mean + half))
    return np.mean(covered, axis=0), np.mean(2.0 * half, axis=0)
