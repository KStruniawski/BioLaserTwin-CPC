"""Shared split-conformal calibration for fair UQ across all inverse solvers."""

import numpy as np
from scipy.stats import norm


class SplitConformalCalibrator:
    """Per-latent multiplicative split-conformal calibrator fitted on Level 1 val."""

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.scales: np.ndarray | None = None

    def fit(self, y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> "SplitConformalCalibrator":
        """Compute finite-sample conformal order-statistic scaling factors per latent."""
        scores = np.abs(y_true - y_mean) / np.maximum(y_std, 1e-8)
        n = scores.shape[0]
        q_level = min(1.0, ((1.0 - self.alpha) * (n + 1)) / max(n, 1))
        self.scales = np.quantile(scores, q_level, axis=0, method="higher")
        return self

    def calibrate_std(self, y_std: np.ndarray) -> np.ndarray:
        """Scale predictive stds so z_{1-alpha/2} * std_cal matches conformal half-width."""
        if self.scales is None:
            raise RuntimeError("SplitConformalCalibrator must be fitted before calibrate_std().")
        z_alpha = float(norm.ppf(1.0 - 0.5 * self.alpha))
        return np.maximum(y_std, 1e-8) * (self.scales / z_alpha)
