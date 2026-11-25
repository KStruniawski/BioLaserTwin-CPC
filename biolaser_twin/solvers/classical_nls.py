"""Classical whitened MAP/NLS baseline solver with multistart, masking, and fit-failure flag."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from biolaser_twin.constants import LATENT_PRIOR_SCALES, NominalOpticalConstants
from biolaser_twin.identifiability import DesignConfig, reduced_forward_observables
from biolaser_twin.observations import ObservationRecord


@dataclass(frozen=True)
class NLSResult:
    """Inversion output with predictive mean, Laplace std, and failure flags."""

    means: np.ndarray
    stds: np.ndarray
    fit_failed: np.ndarray


class ClassicalNLSSolver:
    """Classical per-experiment whitened MAP/NLS inversion using reduced-order nominal physics."""

    def __init__(
        self,
        constants: NominalOpticalConstants | None = None,
        noise_cov: np.ndarray | None = None,
        prior_mean: np.ndarray | None = None,
        prior_std: np.ndarray | None = None,
        n_starts: int = 3,
    ) -> None:
        self.constants = constants or NominalOpticalConstants()
        self.noise_cov = (
            noise_cov
            if noise_cov is not None
            else np.diag(np.array([0.08, 2.0, 2.0, 0.3, 0.3, 0.02]) ** 2)
        )
        self.prior_mean = (
            prior_mean if prior_mean is not None else np.array([40.0, 0.35, 0.5, 2.5], dtype=float)
        )
        self.prior_std = (
            prior_std if prior_std is not None else LATENT_PRIOR_SCALES * 3.0
        )
        self.n_starts = n_starts
        self.bounds_lo = np.array([10.0, 0.0, 0.0, 0.01], dtype=float)
        self.bounds_hi = np.array([80.0, 1.0, 1.0, 25.0], dtype=float)

    def fit(self, *args, **kwargs) -> "ClassicalNLSSolver":
        """No-op training (classical physics inversion requires no training set)."""
        return self

    def predict_records(self, records: list[ObservationRecord]) -> NLSResult:
        """Invert named ObservationRecords via Cholesky-whitened multistart MAP."""
        means = []
        stds = []
        failed = []
        chol_cache: dict[tuple[bool, ...], np.ndarray] = {}

        starts = [
            self.prior_mean,
            np.array([28.0, 0.2, 0.2, 1.0]),
            np.array([55.0, 0.7, 0.8, 5.0]),
        ][: self.n_starts]

        for rec in records:
            mask = rec.mask
            if not np.any(mask):
                means.append(self.prior_mean.copy())
                stds.append(self.prior_std.copy())
                failed.append(True)
                continue

            key = tuple(bool(b) for b in mask)
            if key not in chol_cache:
                sub_cov = self.noise_cov[np.ix_(mask, mask)]
                chol_cache[key] = np.linalg.cholesky(sub_cov + 1e-10 * np.eye(sub_cov.shape[0]))
            l_sub = chol_cache[key]

            def _resid(theta: np.ndarray) -> np.ndarray:
                pred = reduced_forward_observables(theta, rec.design, self.constants)
                data_term = solve_triangular(l_sub, pred[mask] - rec.values[mask], lower=True)
                prior_term = (theta - self.prior_mean) / self.prior_std
                return np.concatenate([data_term, prior_term])

            best_res = None
            for x0 in starts:
                x0_clip = np.clip(x0, self.bounds_lo + 1e-3, self.bounds_hi - 1e-3)
                res = least_squares(_resid, x0=x0_clip, bounds=(self.bounds_lo, self.bounds_hi), max_nfev=150)
                if best_res is None or res.cost < best_res.cost:
                    best_res = res

            assert best_res is not None
            means.append(best_res.x)
            jtj = best_res.jac.T @ best_res.jac + 1e-8 * np.eye(len(best_res.x))
            cov = np.linalg.pinv(jtj)
            stds.append(np.sqrt(np.clip(np.diag(cov), 1e-8, None)))
            failed.append(not bool(best_res.success))

        return NLSResult(
            means=np.asarray(means, dtype=float),
            stds=np.asarray(stds, dtype=float),
            fit_failed=np.asarray(failed, dtype=bool),
        )

    def predict_with_uncertainty(
        self,
        obs_vectors: np.ndarray,
        design: DesignConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper for array inputs using an explicit DesignConfig."""
        d = design or DesignConfig()
        records = [
            ObservationRecord(
                values=v,
                mask=np.isfinite(v),
                design=d,
                lasing_405=True,
                lasing_440=True,
            )
            for v in obs_vectors
        ]
        res = self.predict_records(records)
        return res.means, res.stds
