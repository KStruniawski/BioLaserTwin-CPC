"""PlainELM and coupled Gauss-Newton PhotonicPIELM solvers with full Cholesky whitening."""

import numpy as np
from scipy.linalg import solve_triangular
from biolaser_twin.constants import NominalOpticalConstants
from biolaser_twin.identifiability import numerical_jacobian, reduced_forward_observables
from biolaser_twin.observations import ObservationRecord


class PlainELMSolver:
    """Data-only Extreme Learning Machine baseline with standardized targets and aleatoric UQ."""

    def __init__(
        self,
        hidden_dim: int = 128,
        reg_lambda: float = 1e-2,
        n_ensemble: int = 5,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.reg_lambda = reg_lambda
        self.n_ensemble = n_ensemble
        self.seed = seed
        self._members: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None
        self._aleatoric_var: np.ndarray | None = None

    def _standardize_x(self, features: np.ndarray) -> np.ndarray:
        assert self._x_mean is not None and self._x_std is not None
        return (features - self._x_mean) / self._x_std

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "PlainELMSolver":
        """Fit an ensemble of regularized ELMs on standardized features and targets."""
        self._x_mean = np.mean(features, axis=0)
        self._x_std = np.maximum(np.std(features, axis=0), 1e-6)
        self._y_mean = np.mean(targets, axis=0)
        self._y_std = np.maximum(np.std(targets, axis=0), 1e-6)

        x_norm = self._standardize_x(features)
        y_norm = (targets - self._y_mean) / self._y_std

        self._members = []
        for m_idx in range(self.n_ensemble):
            rng = np.random.default_rng(self.seed + m_idx)
            w_in = rng.normal(0.0, 0.8, size=(x_norm.shape[1], self.hidden_dim))
            b_in = rng.uniform(-1.0, 1.0, size=(self.hidden_dim,))
            h = np.tanh(x_norm @ w_in + b_in)
            lhs = h.T @ h + self.reg_lambda * np.eye(self.hidden_dim)
            w_out = np.linalg.solve(lhs, h.T @ y_norm)
            self._members.append((w_in, b_in, w_out))

        pred_mean, _ = self._raw_ensemble_stats(features)
        resid_sq = np.mean((targets - pred_mean) ** 2, axis=0)
        self._aleatoric_var = np.maximum(resid_sq, 1e-6)
        return self

    def _raw_ensemble_stats(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._y_mean is not None and self._y_std is not None
        x_norm = self._standardize_x(features)
        preds = np.stack(
            [
                (np.tanh(x_norm @ w_in + b_in) @ w_out) * self._y_std + self._y_mean
                for w_in, b_in, w_out in self._members
            ],
            axis=0,
        )
        return np.mean(preds, axis=0), np.var(preds, axis=0)

    def predict_with_uncertainty(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ensemble mean and total (epistemic + aleatoric) predictive std."""
        if not self._members or self._aleatoric_var is None:
            raise RuntimeError("PlainELMSolver must be fitted before prediction.")
        mean, ep_var = self._raw_ensemble_stats(features)
        total_std = np.sqrt(np.maximum(ep_var + self._aleatoric_var[None, :], 1e-8))
        return mean, total_std

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return ensemble mean prediction."""
        mean, _ = self.predict_with_uncertainty(features)
        return mean


class PhotonicPIELMSolver(PlainELMSolver):
    """Coupled Gauss-Newton PhotonicPIELM with Cholesky whitening and step acceptance."""

    def __init__(
        self,
        hidden_dim: int = 128,
        reg_lambda: float = 1e-2,
        phys_lambda: float = 0.25,
        n_ensemble: int = 5,
        n_gn_iters: int = 5,
        damping: float = 1e-2,
        constants: NominalOpticalConstants | None = None,
        noise_cov: np.ndarray | None = None,
        active_obs_mask: np.ndarray | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, reg_lambda=reg_lambda, n_ensemble=n_ensemble, seed=seed)
        self.phys_lambda = phys_lambda
        self.n_gn_iters = n_gn_iters
        self.damping = damping
        self.constants = constants or NominalOpticalConstants()
        self.noise_cov = (
            noise_cov
            if noise_cov is not None
            else np.diag(np.array([0.08, 2.0, 2.0, 0.3, 0.3, 0.02]) ** 2)
        )
        self.active_obs_mask = active_obs_mask if active_obs_mask is not None else np.ones(6, dtype=bool)

    def _objective(
        self,
        w_out: np.ndarray,
        h_sup: np.ndarray,
        y_sup_norm: np.ndarray,
        h_col: np.ndarray,
        collocation_records: list[ObservationRecord],
        chol_cache: dict[tuple[bool, ...], np.ndarray],
    ) -> float:
        assert self._y_mean is not None and self._y_std is not None
        n_sup = max(1, h_sup.shape[0])
        n_col = max(1, h_col.shape[0])
        sup_err = h_sup @ w_out - y_sup_norm
        loss = 0.5 * float(np.sum(sup_err**2)) / float(n_sup)
        loss += 0.5 * (self.reg_lambda / float(n_sup)) * float(np.sum(w_out**2))

        z_col_pred = (h_col @ w_out) * self._y_std + self._y_mean
        weight_p = 0.5 * self.phys_lambda / float(n_col)
        for i, rec in enumerate(collocation_records):
            mask = rec.mask & self.active_obs_mask
            if not np.any(mask):
                continue
            key = tuple(bool(b) for b in mask)
            l_sub = chol_cache[key]
            g_pred = reduced_forward_observables(z_col_pred[i], rec.design, self.constants)
            r_white = solve_triangular(l_sub, g_pred[mask] - rec.values[mask], lower=True)
            loss += weight_p * float(np.sum(r_white**2))
        return loss

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        collocation_features: np.ndarray | None = None,
        collocation_records: list[ObservationRecord] | None = None,
    ) -> "PhotonicPIELMSolver":
        """Fit ensemble via coupled Gauss-Newton updates with exact ridge gradient."""
        super().fit(features, targets)
        if (
            collocation_features is None
            or collocation_records is None
            or self.phys_lambda <= 0.0
            or self.n_gn_iters <= 0
        ):
            return self

        assert self._y_mean is not None and self._y_std is not None
        x_sup_norm = self._standardize_x(features)
        y_sup_norm = (targets - self._y_mean) / self._y_std
        x_col_norm = self._standardize_x(collocation_features)

        n_sup = max(1, features.shape[0])
        n_col = max(1, collocation_features.shape[0])
        n_lat = targets.shape[1]

        # Precompute Cholesky factors for each unique missingness mask
        chol_cache: dict[tuple[bool, ...], np.ndarray] = {}
        for rec in collocation_records:
            mask = rec.mask & self.active_obs_mask
            if np.any(mask):
                key = tuple(bool(b) for b in mask)
                if key not in chol_cache:
                    sub_cov = self.noise_cov[np.ix_(mask, mask)]
                    chol_cache[key] = np.linalg.cholesky(sub_cov + 1e-10 * np.eye(sub_cov.shape[0]))

        updated_members = []
        for w_in, b_in, w_out_init in self._members:
            h_sup = np.tanh(x_sup_norm @ w_in + b_in)
            h_col = np.tanh(x_col_norm @ w_in + b_in)
            w_out = w_out_init.copy()

            ht_h_sup = (h_sup.T @ h_sup) / float(n_sup)
            ht_y_sup = (h_sup.T @ y_sup_norm) / float(n_sup)
            reg_scaled = self.reg_lambda / float(n_sup)
            cur_obj = self._objective(w_out, h_sup, y_sup_norm, h_col, collocation_records, chol_cache)

            for _ in range(self.n_gn_iters):
                n_params = self.hidden_dim * n_lat
                hess = np.zeros((n_params, n_params), dtype=float)
                grad = np.zeros(n_params, dtype=float)

                for k in range(n_lat):
                    sl = slice(k * self.hidden_dim, (k + 1) * self.hidden_dim)
                    hess[sl, sl] += ht_h_sup + reg_scaled * np.eye(self.hidden_dim)
                    grad[sl] += ht_h_sup @ w_out[:, k] - ht_y_sup[:, k] + reg_scaled * w_out[:, k]

                z_col_pred = (h_col @ w_out) * self._y_std + self._y_mean
                weight_p = self.phys_lambda / float(n_col)

                for i, rec in enumerate(collocation_records):
                    mask = rec.mask & self.active_obs_mask
                    if not np.any(mask):
                        continue
                    l_sub = chol_cache[tuple(bool(b) for b in mask)]
                    z_i = z_col_pred[i]
                    g_pred = reduced_forward_observables(z_i, rec.design, self.constants)
                    r_white = solve_triangular(l_sub, g_pred[mask] - rec.values[mask], lower=True)
                    jac_z = numerical_jacobian(z_i, rec.design, self.constants)[mask, :]
                    jac_white = solve_triangular(l_sub, jac_z, lower=True) * self._y_std[None, :]

                    jtj_z = jac_white.T @ jac_white
                    jtr_z = jac_white.T @ r_white
                    hi = h_col[i]
                    hi_outer = np.outer(hi, hi)
                    for a in range(n_lat):
                        sa = slice(a * self.hidden_dim, (a + 1) * self.hidden_dim)
                        grad[sa] += weight_p * jtr_z[a] * hi
                        for b in range(n_lat):
                            sb = slice(b * self.hidden_dim, (b + 1) * self.hidden_dim)
                            hess[sa, sb] += weight_p * jtj_z[a, b] * hi_outer

                hess += self.damping * np.eye(n_params)
                delta = -np.linalg.solve(hess, grad)
                if float(np.linalg.norm(delta)) < 1e-7:
                    break

                # Backtracking line search for objective-based step acceptance
                accepted = False
                for step_scale in (1.0, 0.5, 0.25, 0.1):
                    w_cand = w_out + (step_scale * delta).reshape(n_lat, self.hidden_dim).T
                    cand_obj = self._objective(w_cand, h_sup, y_sup_norm, h_col, collocation_records, chol_cache)
                    if cand_obj <= cur_obj:
                        w_out = w_cand
                        cur_obj = cand_obj
                        accepted = True
                        break
                if not accepted:
                    break

            updated_members.append((w_in, b_in, w_out))

        self._members = updated_members
        pred_mean, _ = self._raw_ensemble_stats(features)
        self._aleatoric_var = np.maximum(np.mean((targets - pred_mean) ** 2, axis=0), 1e-6)
        return self
