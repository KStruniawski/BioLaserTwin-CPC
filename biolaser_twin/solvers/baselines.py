"""Data-driven baselines (MeanOnly, MLP ensemble, FrameCNN, FrameViT, HybridCNNFeatures) with minibatching & early stopping."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MeanOnlySolver:
    """Trivial baseline predicting training target mean and std."""

    def __init__(self) -> None:
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None

    def fit(self, targets: np.ndarray) -> "MeanOnlySolver":
        self._y_mean = np.mean(targets, axis=0)
        self._y_std = np.maximum(np.std(targets, axis=0), 1e-6)
        return self

    def predict_with_uncertainty(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._y_mean is not None and self._y_std is not None
        mean = np.repeat(self._y_mean[None, :], n_samples, axis=0)
        std = np.repeat(self._y_std[None, :], n_samples, axis=0)
        return mean, std


def _train_nll_loop(
    model_params: list,
    forward_fn,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor | None = None,
    y_va: torch.Tensor | None = None,
    max_epochs: int = 120,
    batch_size: int = 32,
    lr: float = 3e-3,
    patience: int = 15,
    extra_tr: torch.Tensor | None = None,
    extra_va: torch.Tensor | None = None,
) -> None:
    """Shared minibatch Adam + early-stopping Gaussian NLL training loop."""
    ds = TensorDataset(x_tr, y_tr) if extra_tr is None else TensorDataset(x_tr, extra_tr, y_tr)
    loader = DataLoader(ds, batch_size=min(batch_size, len(x_tr)), shuffle=True)
    opt = torch.optim.AdamW(model_params, lr=lr, weight_decay=1e-4)

    best_val = float("inf")
    best_state = [p.detach().clone() for p in model_params]
    wait = 0
    out_dim = y_tr.shape[1]

    for _ in range(max_epochs):
        for batch in loader:
            opt.zero_grad()
            if extra_tr is None:
                xb, yb = batch
                raw = forward_fn(xb)
            else:
                xb, eb, yb = batch
                raw = forward_fn(xb, eb)
            mean = raw[:, :out_dim]
            var = nn.functional.softplus(raw[:, out_dim:]) + 1e-4
            loss = torch.mean(0.5 * (torch.log(var) + ((yb - mean) ** 2) / var))
            loss.backward()
            opt.step()

        if x_va is not None and y_va is not None:
            with torch.no_grad():
                raw_v = forward_fn(x_va) if extra_va is None else forward_fn(x_va, extra_va)
                mv = raw_v[:, :out_dim]
                vv = nn.functional.softplus(raw_v[:, out_dim:]) + 1e-4
                val_loss = float(torch.mean(0.5 * (torch.log(vv) + ((y_va - mv) ** 2) / vv)))
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = [p.detach().clone() for p in model_params]
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if x_va is not None:
        with torch.no_grad():
            for p, bp in zip(model_params, best_state):
                p.copy_(bp)


class TabularMLPSolver:
    """PyTorch MLP ensemble on extracted observable features with minibatching & early stopping."""

    def __init__(
        self,
        hidden_dim: int = 64,
        epochs: int = 120,
        lr: float = 5e-3,
        n_ensemble: int = 3,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.n_ensemble = n_ensemble
        self.seed = seed
        self._nets: list[nn.Module] = []
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None
        self._out_dim: int = 0

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        val_features: np.ndarray | None = None,
        val_targets: np.ndarray | None = None,
    ) -> "TabularMLPSolver":
        self._x_mean = np.mean(features, axis=0)
        self._x_std = np.maximum(np.std(features, axis=0), 1e-6)
        self._y_mean = np.mean(targets, axis=0)
        self._y_std = np.maximum(np.std(targets, axis=0), 1e-6)

        x_norm = torch.tensor((features - self._x_mean) / self._x_std, dtype=torch.float32)
        y_norm = torch.tensor((targets - self._y_mean) / self._y_std, dtype=torch.float32)
        x_va_t = torch.tensor((val_features - self._x_mean) / self._x_std, dtype=torch.float32) if val_features is not None else None
        y_va_t = torch.tensor((val_targets - self._y_mean) / self._y_std, dtype=torch.float32) if val_targets is not None else None

        self._out_dim = targets.shape[1]
        self._nets = []

        for m_idx in range(self.n_ensemble):
            torch.manual_seed(self.seed + m_idx)
            net = nn.Sequential(
                nn.Linear(features.shape[1], self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2 * self._out_dim),
            )
            net.train()
            _train_nll_loop(
                list(net.parameters()),
                net,
                x_norm,
                y_norm,
                x_va=x_va_t,
                y_va=y_va_t,
                max_epochs=self.epochs,
                lr=self.lr,
            )
            net.eval()
            self._nets.append(net)
        return self

    def predict_with_uncertainty(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._nets or self._x_mean is None or self._y_mean is None or self._y_std is None:
            raise RuntimeError("TabularMLPSolver must be fitted before prediction.")
        assert self._x_std is not None
        x_norm = torch.tensor((features - self._x_mean) / self._x_std, dtype=torch.float32)
        means, vars_ = [], []
        with torch.no_grad():
            for net in self._nets:
                net.eval()
                raw = net(x_norm)
                m = raw[:, : self._out_dim].numpy() * self._y_std + self._y_mean
                v = (nn.functional.softplus(raw[:, self._out_dim :]).numpy() + 1e-4) * (self._y_std**2)
                means.append(m)
                vars_.append(v)
        m_stack = np.stack(means, axis=0)
        mean = np.mean(m_stack, axis=0)
        total_var = np.mean(np.stack(vars_, axis=0), axis=0) + np.var(m_stack, axis=0)
        return mean, np.sqrt(np.maximum(total_var, 1e-8))


class _SpectralBackboneMixin:
    """Shared intensity-preserving spectral preprocessing supporting single or dual excitation."""

    def _prepare_frames(self, frames: np.ndarray, fit_scale: bool = False) -> torch.Tensor:
        # Supports (N, n_exc, n_sweeps, n_spatial, n_px), (N, n_sweeps, n_spatial, n_px), or (N, n_spatial, n_px)
        if frames.ndim == 5:
            n_y = frames.shape[3]
            y_lo, y_hi = max(0, (3 * n_y) // 8), min(n_y, max(1, (5 * n_y) // 8))
            roi = np.mean(frames[:, :, :, y_lo:y_hi, :], axis=3)  # (N, n_exc, n_sweeps, n_px)
            x_arr = roi.reshape(frames.shape[0], -1, frames.shape[-1])
        elif frames.ndim == 4:
            n_y = frames.shape[2]
            y_lo, y_hi = max(0, (3 * n_y) // 8), min(n_y, max(1, (5 * n_y) // 8))
            x_arr = np.mean(frames[:, :, y_lo:y_hi, :], axis=2)
        elif frames.ndim == 3:
            n_y = frames.shape[1]
            y_lo, y_hi = max(0, (3 * n_y) // 8), min(n_y, max(1, (5 * n_y) // 8))
            x_arr = np.mean(frames[:, y_lo:y_hi, :], axis=1)[:, None, :]
        else:
            x_arr = frames[:, None, :]

        x_log = np.log1p(np.clip(x_arr, a_min=0.0, a_max=None))
        if fit_scale:
            self._frame_scale = max(1.0, float(np.percentile(x_log, 99)))
        scale = getattr(self, "_frame_scale", 10.0)
        return torch.tensor(x_log / scale, dtype=torch.float32)


class FrameCNNSolver(_SpectralBackboneMixin):
    """End-to-end spectral-preserving CNN baseline on full dual-excitation sweeps."""

    def __init__(self, epochs: int = 100, lr: float = 2e-3, seed: int = 42) -> None:
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self._net: nn.Module | None = None
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None
        self._out_dim: int = 0
        self._frame_scale: float = 10.0

    def fit(
        self,
        frames: np.ndarray,
        targets: np.ndarray,
        val_frames: np.ndarray | None = None,
        val_targets: np.ndarray | None = None,
    ) -> "FrameCNNSolver":
        torch.manual_seed(self.seed)
        x_t = self._prepare_frames(frames, fit_scale=True)
        self._y_mean = np.mean(targets, axis=0)
        self._y_std = np.maximum(np.std(targets, axis=0), 1e-6)
        y_norm = torch.tensor((targets - self._y_mean) / self._y_std, dtype=torch.float32)
        x_va = self._prepare_frames(val_frames, fit_scale=False) if val_frames is not None else None
        y_va = torch.tensor((val_targets - self._y_mean) / self._y_std, dtype=torch.float32) if val_targets is not None else None
        self._out_dim = targets.shape[1]

        in_ch = x_t.shape[1]
        self._net = nn.Sequential(
            nn.Conv1d(in_ch, 24, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(24, 48, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(64 * 32, 64),
            nn.SiLU(),
            nn.Linear(64, 2 * self._out_dim),
        )
        self._net.train()
        _train_nll_loop(
            list(self._net.parameters()),
            self._net,
            x_t,
            y_norm,
            x_va=x_va,
            y_va=y_va,
            max_epochs=self.epochs,
            lr=self.lr,
        )
        self._net.eval()
        return self

    def predict_with_uncertainty(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._net is None or self._y_mean is None or self._y_std is None:
            raise RuntimeError("FrameCNNSolver must be fitted before prediction.")
        self._net.eval()
        x_t = self._prepare_frames(frames, fit_scale=False)
        with torch.no_grad():
            raw = self._net(x_t)
            mean = raw[:, : self._out_dim].numpy() * self._y_std + self._y_mean
            std = np.sqrt(nn.functional.softplus(raw[:, self._out_dim :]).numpy() + 1e-4) * self._y_std
        return mean, std


class FrameViTSolver(_SpectralBackboneMixin):
    """Lightweight patch-based spectral Vision Transformer with positional embeddings and eval()."""

    def __init__(self, patch_size: int = 8, d_model: int = 32, epochs: int = 150, lr: float = 1e-3, seed: int = 42) -> None:
        self.patch_size = patch_size
        self.d_model = d_model
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self._proj: nn.Conv1d | None = None
        self._pos_emb: nn.Parameter | None = None
        self._enc: nn.TransformerEncoder | None = None
        self._head: nn.Linear | None = None
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None
        self._out_dim: int = 0
        self._frame_scale: float = 10.0

    def _forward(self, x_b: torch.Tensor) -> torch.Tensor:
        assert self._proj is not None and self._pos_emb is not None and self._enc is not None and self._head is not None
        tokens = self._proj(x_b).transpose(1, 2)
        tokens = tokens + self._pos_emb[:, : tokens.shape[1], :]
        pooled = torch.mean(self._enc(tokens), dim=1)
        return self._head(pooled)

    def fit(
        self,
        frames: np.ndarray,
        targets: np.ndarray,
        val_frames: np.ndarray | None = None,
        val_targets: np.ndarray | None = None,
    ) -> "FrameViTSolver":
        torch.manual_seed(self.seed)
        x_t = self._prepare_frames(frames, fit_scale=True)
        self._y_mean = np.mean(targets, axis=0)
        self._y_std = np.maximum(np.std(targets, axis=0), 1e-6)
        y_norm = torch.tensor((targets - self._y_mean) / self._y_std, dtype=torch.float32)
        x_va = self._prepare_frames(val_frames, fit_scale=False) if val_frames is not None else None
        y_va = torch.tensor((val_targets - self._y_mean) / self._y_std, dtype=torch.float32) if val_targets is not None else None
        self._out_dim = targets.shape[1]

        self._proj = nn.Conv1d(x_t.shape[1], self.d_model, kernel_size=self.patch_size, stride=self.patch_size)
        n_tokens = max(1, x_t.shape[-1] // self.patch_size)
        self._pos_emb = nn.Parameter(torch.zeros(1, n_tokens, self.d_model))
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=64, batch_first=True)
        self._enc = nn.TransformerEncoder(layer, num_layers=2)
        self._head = nn.Linear(self.d_model, 2 * self._out_dim)

        self._proj.train()
        self._enc.train()
        self._head.train()
        params = list(self._proj.parameters()) + [self._pos_emb] + list(self._enc.parameters()) + list(self._head.parameters())
        _train_nll_loop(
            params,
            self._forward,
            x_t,
            y_norm,
            x_va=x_va,
            y_va=y_va,
            max_epochs=self.epochs,
            lr=self.lr,
        )
        self._proj.eval()
        self._enc.eval()
        self._head.eval()
        return self

    def predict_with_uncertainty(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._proj is None or self._pos_emb is None or self._enc is None or self._head is None or self._y_mean is None or self._y_std is None:
            raise RuntimeError("FrameViTSolver must be fitted before prediction.")
        self._proj.eval()
        self._enc.eval()
        self._head.eval()
        x_t = self._prepare_frames(frames, fit_scale=False)
        with torch.no_grad():
            raw = self._forward(x_t)
            mean = raw[:, : self._out_dim].numpy() * self._y_std + self._y_mean
            std = np.sqrt(nn.functional.softplus(raw[:, self._out_dim :]).numpy() + 1e-4) * self._y_std
        return mean, std


class HybridCNNFeaturesSolver(_SpectralBackboneMixin):
    """Hybrid CNN + extracted observable features baseline (protocol §3)."""

    def __init__(self, epochs: int = 100, lr: float = 3e-3, seed: int = 42) -> None:
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self._cnn: nn.Module | None = None
        self._head: nn.Module | None = None
        self._f_mean: np.ndarray | None = None
        self._f_std: np.ndarray | None = None
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None
        self._out_dim: int = 0
        self._frame_scale: float = 10.0

    def _forward(self, x_b: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
        assert self._cnn is not None and self._head is not None
        return self._head(torch.cat([self._cnn(x_b), f_b], dim=1))

    def fit(
        self,
        frames: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        val_frames: np.ndarray | None = None,
        val_features: np.ndarray | None = None,
        val_targets: np.ndarray | None = None,
    ) -> "HybridCNNFeaturesSolver":
        torch.manual_seed(self.seed)
        x_t = self._prepare_frames(frames, fit_scale=True)
        self._f_mean = np.mean(features, axis=0)
        self._f_std = np.maximum(np.std(features, axis=0), 1e-6)
        f_t = torch.tensor((features - self._f_mean) / self._f_std, dtype=torch.float32)

        self._y_mean = np.mean(targets, axis=0)
        self._y_std = np.maximum(np.std(targets, axis=0), 1e-6)
        y_norm = torch.tensor((targets - self._y_mean) / self._y_std, dtype=torch.float32)
        x_va = self._prepare_frames(val_frames, fit_scale=False) if val_frames is not None else None
        f_va = torch.tensor((val_features - self._f_mean) / self._f_std, dtype=torch.float32) if val_features is not None else None
        y_va = torch.tensor((val_targets - self._y_mean) / self._y_std, dtype=torch.float32) if val_targets is not None else None
        self._out_dim = targets.shape[1]

        self._cnn = nn.Sequential(
            nn.Conv1d(x_t.shape[1], 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )
        self._head = nn.Sequential(
            nn.Linear(16 * 16 + features.shape[1], 64),
            nn.SiLU(),
            nn.Linear(64, 2 * self._out_dim),
        )
        self._cnn.train()
        self._head.train()
        params = list(self._cnn.parameters()) + list(self._head.parameters())
        _train_nll_loop(
            params,
            self._forward,
            x_t,
            y_norm,
            x_va=x_va,
            y_va=y_va,
            extra_tr=f_t,
            extra_va=f_va,
            max_epochs=self.epochs,
            lr=self.lr,
        )
        self._cnn.eval()
        self._head.eval()
        return self

    def predict_with_uncertainty(self, frames: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._cnn is None or self._head is None or self._f_mean is None or self._f_std is None or self._y_mean is None or self._y_std is None:
            raise RuntimeError("HybridCNNFeaturesSolver must be fitted before prediction.")
        self._cnn.eval()
        self._head.eval()
        x_t = self._prepare_frames(frames, fit_scale=False)
        f_t = torch.tensor((features - self._f_mean) / self._f_std, dtype=torch.float32)
        with torch.no_grad():
            raw = self._forward(x_t, f_t)
            mean = raw[:, : self._out_dim].numpy() * self._y_std + self._y_mean
            std = np.sqrt(nn.functional.softplus(raw[:, self._out_dim :]).numpy() + 1e-4) * self._y_std
        return mean, std
