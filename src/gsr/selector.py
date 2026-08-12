from __future__ import annotations

from dataclasses import dataclass
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .gates import StochasticSensorGate
from .story7 import PAPER_STORY7_PROTOCOL, Story7MLP


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class _GatedMLP(nn.Module):
    def __init__(self, n_sensors: int, features: int, context: int, output: int, hidden: int, sigma: float):
        super().__init__()
        self.gate = StochasticSensorGate(n_sensors, sigma=sigma)
        self.predictor = Story7MLP(n_sensors * features + context, output, PAPER_STORY7_PROTOCOL)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        flat = self.gate(x).flatten(1)
        if context is not None:
            flat = torch.cat([flat, context], dim=1)
        return self.predictor(flat)


@dataclass(frozen=True)
class GSRResult:
    ranking: np.ndarray
    probabilities: np.ndarray
    best_validation_mse: float


class GSRRegressor:
    """Small reference implementation of regression GSR.

    The validation set controls early stopping. Test data are intentionally not
    accepted by this API.
    """

    def __init__(self, *, lambda_gsr: float = 1e-4, sigma: float = 0.5, hidden: int = 64,
                 lr: float = 1e-3, gate_lr: float = 2e-3,
                 lambda_warmup_epochs: int = 30, lambda_ramp_epochs: int = 70,
                 epochs: int = 500, patience: int = 60, seed: int = 42,
                 batch_size: int = 512, weight_decay: float = 1e-5, device: str = "cpu"):
        self.lambda_gsr, self.sigma, self.hidden = float(lambda_gsr), float(sigma), int(hidden)
        self.lr, self.gate_lr = float(lr), float(gate_lr)
        self.lambda_warmup_epochs = int(lambda_warmup_epochs)
        self.lambda_ramp_epochs = int(lambda_ramp_epochs)
        self.epochs, self.patience, self.seed = int(epochs), int(patience), int(seed)
        self.batch_size, self.weight_decay = int(batch_size), float(weight_decay)
        self.device = str(device)
        self.result_: GSRResult | None = None

    def _lambda_scale(self, epoch: int) -> float:
        """Warm up and then linearly ramp the fixed sparsity coefficient."""
        if epoch <= self.lambda_warmup_epochs:
            return 0.0
        if self.lambda_ramp_epochs <= 0:
            return 1.0
        return min(1.0, (epoch - self.lambda_warmup_epochs) / self.lambda_ramp_epochs)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
            context_train: np.ndarray | None = None, context_val: np.ndarray | None = None) -> GSRResult:
        _seed(self.seed)
        x_train, x_val = np.asarray(x_train, np.float32), np.asarray(x_val, np.float32)
        mean, std = x_train.mean(axis=0, keepdims=True), x_train.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        xt, xv = map(lambda a: torch.as_tensor(a, dtype=torch.float32), ((x_train-mean)/std, (x_val-mean)/std))
        yt, yv = map(lambda a: torch.as_tensor(a, dtype=torch.float32), (y_train, y_val))
        if yt.ndim == 1: yt, yv = yt[:, None], yv[:, None]
        if context_train is None:
            ct = cv = None
        else:
            context_train, context_val = np.asarray(context_train, np.float32), np.asarray(context_val, np.float32)
            cm, cs = context_train.mean(0, keepdims=True), context_train.std(0, keepdims=True); cs[cs < 1e-6] = 1.0
            ct = torch.as_tensor((context_train-cm)/cs, dtype=torch.float32)
            cv = torch.as_tensor((context_val-cm)/cs, dtype=torch.float32)
        model = _GatedMLP(xt.shape[1], xt.shape[2], 0 if ct is None else ct.shape[1], yt.shape[1], self.hidden, self.sigma).to(self.device)
        optimizer = torch.optim.AdamW([
            {"params": model.predictor.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": [model.gate.mu], "lr": self.gate_lr, "weight_decay": 0.0},
        ])
        tensors = (xt, yt) if ct is None else (xt, ct, yt)
        loader = DataLoader(TensorDataset(*tensors), batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        best, state, stale = float("inf"), None, 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            for batch in loader:
                xb, cb, yb = (batch[0], None, batch[1]) if ct is None else batch
                xb, yb = xb.to(self.device), yb.to(self.device)
                cb = None if cb is None else cb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                penalty = self.lambda_gsr * self._lambda_scale(epoch) * model.gate.expected_active() / xt.shape[1]
                loss = nn.functional.mse_loss(model(xb, cb), yb) + penalty
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            model.eval()
            with torch.no_grad(): val = float(nn.functional.mse_loss(model(xv.to(self.device), None if cv is None else cv.to(self.device)), yv.to(self.device)))
            if val < best - 1e-10:
                best, state, stale = val, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience: break
        if state is None: raise RuntimeError("training did not produce a checkpoint")
        model.load_state_dict(state)
        probabilities = model.gate.probabilities().detach().cpu().numpy()
        ranking = np.lexsort((np.arange(len(probabilities)), -probabilities))
        self.result_ = GSRResult(ranking, probabilities, best)
        return self.result_
