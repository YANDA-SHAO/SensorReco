from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Story7Protocol:
    batch_size: int = 512
    epochs: int = 220
    patience: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.10
    hidden_dims: tuple[int, int, int] = (512, 256, 128)
    eval_batch_size: int = 2048

    def to_dict(self):
        out = asdict(self); out["hidden_dims"] = list(self.hidden_dims); return out


PAPER_STORY7_PROTOCOL = Story7Protocol()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Story7MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, protocol: Story7Protocol = PAPER_STORY7_PROTOCOL):
        super().__init__()
        h1, h2, h3 = protocol.hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(protocol.dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(protocol.dropout),
            nn.Linear(h2, h3), nn.ReLU(), nn.Linear(h3, output_dim),
        )

    def forward(self, x): return self.net(x)


class Story7Fit:
    def __init__(self, model, scaler, best_val_loss, best_epoch, device, eval_batch_size):
        self.model, self.scaler = model, scaler
        self.best_val_loss, self.best_epoch = best_val_loss, best_epoch
        self.device, self.eval_batch_size = device, eval_batch_size

    def predict(self, x):
        xs = self.scaler.transform(np.asarray(x)).astype(np.float32)
        loader = DataLoader(torch.as_tensor(xs), batch_size=self.eval_batch_size)
        self.model.eval(); out = []
        with torch.no_grad():
            for xb in loader: out.append(self.model(xb.to(self.device)).cpu().numpy())
        return np.concatenate(out)


def fit_story7_mlp(x_train, y_train, x_checkpoint, y_checkpoint, *, seed=42,
                   device="cpu", protocol=PAPER_STORY7_PROTOCOL):
    set_seed(int(seed))
    scaler = StandardScaler().fit(np.asarray(x_train, dtype=np.float32))
    xt = scaler.transform(x_train).astype(np.float32)
    xv = scaler.transform(x_checkpoint).astype(np.float32)
    yt, yv = np.asarray(y_train, np.float32), np.asarray(y_checkpoint, np.float32)
    if yt.ndim == 1: yt, yv = yt[:, None], yv[:, None]
    model = Story7MLP(xt.shape[1], yt.shape[1], protocol).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay)
    gen = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(TensorDataset(torch.as_tensor(xt), torch.as_tensor(yt)),
                        batch_size=protocol.batch_size, shuffle=True, generator=gen)
    xv_t, yv_t = torch.as_tensor(xv, device=device), torch.as_tensor(yv, device=device)
    best, state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, protocol.epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb.to(device)), yb.to(device)); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val = float(nn.functional.mse_loss(model(xv_t), yv_t))
        if val < best:
            best, best_epoch, stale, state = val, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= protocol.patience: break
    if state is None: raise RuntimeError("Story-7 MLP produced no checkpoint")
    model.load_state_dict(state)
    return Story7Fit(model, scaler, best, best_epoch, device, protocol.eval_batch_size)
