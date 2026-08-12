from __future__ import annotations

import math
import torch
from torch import nn


class StochasticSensorGate(nn.Module):
    """One stochastic gate for the complete representation of each sensor."""

    def __init__(self, n_sensors: int, sigma: float = 0.5, mu_init: float = 0.0):
        super().__init__()
        if n_sensors < 1 or sigma <= 0:
            raise ValueError("n_sensors and sigma must be positive")
        self.sigma = float(sigma)
        self.mu = nn.Parameter(torch.full((int(n_sensors),), float(mu_init)))

    def probabilities(self) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(self.mu / self.sigma / math.sqrt(2.0)))

    def expected_active(self) -> torch.Tensor:
        return self.probabilities().sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != len(self.mu):
            raise ValueError("x must have shape [samples, sensors, features]")
        raw = self.mu + self.sigma * torch.randn_like(self.mu) if self.training else self.mu
        z = raw.clamp(0.0, 1.0)
        return x * z.view(1, -1, 1)
