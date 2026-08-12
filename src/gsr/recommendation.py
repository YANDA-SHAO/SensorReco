from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    best_k: int
    best_utility: float
    selected_k: int
    retention: float


def select_operating_point(budgets, values, *, threshold: float = 0.95, higher_is_better: bool = True) -> OperatingPoint:
    """Select the smallest k retaining the requested fraction of best utility."""
    k, value = np.asarray(budgets, dtype=int), np.asarray(values, dtype=float)
    if k.ndim != 1 or value.shape != k.shape or len(k) == 0 or not np.isfinite(value).all():
        raise ValueError("budgets and values must be finite one-dimensional arrays")
    idx = int(np.argmax(value) if higher_is_better else np.argmin(value))
    best = float(value[idx])
    retention = value / best if higher_is_better else best / value
    eligible = np.flatnonzero(retention >= threshold)
    if not len(eligible): raise ValueError("no budget reaches the threshold")
    chosen = eligible[np.argmin(k[eligible])]
    return OperatingPoint(int(k[idx]), best, int(k[chosen]), float(retention[chosen]))


def one_standard_error_k(budgets, means, standard_deviations, n_repeats, *, higher_is_better: bool = True) -> int:
    """Smallest budget within one standard error of the best mean utility."""
    k, mean, sd = np.asarray(budgets, int), np.asarray(means, float), np.asarray(standard_deviations, float)
    n = np.asarray(n_repeats, float)
    best_index = int(np.argmax(mean) if higher_is_better else np.argmin(mean))
    tolerance = sd[best_index] / np.sqrt(n[best_index])
    eligible = mean >= mean[best_index] - tolerance if higher_is_better else mean <= mean[best_index] + tolerance
    return int(k[eligible].min())


def leave_one_seed_out_k(budgets, values_by_seed, *, threshold: float = 0.95, higher_is_better: bool = True) -> np.ndarray:
    """Recompute the threshold crossing while leaving out each training seed."""
    values = np.asarray(values_by_seed, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(budgets) or values.shape[0] < 2:
        raise ValueError("values_by_seed must be [seeds, budgets] with at least two seeds")
    return np.asarray([
        select_operating_point(budgets, np.delete(values, held, axis=0).mean(axis=0), threshold=threshold, higher_is_better=higher_is_better).selected_k
        for held in range(values.shape[0])
    ])
