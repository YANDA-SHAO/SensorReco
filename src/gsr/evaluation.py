from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from .story7 import PAPER_STORY7_PROTOCOL, fit_story7_mlp


def design_matrix(x: np.ndarray, sensors, context: np.ndarray | None = None) -> np.ndarray:
    chosen = np.sort(np.asarray(sensors, dtype=int))
    out = np.asarray(x)[:, chosen].reshape(len(x), -1)
    return out if context is None else np.concatenate([out, np.asarray(context)], axis=1)


def default_regressor():
    # LSQR is stable for the strongly correlated, high-dimensional sensor
    # descriptor matrices encountered near the full budget.
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="lsqr"))


def topk_retrain_curve(x, y, split, ranking, *, context=None, estimator=None, budgets=None) -> pd.DataFrame:
    """Retrain a fresh model for every nested top-k configuration."""
    split, ranking = np.asarray(split).astype(str), np.asarray(ranking, dtype=int)
    train, val = split == "train", split == "val"
    rows = []
    requested = list(range(1, len(ranking) + 1)) if budgets is None else sorted({int(k) for k in budgets})
    if not requested or requested[0] < 1 or requested[-1] > len(ranking):
        raise ValueError("budgets must lie within 1..number of ranked sensors")
    for k in requested:
        design = design_matrix(x, ranking[:k], context)
        model = clone(default_regressor() if estimator is None else estimator)
        model.fit(design[train], np.asarray(y)[train])
        rows.append({"budget": k, "validation_rmse": mean_squared_error(y[val], model.predict(design[val])) ** 0.5})
    return pd.DataFrame(rows)


def paired_story7_topk_curve(x, y, split, ranking, *, context=None, budgets=None,
                             base_seed=42, retrain_reps=5, validation_seed=20260708,
                             utility_fraction=.50, device="cpu", protocol=PAPER_STORY7_PROTOCOL,
                             evaluate_test=True):
    """Paper protocol: fixed checkpoint/utility validation split and paired MLP retraining."""
    split, ranking = np.asarray(split).astype(str), np.asarray(ranking, int)
    tr, va, test = np.flatnonzero(split == "train"), np.flatnonzero(split == "val"), np.flatnonzero(split == "test")
    cp, utility = train_test_split(va, test_size=int(round(len(va)*utility_fraction)),
                                   random_state=validation_seed, shuffle=True)
    requested = range(1, len(ranking)+1) if budgets is None else sorted(set(map(int, budgets)))
    rows = []
    for k in requested:
        design = design_matrix(x, ranking[:k], context)
        for rep in range(retrain_reps):
            seed = int(base_seed + 100_000 + rep)
            fit = fit_story7_mlp(design[tr], np.asarray(y)[tr], design[cp], np.asarray(y)[cp],
                                 seed=seed, device=device, protocol=protocol)
            pred = fit.predict(design[utility])
            row = {"budget": k, "retrain_rep": rep, "train_seed": seed,
                   "validation_rmse": mean_squared_error(np.asarray(y)[utility], pred) ** .5,
                   "best_epoch": fit.best_epoch}
            if evaluate_test and len(test):
                test_pred = fit.predict(design[test])
                row["held_out_test_rmse"] = mean_squared_error(np.asarray(y)[test], test_pred) ** .5
            rows.append(row)
    return pd.DataFrame(rows)
