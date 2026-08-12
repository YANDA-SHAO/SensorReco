import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

from gsr.evaluation import default_regressor, design_matrix
from gsr.story7 import fit_story7_mlp


def permutation_ranking(x_train, y_train, x_val, y_val, *, context_train=None, context_val=None,
                        mode: str = "global", high_error_fraction: float = 0.20,
                        repeats: int = 5, seed: int = 42, estimator=None) -> np.ndarray:
    """Dense-model joint sensor-block permutation ranking."""
    if mode not in {"global", "high_error"}:
        raise ValueError("mode must be global or high_error")
    m = np.asarray(x_train).shape[1]
    model = clone(default_regressor() if estimator is None else estimator)
    model.fit(design_matrix(x_train, range(m), context_train), y_train)
    dense = design_matrix(x_val, range(m), context_val)
    pred = model.predict(dense)
    indices = np.arange(len(y_val))
    if mode == "high_error":
        error = np.mean((np.asarray(pred) - np.asarray(y_val)) ** 2, axis=1) if np.asarray(y_val).ndim > 1 else (pred - y_val) ** 2
        n = max(1, int(np.ceil(high_error_fraction * len(indices))))
        indices = np.lexsort((indices, -error))[:n]
    baseline = mean_squared_error(np.asarray(y_val)[indices], np.asarray(pred)[indices])
    rng, scores = np.random.default_rng(seed), np.zeros(m)
    for sensor in range(m):
        losses = []
        for _ in range(repeats):
            xp = np.array(x_val, copy=True)
            perm = rng.permutation(indices)
            xp[indices, sensor, :] = xp[perm, sensor, :]
            perturbed = model.predict(design_matrix(xp, range(m), context_val))
            losses.append(mean_squared_error(np.asarray(y_val)[indices], perturbed[indices]))
        scores[sensor] = np.mean(losses) - baseline
    return np.lexsort((np.arange(m), -scores))


def story7_permutation_ranking(x_train, y_train, x_checkpoint, y_checkpoint,
                               x_utility, y_utility, *, context_train=None,
                               context_checkpoint=None, context_utility=None, mode="global",
                               high_error_quantile=.80, repeats=5, seed=42, device="cpu"):
    """Paper PFI using one dense Story-7 MLP and complete-block permutations."""
    m = np.asarray(x_train).shape[1]
    full = list(range(m))
    dtr = design_matrix(x_train, full, context_train); dcp = design_matrix(x_checkpoint, full, context_checkpoint)
    fit = fit_story7_mlp(dtr, y_train, dcp, y_checkpoint, seed=seed, device=device)
    base_design = design_matrix(x_utility, full, context_utility); pred = fit.predict(base_design)
    idx = np.arange(len(y_utility))
    if mode == "high_error":
        err = np.mean((np.asarray(pred)-np.asarray(y_utility))**2, axis=1)
        count = max(1, int(np.ceil((1-high_error_quantile)*len(idx))))
        idx = np.lexsort((idx, -err))[:count]
    baseline = mean_squared_error(np.asarray(y_utility)[idx], pred[idx])
    rng, scores = np.random.default_rng(seed), np.zeros(m)
    for sensor in range(m):
        losses = []
        for _ in range(repeats):
            xp = np.array(x_utility, copy=True); perm = rng.permutation(idx)
            xp[idx, sensor] = xp[perm, sensor]
            pp = fit.predict(design_matrix(xp, full, context_utility))
            losses.append(mean_squared_error(np.asarray(y_utility)[idx], pp[idx]))
        scores[sensor] = np.mean(losses)-baseline
    return np.lexsort((np.arange(m), -scores))
