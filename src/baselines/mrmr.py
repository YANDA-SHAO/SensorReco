import numpy as np


def _residualize(a, context):
    a = np.asarray(a, dtype=float)
    if context is None:
        return a
    c = np.asarray(context, dtype=float)
    design = np.column_stack([np.ones(len(c)), c])
    return a - design @ np.linalg.lstsq(design, a, rcond=None)[0]


def _unit_columns(a):
    a = a - a.mean(axis=0, keepdims=True)
    norm = np.sqrt((a * a).sum(axis=0, keepdims=True))
    norm[norm < 1e-12] = 1.0
    return a / norm


def grouped_mrmr(x_train: np.ndarray, y_train: np.ndarray, *, context_train=None,
                 redundancy_weight: float = 1.0) -> np.ndarray:
    """Grouped relevance-minus-redundancy greedy ranking for regression."""
    x = np.asarray(x_train, dtype=float)
    n, m, f = x.shape
    xf = _unit_columns(_residualize(x.reshape(n, m * f), context_train))
    y = np.asarray(y_train)
    y = y[:, None] if y.ndim == 1 else y
    yu = _unit_columns(_residualize(y, context_train))
    relevance = ((xf.T @ yu) ** 2).reshape(m, f, -1).mean(axis=(1, 2))
    corr2 = (xf.T @ xf) ** 2
    redundancy = corr2.reshape(m, f, m, f).mean(axis=(1, 3))
    np.fill_diagonal(redundancy, 0)
    relevance /= max(relevance.max(), 1e-12)
    redundancy /= max(redundancy.max(), 1e-12)
    selected, remaining = [], list(range(m))
    while remaining:
        score = {j: relevance[j] - redundancy_weight * (redundancy[j, selected].mean() if selected else 0.0) for j in remaining}
        chosen = min(remaining, key=lambda j: (-score[j], j))
        selected.append(chosen)
        remaining.remove(chosen)
    return np.asarray(selected, dtype=int)
