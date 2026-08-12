import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

from gsr.evaluation import default_regressor, design_matrix
from gsr.story7 import fit_story7_mlp


def loro_ranking(x_train, y_train, x_val, y_val, *, context_train=None, context_val=None, estimator=None) -> np.ndarray:
    """Retrained leave-one-resource-out full-context ablation ranking."""
    m = np.asarray(x_train).shape[1]
    base = default_regressor() if estimator is None else estimator
    full = clone(base).fit(design_matrix(x_train, range(m), context_train), y_train)
    full_loss = mean_squared_error(y_val, full.predict(design_matrix(x_val, range(m), context_val)))
    scores = np.zeros(m)
    for omitted in range(m):
        kept = [j for j in range(m) if j != omitted]
        model = clone(base).fit(design_matrix(x_train, kept, context_train), y_train)
        scores[omitted] = mean_squared_error(y_val, model.predict(design_matrix(x_val, kept, context_val))) - full_loss
    return np.lexsort((np.arange(m), -scores))


def story7_loro_ranking(x_train, y_train, x_checkpoint, y_checkpoint, x_utility, y_utility,
                        *, context_train=None, context_checkpoint=None, context_utility=None,
                        seed=42, device="cpu"):
    """Paper retrained LORO; every ablation uses the same Story-7 MLP protocol."""
    m = np.asarray(x_train).shape[1]; full = list(range(m))
    def train_score(kept):
        fit = fit_story7_mlp(design_matrix(x_train, kept, context_train), y_train,
                             design_matrix(x_checkpoint, kept, context_checkpoint), y_checkpoint,
                             seed=seed, device=device)
        pred = fit.predict(design_matrix(x_utility, kept, context_utility))
        return mean_squared_error(y_utility, pred)
    dense = train_score(full); scores = np.zeros(m)
    for omitted in range(m): scores[omitted] = train_score([j for j in full if j != omitted])-dense
    return np.lexsort((np.arange(m), -scores))
