import numpy as np
from scipy.linalg import qr


def qr_response_diversity(x_train: np.ndarray) -> np.ndarray:
    """Train-only pivoted-QR response-diversity ranking."""
    x = np.asarray(x_train, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("x_train must be [samples, sensors, features]")
    centered = x - x.mean(axis=0, keepdims=True)
    scale = centered.std(axis=(0, 1), keepdims=True)
    scale[scale < 1e-12] = 1.0
    matrix = (centered / scale).transpose(0, 2, 1).reshape(-1, x.shape[1])
    return np.asarray(qr(matrix, pivoting=True, mode="economic")[2], dtype=int)
