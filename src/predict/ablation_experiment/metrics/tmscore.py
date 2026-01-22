"""TM-Score"""
import numpy as np


def tmsocre(x: np.ndarray, y: np.ndarray) -> float:
    """TM-Score"""
    assert x.shape == y.shape
    Ln = x.shape[0]
    if Ln < 15:
        d0 = -1.8
    else:
        d0 = 1.24 * (Ln - 15) ** (1 / 3) - 1.8
    d = np.sqrt(np.sum((x - y) ** 2, axis=1))
    tm_score = np.sum(1 / (1 + (d / d0) ** 2)) / Ln
    return tm_score
