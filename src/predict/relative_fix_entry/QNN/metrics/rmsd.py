"""RMSD"""
import numpy as np


def rmsd(x: np.ndarray, y: np.ndarray) -> float:
    """Root Mean Square Deviation"""
    assert x.shape == y.shape
    n = x.shape[0]
    return np.sqrt(np.sum(np.sum((x - y) ** 2)) / n)
