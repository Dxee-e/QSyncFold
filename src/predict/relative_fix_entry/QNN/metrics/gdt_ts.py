"""GDT-TS"""

import numpy as np

THRESHOLDS = [1.0, 2.0, 4.0, 8.0]


def gdt_ts(source: np.ndarray, target: np.ndarray) -> float:
    """GDT-TS"""
    assert source.shape == target.shape

    num_atoms = len(source)

    dist = np.sqrt(np.sum((target - source) ** 2, axis=-1))

    atom_match_list = []
    for threshold in THRESHOLDS:
        matches = np.sum(dist <= threshold)
        atom_match_list.append(matches)

    gdt_ts_score = np.mean(atom_match_list) / num_atoms
    return gdt_ts_score
