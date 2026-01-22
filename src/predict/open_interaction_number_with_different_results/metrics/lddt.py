"""LDDT
"""

from typing import List

import numpy as np


def lddt(native_coords: np.ndarray, predicted_coords:np.ndarray, cutoff:float=15, thresholds:List[float]=[0.5, 1, 2, 4]):
    """LDDT"""

    def calculate_distance_matrix(coords):
        dists = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))
        return dists

    native_dists = calculate_distance_matrix(native_coords)
    predicted_dists = calculate_distance_matrix(predicted_coords)

    lddt_scores = []
    N = len(native_coords)

    for i in range(N):
        valid_pairs = 0
        score_sum = 0

        for j in range(N):
            if i == j or abs(i - j) < 1:  # skip self and neighbors
                continue

            if native_dists[i, j] > cutoff: # skip distance>cutoff
                continue

            valid_pairs += 1
            diff = abs(native_dists[i, j] - predicted_dists[i, j])

            if diff < thresholds[0]:
                score_sum += 1.0
            elif diff < thresholds[1]:
                score_sum += 0.8
            elif diff < thresholds[2]:
                score_sum += 0.6
            elif diff < thresholds[3]:
                score_sum += 0.4

        if valid_pairs > 0:
            lddt_scores.append(score_sum / valid_pairs)
        else:
            lddt_scores.append(0.0)

    return np.mean(lddt_scores)