"""QAOA
run QAOA model.
Due to it cannot train, we will only use test dataset for benchmark.
"""

import os

import numpy as np
from tqdm import tqdm

from dataloader import DataLoader

from model import solve_problem

# length range are limited to [4, 9] due to device limitation
loader = DataLoader(dataset_type="test", min_len=5, max_len=9)


def isSkip(pdb_id: str) -> bool:
    cur_results = os.listdir("./result")
    if f"{pdb_id}.npz" in cur_results:
        return True
    return False


for pdb_id, length, sequence, coord in tqdm(loader):
    if isSkip(pdb_id):
        continue
    target = solve_problem(sequence)
    result = {
        "pdb_id": np.array(pdb_id, dtype=str),
        "length": np.array(length, dtype=int),
        "sequence": np.array(sequence, dtype=str),
        "source": coord,
        "target": target,
    }
    np.savez(f"./result/{pdb_id}.npz", **result)
