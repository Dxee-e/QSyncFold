import os
from typing import Tuple

import numpy as np
import pandas as pd

MAX_LENGTH = 999999


class DataLoader:
    """Load data"""

    def __init__(
        self, dataset_type: str = "test", min_len: int = 0, max_len: int = MAX_LENGTH
    ):
        """load range include both min_len and max_len"""
        self.data_dir = "../../../data/FormatData/"

        idx = pd.read_csv("../../../data/dataset_index.csv")
        idx = idx[
            (idx["dataset_type"] == dataset_type)
            & (idx["length"] >= min_len)
            & (idx["length"] <= max_len)
        ]
        self.dataset_index = idx

    def __getitem__(self, item) -> Tuple[str, int, str, np.ndarray]:
        """
        Get item
        :param item: int index
        :return: (pdb id, length, sequence, coord)
        """
        pdb_id = self.dataset_index.iloc[item]["pdb_id"]
        try:
            data = np.load(os.path.join(self.data_dir, f"{pdb_id}.npz"))
        except Exception as e:
            raise IOError(f"Error loading npz {pdb_id}: {e}")

        length = len(data["sequence"])
        sequence = "".join(data["sequence"])
        coord = data["coord"]
        pdb_id = data["pdb_id"]
        return pdb_id, length, sequence, coord

    def __len__(self):
        return len(self.dataset_index)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
