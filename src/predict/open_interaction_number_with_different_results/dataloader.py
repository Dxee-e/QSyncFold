"""dataloader
Offline load
Range [a, b], include a and b"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch import Tensor, float32, float64, int32, int64, tensor

from config import model_config


class DataLoader:
    """DataLoader
    item return with (pdb_id, length, sequence, amino_acid_ids, structure)"""

    def __init__(
        self,
        min_length: int,
        max_length: int,
        dataset_type: str,
        select_fold: Optional[Union[int, List[int]]],
    ):
        assert dataset_type in ("train", "test")
        self.data_dir = "../../../data/"
        index = pd.read_csv(os.path.join(self.data_dir, "dataset_index.csv"))
        index = index[
            (index["dataset_type"] == dataset_type)
            & (index["length"] >= min_length)
            & (index["length"] <= max_length)
        ]
        if select_fold is None and dataset_type == "train":
            raise ValueError("kfold must be specified for training dataset")

        kfold = [select_fold] if isinstance(select_fold, int) else select_fold
        self.index = index[index["kfold"].isin(kfold)] if kfold is not None else index

        # load data (pdb id, length, sequence, structure)
        data: List[Tuple[str, int, str, Tensor]] = []
        for _, row in self.index.iterrows():
            pdb_id = row["pdb_id"]
            length = row["length"]
            npz = np.load(os.path.join(self.data_dir, "FormatData", f"{pdb_id}.npz"))
            sequence = "".join(npz["sequence"])
            structure = tensor(npz["coord"], dtype=model_config.dtype_float).to(
                model_config.device
            )
            data.append(
                (
                    pdb_id,
                    length,
                    sequence,
                    structure,
                )
            )
        self.data = data

        # random index warp
        self.idx_wrapped = np.array(range(len(self.data)), dtype=np.int32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :returns (pdb id, length, sequence, structures)
        """
        row = self.data[self.idx_wrapped[idx]]
        return row[0], row[1], row[2], row[3].to(model_config.device)

    def shuffle(self):
        """shuffle data randomly"""
        model_config.np_generator.shuffle(self.idx_wrapped)
