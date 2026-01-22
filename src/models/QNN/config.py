import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch


class ModelConfig:
    def __init__(self):
        # amino acids infos
        self.amino_acids_infos = pd.read_csv("amino_acids_infos.csv")
        self.aa_name2idx = {
            row["name"]: row["index"] for _, row in self.amino_acids_infos.iterrows()
        }
        self.num_amino_acids = len(self.amino_acids_infos)

        # dataset info
        self.dataset_min_length = 5
        self.dataset_max_length = 9
        self.all_kfold = 5
        self.cur_val_kfold: Optional[int] = None

        # tensor type
        self.dtype_bit = 32
        assert self.dtype_bit in (32, 64)
        self.dtype_int = torch.int32 if self.dtype_bit == 32 else torch.int64
        self.dtype_float = torch.float32 if self.dtype_bit == 32 else torch.float64

        # device
        self.device = "cpu"
        self.quantum_device = "lightning.qubit"
        self.diff_method = "adjoint"

        # random generator
        self.seed = 55555
        self.np_generator = np.random.default_rng(seed=self.seed)
        self.torch_generator = torch.Generator(device=self.device).manual_seed(
            self.seed
        )
        self.num_try_random = 20

        # init const
        self.init_self_strength = 0.5
        self.qc_k_num = self.dataset_max_length // 2 + 1

        # optimizer
        self.learning_rate = 0.001
        self.epoch = 200

        # print("please setup validation kfold and run init.")

    def init(self, val_kfold: int):
        self.cur_val_kfold = val_kfold
        # load distribution
        distribution_path = f"./preprocess_data/distribution_min{self.dataset_min_length}_max{self.dataset_max_length}_val{self.cur_val_kfold}.npy"
        if not os.path.exists(distribution_path):
            import preprocess

            preprocess.run(
                self.dataset_min_length,
                self.dataset_max_length,
                self.cur_val_kfold,
                self.all_kfold,
            )

        self.distribution = np.load(distribution_path, allow_pickle=True).item()
        self.bone_length_range = (
            torch.tensor(
                self.distribution["min"] - 0.02,
                dtype=self.dtype_float,
                device=self.device,
            ),
            torch.tensor(
                self.distribution["max"] + 0.02,
                dtype=self.dtype_float,
                device=self.device,
            ),
        )
        self.bone_length_interval = (
            self.bone_length_range[1] - self.bone_length_range[0]
        )

    def trans_aa_name2idx(self, aa_name: str) -> List[int]:
        return [self.aa_name2idx[i] for i in aa_name]

    def get_distribution(self, aa0, aa1):
        aa0, aa1 = min(aa0, aa1), max(aa0, aa1)
        return self.distribution[(aa0, aa1)]

    def init_best_saved_epoch(self) -> int:
        df = pd.read_csv(f'result_val{self.cur_val_kfold}.csv')
        min_loss_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
        self.cur_epoch = min_loss_epoch
        return min_loss_epoch

model_config = ModelConfig()