from dataloader import DataLoader
import numpy as np
import torch
from scipy import stats
from typing import List
import pandas as pd

def run(min_length:int, max_length:int, val_kfold:int, all_kfold:int):
    train_kfold = [i for i in range(all_kfold) if i != val_kfold]
    dl = DataLoader(
        min_length=min_length,
        max_length=max_length,
        dataset_type="train",
        select_fold=train_kfold,
    )
    amino_acids_infos = pd.read_csv("amino_acids_infos.csv")
    aa_name2idx = {
        row["name"]: row["index"] for _, row in amino_acids_infos.iterrows()
    }
    num_aa = len(amino_acids_infos)
    stats_result = {(i, j): [] for i in range(num_aa) for j in range(num_aa)}
    for pdb_id, length, sequence, structure in dl:
        distance = torch.sqrt(
            torch.sum((structure.unsqueeze(0) - structure.unsqueeze(1)) ** 2, dim=-1)
        )
        for i in range(1, length):
            aa0 = aa_name2idx[sequence[i - 1]]
            aa1 = aa_name2idx[sequence[i]]
            aa0, aa1 = min(aa0, aa1), max(aa0, aa1)
            stats_result[(aa0, aa1)].append(distance[i - 1, i].item())
    all_stats = []
    for k, v in stats_result.items():
        if len(v) > 0:
            all_stats.extend(v)
    all_stats = np.array(all_stats)
    all_stats_min, all_stats_max = np.min(all_stats), np.max(all_stats)
    all_state_mu = np.mean(all_stats)
    std_mu = all_state_mu
    std_sigma = (all_stats_max - all_stats_min) / 4
    print(all_stats_min, all_stats_max, all_state_mu)

    result = {(i, j): None for i in range(num_aa) for j in range(num_aa)}
    for k, v in stats_result.items():
        if len(v) > 3:
            mu = np.mean(v)
            # sigma = np.std(v)
            sigma = (np.max(v) - np.min(v)) / 4

            if sigma < 0.000001:
                mu, sigma = std_mu, std_sigma
            else:
                # test for normal distribution
                if len(v) < 5000:  # Shapiro-Wilk Test
                    shapiro_test = stats.shapiro(v)
                    if shapiro_test.pvalue < 0.05:  # refuse
                        mu, sigma = std_mu, std_sigma
                else:  # Kolmogorov-Smirnov Test
                    ks_test = stats.kstest((v - np.mean(v)) / np.std(v), "norm")
                    if ks_test.pvalue < 0.05:  # refuse
                        mu, sigma = std_mu, std_sigma
            result[k] = (mu, sigma)
        else:
            result[k] = (std_mu, std_sigma)
    result["min"] = all_stats_min
    result["max"] = all_stats_max
    result["mean"] = all_state_mu
    np.save(f"./preprocess_data/distribution_min{min_length}_max{max_length}_val{val_kfold}.npy", result)
