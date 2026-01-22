import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataloader import DataLoader
from model import Model
from config import model_config

all_kfold = 5
dl = DataLoader(min_length=model_config.dataset_min_length,
                max_length=model_config.dataset_max_length,
                dataset_type="test", select_fold=None)
models = []
for kfold in range(all_kfold):
    model_config.init(kfold)
    epoch = model_config.init_best_saved_epoch()
    net = Model(model_config)
    net.load_state_dict(torch.load(f"./saved_model/epoch{epoch}_val{kfold}.pth"))
    models.append(net)

results = {}
for i, (pdb_id, length, sequence, structure) in tqdm(enumerate(dl), total=len(dl)):
    stack_out = []
    for net in models:
        out = net(sequence)
        stack_out.append(out.detach().numpy())
    results[pdb_id] = np.stack(stack_out, axis=0)
np.savez("results.npz", **results)
