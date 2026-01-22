import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataloader import DataLoader
from model import Model
from config import model_config
import pandas as pd
import time
import tracemalloc


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

results = {
    'time': [],
    'memory': [],
}
for i, (pdb_id, length, sequence, structure) in tqdm(enumerate(dl), total=len(dl)):
    r_times, r_memory = [], []
    for net in models:
        tracemalloc.start()
        start_time = time.time()
        try:
            out = net(sequence)
        finally:
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        r_times.append(end_time - start_time)
        r_memory.append(peak / (1024**2))  # MB
    results['time'].append(np.mean(r_times))
    results['memory'].append(np.mean(r_memory))

results = pd.DataFrame(results)
results.to_csv('./time_memory.csv', index=False)
