import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from model import Model
from config import model_config

all_kfold = 5
models = []
for kfold in range(all_kfold):
    model_config.init(kfold)
    epoch = model_config.init_best_saved_epoch()
    net = Model(model_config)
    net.eval()
    net.load_state_dict(torch.load(f"./saved_model/epoch{epoch}_val{kfold}.pth", weights_only=True))
    models.append(net)

def run_sequence(sequence, work_interaction):
    result = []
    for net in models:
        net.mc.__init__()
        out = net(sequence, work_interaction).detach().numpy()
        result.append(out)
    result = np.stack(result, axis=0)
    return result


sequence = 'IPMSIPPEVK'

results = {} 
N = len(sequence)
results[tuple(list(range(N)))] = run_sequence(sequence, list(range(N)))
temp_list = []
for i in tqdm(range(N)):
    temp_list.append(i)
    results[tuple(temp_list)] = run_sequence(sequence, temp_list)

np.savez('QNN_results.npz', results=results)
