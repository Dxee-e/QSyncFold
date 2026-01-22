from random import seed
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

def run_sequence(sequence):
    result = []
    for net in models:
        net.mc.__init__()
        _, out = net(sequence)
        out = out.detach().numpy()
        result.append(out)
    result = np.stack(result, axis=0)
    return result

all_amino_acids = pd.read_csv('./amino_acids_infos.csv')['name'].tolist()
length = 9
num_try = 50000
rng = np.random.default_rng(seed=55555)
all_inputs_sequence = [''.join(rng.choice(all_amino_acids, length)) for _ in range(num_try)]

results = {}
for i in tqdm(range(num_try)):
    result = run_sequence(all_inputs_sequence[i])
    results[all_inputs_sequence[i]] = result

    if i % 1000 == 0 and i != 0:
        np.savez(f'./results_chunk/QNN_results_{i}.npz', results=results)
        results = {}

# np.savez('QNN_results.npz', results=results)
# combine after