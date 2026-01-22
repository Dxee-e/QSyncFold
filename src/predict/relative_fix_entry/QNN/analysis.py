import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from model import Model
from config import model_config

all_kfold = 5
p_list = []
for kfold in range(all_kfold):
    model_config.init(kfold)
    epoch = model_config.init_best_saved_epoch()
    parameters = torch.load(f"./saved_model/epoch{epoch}_val{kfold}.pth", weights_only=True)
    p = parameters['interaction_strength_k']
    p_list.append(p)
p = torch.stack(p_list, dim=0).numpy()
p_mean = np.mean(p, axis=0)
p_std = np.std(p, axis=0)
print(p_mean, p_std)
print(p)

df = {
    'rel_l1_dist': [0],
    'value': [0],
    'min_v': [0],
    'max_v': [0],
}

for i in range(1, p_mean.shape[0]):
    df['rel_l1_dist'].append(i)
    df['value'].append(p[:,i].mean())
    df['min_v'].append(p[:,i].min())
    df['max_v'].append(p[:,i].max())

    df['rel_l1_dist'].append(-i)
    df['value'].append(p[:,i].mean())
    df['min_v'].append(p[:,i].min())
    df['max_v'].append(p[:,i].max())

df = pd.DataFrame(df)
df = df.sort_values(by='rel_l1_dist')
df.to_csv('interaction_strength_k.csv', index=False)