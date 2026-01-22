import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from config import model_config

parameters = []
for kfold in range(5):
    model_config.init(kfold)
    epoch = model_config.init_best_saved_epoch()
    data = torch.load(f"./saved_model/epoch{epoch}_val{kfold}.pth", weights_only=True)

    # print(data['interaction_strength_k'].shape)
    parameters.append(data['interaction_strength_k'].cpu().numpy())
parameters = np.stack(parameters, axis=0)
parameters = parameters[:, 1:] # 0-hop does not used
print(parameters.shape)
# np.save("hop_parameters.npy", parameters)

df = {
    'hop': list(range(1, 5)),
    'kfold0': parameters[0, :].tolist(),
    'kfold1': parameters[1, :].tolist(),
    'kfold2': parameters[2, :].tolist(),
    'kfold3': parameters[3, :].tolist(),
    'kfold4': parameters[4, :].tolist(),
}
df = pd.DataFrame(df)
df.to_csv("hop_parameters.csv", index=False)