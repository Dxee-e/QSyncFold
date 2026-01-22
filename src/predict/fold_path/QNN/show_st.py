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
        out = net(sequence).detach().numpy()
        result.append(out)
    result = np.stack(result, axis=0)
    return result



# Amyloid-beta precursor protein [111, 120] 10bp where 115 L>P & 116 A>D will cause Beta-thalassemia (B-THAL)
sequence = 'TPPPGTRVRA'
result = run_sequence(sequence)
print(result.shape)

result = np.mean(result, axis=0)

np.save('predict_structure.npy', result)

# draw 3D sphere 
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.scatter(result[:, 0], result[:, 1], result[:, 2], c='r', marker='o')
#for i in range(len(sequence)-1):
#    ax.plot(result[i:i+2, 0], result[i:i+2, 1], result[i:i+2, 2], c='b')
#plt.show()