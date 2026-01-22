from config import model_config
from tqdm import tqdm
from model import Model
import pandas as pd
import numpy as np
import torch

models = []
for i in range(model_config.all_kfold):
    model_config.__init__()
    model_config.init(i)
    models.append(Model(model_config))
    
# load data
data_index = pd.read_csv('../../../data/dataset_index.csv')
data_index = data_index[data_index['dataset_type']=='test']
# data_index = data_index[(data_index['length']>=5) & (data_index['length']<=9)]
data_index = data_index[data_index['length']==9]

saved_data = {}
for _, row in tqdm(data_index.iterrows(), total=len(data_index)):
    pdb_id = row['pdb_id']

    saved_data[pdb_id] = {}
    
    reference = np.load(f'../../../data/FormatData/{pdb_id}.npz')
    sequence = ''.join(reference['sequence'])
    reference_st = reference['coord']
    length = len(sequence)
    
    for k in tqdm(range(length+1), desc=pdb_id):
        temp = []
        for i in range(model_config.all_kfold):
            model_config.__init__()
            model_config.init(i)
            best_epoch = model_config.init_best_saved_epoch()
            net = Model(model_config)
            net.load_state_dict(torch.load(f'./saved_model/epoch{best_epoch}_val{i}.pth', weights_only=True))
            
            out = net(sequence, k).cpu().detach().numpy()
            temp.append(out)
        temp = np.stack(temp, axis=0)
        saved_data[pdb_id][k] = temp
        
np.savez('saved_data.npz', **saved_data)
