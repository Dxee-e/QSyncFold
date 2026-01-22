protein_id = 'P13569'
file_name = f'AF-{protein_id}-F1-model_v4.pdb'
start_pos = 108
end_pos = 118
sequence = 'SYDPDNKEER'
mutate_pos = 117
mutate_amino_acid = 'P'

import numpy as np
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def fix_coords_by_xyz(x: np.ndarray) -> np.ndarray:
    # first at [0,0,0]
    x = x - x[0, :]
    # second at [0,0,z]
    theta_z = np.arctan2(x[1, 0], x[1, 1])
    theta_x = np.arctan2(np.sqrt(x[1, 0] ** 2 + x[1, 1] ** 2), x[1, 2])
    rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_mat = np.matmul(rx, rz)
    x = np.matmul(rot_mat, x.T).T
    # third at [0, y, z]
    theta_z = np.arctan2(x[2, 0], x[2, 1])
    rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    x = np.matmul(rz, x.T).T
    return x

def fix_coords_by_svd(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    n = reference.shape[0]
    # center on centroid
    cc_tar = sum(target) / n
    cc_src = sum(reference) / n
    c_target = target - cc_tar
    c_source = reference - cc_src
    # correlation matrix
    a = c_target.T @ c_source
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    rot = (vt.T @ u.T).T
    rot = rot * np.linalg.det(rot)
    if np.linalg.det(rot) < 0:
        vt[2, :] *= -1
        rot = (vt.T @ u.T).T
    tran = cc_src - (cc_tar @ rot)
    # transform
    transformed_tar = (target @ rot) + tran
    return transformed_tar


def rmsd(x: np.ndarray, y: np.ndarray) -> float:
    """Root Mean Square Deviation"""
    assert x.shape == y.shape
    n = x.shape[0]
    return np.sqrt(np.sum(np.sum((x - y) ** 2)) / n)


# pred_st = np.load('predict_structure.npy')
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
        out = net(sequence, list(range(len(sequence)))).detach().numpy()
        result.append(out)
    result = np.stack(result, axis=0)
    return result
result = run_sequence(sequence)
print(result.shape)
pred_st = np.mean(result, axis=0)
#print(pred_st.shape)

# ref_st = np.load('ref_coord.npy')
from Bio.PDB import PDBParser
import numpy as np

# read pdb file and extract alpha carbon coordinates

pdb_parser = PDBParser()

st = pdb_parser.get_structure('st', file_name)
print(len(st))
model = st[0]
for chain in model:
    coord = []
    residues = chain.get_residues()
    for residue in residues:
        for atom in residue:
            if atom.get_name() == 'CA':
                coord.append(atom.coord)
    coord = np.stack(coord, axis=0)
print(coord.shape)
ref_st = coord
ref_st = ref_st[start_pos:end_pos, :]
# ref_st = fix_coords_by_xyz(ref_st)
ref_st = fix_coords_by_svd(pred_st, ref_st)
#print(ref_st.shape)


print('rmsd - ref - pred base', rmsd(ref_st, pred_st))

mutate_sequence = list(sequence)
mutate_sequence[mutate_pos - start_pos] = mutate_amino_acid
mutate_sequence = ''.join(mutate_sequence)
result = run_sequence(mutate_sequence)
mu_pred_st = np.mean(result, axis=0)
print('rmsd - pred mutate - base', rmsd(mu_pred_st, pred_st))
print('rmsd - ref - pred mutate', rmsd(ref_st, mu_pred_st))

import matplotlib.pyplot as plt

# draw 3d sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(ref_st[:, 0], ref_st[:, 1], ref_st[:, 2], c='r', marker='o')
ax.scatter(pred_st[:, 0], pred_st[:, 1], pred_st[:, 2], c='b', marker='x')
for i in range(len(ref_st)-1):
    ax.plot(ref_st[i:i+2, 0], ref_st[i:i+2, 1], ref_st[i:i+2, 2], c='r')
    ax.plot(pred_st[i:i+2, 0], pred_st[i:i+2, 1], pred_st[i:i+2, 2], c='b')
plt.show()