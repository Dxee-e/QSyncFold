import numpy as np
import pandas as pd


def main(seq):
    import model_org
    base_coords = model_org.main(seq) # [S, 3]
    diffs = np.diff(base_coords, axis=0)

    # 相对坐标，并归一化，只要方向
    def normalize_vectors(arr, eps: float = 1e-12):
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[-1] != 3:
            raise ValueError(f"expected vectors with last dim=3, got shape {a.shape}")
        norms = np.linalg.norm(a, axis=-1, keepdims=True)
        out = np.zeros_like(a)
        np.divide(a, norms, out=out, where=(norms > eps))
        return out

    base = normalize_vectors(diffs.reshape(-1, 3))[:, None, :]  # [N, 1, 3]

    # 这些都要归一化
    import model_rho
    rho = model_rho.main(seq)['direct']
    import model_theta
    theta = model_theta.main(seq)['direct']
    import model_phi
    phi = model_phi.main(seq)['direct']

    x = rho * np.cos(theta) * np.sin(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(phi)
    coords = np.stack([x, y, z], axis=-1)  # [N, S, 3]
    coords = normalize_vectors(coords.reshape(-1, 3)).reshape(coords.shape)

    # 角度方差
    angle_var = np.var(np.arccos(np.clip(np.sum(coords * base, axis=-1), -1.0, 1.0)))
    print(f"Angle Variance: {angle_var}")

    # 平均合向量长度 用coords-base
    mean_len = np.linalg.norm(np.mean(coords * base, axis=1))
    print(f"Mean Length of Mean Vector (coords-base): {mean_len}")

    return angle_var, mean_len

df = {
    'seq': [],
    'length': [],
    'pdb_id': [],
    'angle_var': [],
    'R': [],
}
data = pd.read_csv('../../../data/dataset_index.csv')
data = data[
    (data['dataset_type'] == 'test') &
    (data['length'] <= 9) &
    (data['length'] >= 5)
]
from tqdm import tqdm
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    print(row)
    pdb_id = row['pdb_id']
    protein = np.load(f'../../../data/FormatData/{pdb_id}.npz', allow_pickle=True)
    seq = protein['sequence']

    angle_var, R = main(seq)
    df['seq'].append(seq)
    df['length'].append(row['length'])
    df['pdb_id'].append(pdb_id)
    df['angle_var'].append(angle_var)
    df['R'].append(R)

result = pd.DataFrame(df)
result.to_csv('angle_var_results.csv', index=False)