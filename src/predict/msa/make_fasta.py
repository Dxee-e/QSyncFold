import pandas as pd
import numpy as np

data = pd.read_csv('../../../data/dataset_index.csv')
# data = data[
#     (data['dataset_type'] == 'train') &
#     (data['length'] <= 9) &
#     (data['length'] >= 5)
# ]
data = data[
    (data['dataset_type'] == 'test') &
    (data['length'] <= 9) &
    (data['length'] >= 5)
]

results = {
    'pdb_id': [],
    'seq': [],
}

for _, row in data.iterrows():
    pdb_id = row['pdb_id']
    protein = np.load(f'../../../data/FormatData/{pdb_id}.npz', allow_pickle=True)
    seq = protein['sequence']
    results['pdb_id'].append(pdb_id)
    results['seq'].append(''.join(seq))


# make fasta
with open('inputs.fasta', 'w') as f:
    for pdb_id, seq in zip(results['pdb_id'], results['seq']):
        f.write(f'>{pdb_id}\n')
        f.write(f'{seq}\n')
print("FASTA file 'inputs.fasta' created.")