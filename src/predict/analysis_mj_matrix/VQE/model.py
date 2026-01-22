"""# Model Define
VQE
loss function: CVaR
ansatz: Real Amplitudes Ansatz
optimizer: COBYLA """

import random
import os
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import pickle
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from qiskit_research.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from qiskit_research.protein_folding.penalty_parameters import PenaltyParameters
from qiskit_research.protein_folding.peptide.peptide import Peptide
from qiskit_research.protein_folding.protein_folding_problem import (
    ProteinFoldingProblem,
)

random.seed(55555)
np.random.seed(55555)
algorithm_globals.random_seed = 55555


def solve_problem(sequence: str):
    # random_interaction = RandomInteraction()
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_back = 10
    penalty_chiral = 10
    penalty_1 = 10
    penalty_terms = PenaltyParameters(penalty_chiral, penalty_back, penalty_1)
    main_chain = sequence
    main_chain = main_chain.replace("X", "G")
    peptide = Peptide(main_chain, [""] * len(main_chain))
    protein_folding_problem = ProteinFoldingProblem(
        peptide, mj_interaction, penalty_terms
    )
    qubit_op = protein_folding_problem.qubit_op()

# solve_problem('ANPHRLPT')
data_index = pd.read_csv('../../../../data/dataset_index.csv')
data_index = data_index[data_index['dataset_type']=='test']
# data_index = data_index[(data_index['length']>=5) & (data_index['length']<=9)]
data_index = data_index[data_index['length']==9]

results = {}
for _, row in tqdm(data_index.iterrows(), total=len(data_index)):
    pdb_id = row['pdb_id']
    # if pdb_id != '1yjp':
    #     continue
    sequence = ''.join(np.load(f'../../../../data/FormatData/{pdb_id}.npz')['sequence'])
    # print(pdb_id, sequence, len(sequence))
    solve_problem(sequence)
    
    records = pickle.load(open('temp.pkl', 'rb'))
    # print(records)
    os.remove('temp.pkl')
    
    records = sorted(records, key=lambda x: (x[0], x[1]))
    print(records)
    # for r in records:
    #     if r[2]==1 or r[3]==1:
    #         print(records)
            
    #         exit(0)
    # NOTICE: all r[2] and r[3] equal 0
    
    records = {(r[0], r[1]): r[4] for r in records}
    
    temp = np.zeros((len(sequence), len(sequence)))
    for (i,j), v in records.items():
        # print(i,j,len(sequence))
        temp[i-1,j-1] = v
    
    results[pdb_id] = temp
    
    while os.path.exists('temp.pkl'):
        time.sleep(0.5)

np.savez('interactions.npz', **results)