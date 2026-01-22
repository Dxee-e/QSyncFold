
import os

import numpy as np
from tqdm import tqdm
import time
import tracemalloc

from dataloader import DataLoader

from model import solve_problem

# length range are limited to [4, 9] due to device limitation
loader = DataLoader(dataset_type="test", min_len=5, max_len=9)




results = {
    'time': [],
    'qubits': [],
    'memory': [],
}
for pdb_id, length, sequence, coord in tqdm(loader):
    tracemalloc.start()
    start_time = time.time()    
    try:
        num_qubits, target = solve_problem(sequence, return_qubits=True)
        # num_qubits, target = solve_problem("QYSNQNV", return_qubits=True)
    finally:
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    results['time'].append(end_time - start_time)
    results['qubits'].append(num_qubits)
    results['memory'].append(peak / (1024**2)) # MB

import pandas as pd
results = pd.DataFrame(results)
results.to_csv('./time_memory.csv', index=False)
