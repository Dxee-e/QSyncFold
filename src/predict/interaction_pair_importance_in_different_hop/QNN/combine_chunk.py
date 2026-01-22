import os
import numpy as np

all_files = os.listdir('./results_chunk')
all_files = [f for f in all_files if f.endswith('.npz')]

for i, f in enumerate(all_files):
    data = np.load('./results_chunk/'+f, allow_pickle=True)['results'].item()
    if i == 0:
        results = data
    else:
        results = {**results, **data}

print(len(results))
np.savez('QNN_results.npz', results=results)