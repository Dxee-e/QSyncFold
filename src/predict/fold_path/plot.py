from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = np.load('./QNN/QNN_results.npz', allow_pickle=True)['results'].item()

N = 10
plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = -14  # pad is in points...
target_color = '#FF9933'
base_color = '#6699CC'

# draw 3d sphere 
fig = plt.figure(figsize=(10, 10))

for i in range(1, N):
    target = data[tuple(list(range(i)))].mean(axis=0)
    base = data[tuple(list(range(N)))].mean(axis=0)

    ax = fig.add_subplot(3, 3, i, projection='3d')
    ax.set_title(f'After {i} time steps:', loc='left', pad=-14)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=target_color, label='target')
    ax.scatter(base[:, 0], base[:, 1], base[:, 2], c=base_color, label='base')

    for j in range(N-1):
        ax.plot([target[j, 0], target[j+1, 0]], [target[j, 1], target[j+1, 1]], [target[j, 2], target[j+1, 2]], c=target_color)
        ax.plot([base[j, 0], base[j+1, 0]], [base[j, 1], base[j+1, 1]], [base[j, 2], base[j+1, 2]], c=base_color)

    if i==1:
        ax.legend()    
fig.tight_layout(pad=1.5)
fig.savefig('QNN_results.svg', format='svg', dpi=1200)
# plt.show()
