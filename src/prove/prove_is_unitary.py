from AnyRYGate import AnyRY
import pennylane as qml
import numpy as np
from tqdm import tqdm
random_generator = np.random.default_rng(seed=55555)

def test(N: int = 8, random_times:int = 50):
    dev = qml.device('lightning.qubit', wires=N)
    @qml.qnode(dev)
    def circuit(theta, row, col):
        AnyRY(theta, row, col, wires=list(range(N)))
        return qml.state()
    mat_identity = np.eye(2**N)

    for row in range(N):
        for col in range(N):
            for i in range(random_times):
                if row == col: continue
                theta = random_generator.random() * np.pi * 2
                mat = qml.matrix(circuit)(theta, row, col)
                conj_mat = np.conjugate(mat.T)
                f0 = np.allclose(mat @ conj_mat, mat_identity)
                f1 = np.allclose(conj_mat @ mat, mat_identity)
                if not f0 or not f1:
                    raise "Not a unitary matrix"

for N in tqdm(range(1, 5)):
    test(N=N, random_times=50)