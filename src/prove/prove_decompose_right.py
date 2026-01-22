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

    for row in range(N):
        for col in range(N):
            if row == col: continue
            theta = random_generator.random(random_times) * np.pi
            circuit_matrix = qml.matrix(circuit)(theta, row, col)

            target_matrix = np.eye(2**N, 2**N, dtype=np.complex128)[np.newaxis, :, :].repeat(random_times, axis=0)
            if row > col:
                theta *= -1
                row, col  = col, row
            v_cos = np.cos(theta/2)
            v_sin = np.sin(theta/2)
            target_matrix[:, row, row] = v_cos
            target_matrix[:, col, col] = v_cos
            target_matrix[:, row, col] = -v_sin
            target_matrix[:, col, row] = v_sin
            result = np.allclose(circuit_matrix, target_matrix)
            if not result:
                raise "Can not decompose it."

for i in tqdm(range(1, 8)):
    test(N=i, random_times=50)