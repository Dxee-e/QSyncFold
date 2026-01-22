import numpy as np
import pennylane as qml
from AnyRYGateCopy import AnyRY

dev = qml.device('lightning.qubit', wires=2)
@qml.qnode(dev)
def circuit():
    # qml.CNOT(wires=[0, 1])
    qml.ctrl(qml.RY, control=[0], control_values=[False])(0.60716311, wires=1)
    # AnyRY(0.60716311, 0, 1, wires=[0, 1])
    return qml.state()

print(qml.matrix(circuit)())