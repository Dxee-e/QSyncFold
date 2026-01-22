import math

import pennylane as qml
from pennylane.operation import AnyWires, Operation, Wires


class AnyRY(Operation):
    grad_method = "A"
    parameter_frequencies = [
        (1,),
    ]
    par_domain = "R"
    num_wires = AnyWires
    num_params = 1
    ndim_params = (0,)
    name = "AnyRY"

    def __init__(
        self, theta, source_state, target_state, wires=None, id=None, reverse_cos=False
    ):
        num_wires = len(wires)
        self.hyperparameters["num_wires"] = num_wires
        wires = Wires(wires)
        assert source_state != target_state
        assert source_state < 2**num_wires
        assert target_state < 2**num_wires

        if source_state > target_state:
            theta = -theta
            source_state, target_state = target_state, source_state
        self.hyperparameters["source_state"] = source_state
        self.hyperparameters["target_state"] = target_state
        if reverse_cos:
            theta = math.pi / 2 - theta
        super().__init__(theta, wires=wires, id=id)

    def decomposition(self):
        num_wires = self.hyperparameters["num_wires"]
        theta = self.parameters[0]
        if num_wires == 1:
            return [qml.RY(theta, wires=self.wires)]
        if num_wires >= 2:
            src = self.hyperparameters["source_state"]
            tar = self.hyperparameters["target_state"]
            src_bin = [i == "1" for i in bin(src)[2:].zfill(num_wires)]
            tar_bin = [i == "1" for i in bin(tar)[2:].zfill(num_wires)]
            diff_bin = []
            for i in range(num_wires):
                if src_bin[i] != tar_bin[i]:
                    diff_bin.append(i)

            ops = []
            change_bin = tar_bin
            all_wires = list(range(num_wires))
            for i in range(len(diff_bin) - 1):
                cur_bin_idx = diff_bin[i]
                control = all_wires[:cur_bin_idx] + all_wires[cur_bin_idx + 1 :]
                control_values = (
                    change_bin[:cur_bin_idx] + change_bin[cur_bin_idx + 1 :]
                )
                op = qml.ctrl(
                    qml.X(wires=cur_bin_idx),
                    control=control,
                    control_values=control_values,
                )
                ops.append(op)
                change_bin[cur_bin_idx] = not change_bin[cur_bin_idx]

            ry_wire = diff_bin[-1]
            control = all_wires[:ry_wire] + all_wires[ry_wire + 1 :]
            control_values = change_bin[:ry_wire] + change_bin[ry_wire + 1 :]
            if src_bin[ry_wire]:
                op = qml.ctrl(
                    qml.X(wires=ry_wire), control=control, control_values=control_values
                )
                ops.append(op)
            op = qml.ctrl(
                qml.RY(theta, wires=ry_wire),
                control=control,
                control_values=control_values,
            )

            ops = ops + [op] + ops[::-1]
            return ops


# # ----- test : qubit <= 1 -----
# @qml.qnode(qml.device('lightning.qubit', wires=1), interface='torch')
# def circuit(init_state, theta):
#     qml.StatePrep(init_state, wires=[0])
#     AnyRY(theta, 0, 1, wires=[0])
#     return qml.state()
# x = torch.rand((2, ))
# x = x / torch.sum(x**2).sqrt()
# print(x)
# print(circuit(x, torch.tensor(torch.pi)))

# # ----- test : qubit >= 2 -----
# num_wires = 10
# @qml.qnode(qml.device('lightning.qubit', wires=range(num_wires)), interface='torch')
# def circuit(init_state, theta):
#     qml.StatePrep(init_state, wires=range(num_wires))
#     AnyRY(theta, 543, 0, wires=range(num_wires))
#     return qml.state()
# x = torch.rand((2**num_wires, ))
# x = x / torch.sum(x**2).sqrt()
# out=circuit(x, torch.tensor(torch.pi)).real
# print(torch.sum(x!=out))

# # ----- test : qubit >= 2 -----
# num_wires = 8
# @qml.qnode(qml.device('lightning.qubit', wires=range(num_wires)), interface='torch', diff_method='adjoint')
# def circuit(init_state, theta):
#     qml.StatePrep(init_state, wires=range(num_wires))
#     AnyRY(theta, 0, 64, wires=range(num_wires))
#     return qml.expval(qml.PauliZ(wires=0))
# x = torch.rand((2**num_wires, ))
# x = x / torch.sum(x**2).sqrt()
# p = torch.nn.Parameter(torch.tensor(0.6, dtype=torch.float32))
# out=circuit(x, p).type(torch.float32)
# out.requires_grad_()
# out.backward()
# print(p.grad)
# print(torch.sum(torch.abs(x-out)<=1e-6))
