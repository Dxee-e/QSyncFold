"""Model Define
"""
import math

import pennylane as qml
import torch
from torch import Tensor, nn, pi, tensor
from torch.nn import functional as F

from AnyRYGate import AnyRY
from config import ModelConfig


class Model(nn.Module):
    """define model"""

    def __init__(self, mc: ModelConfig):
        super().__init__()
        self.mc = mc
        self.define_parameters()

        self.qc_layer = self.create_qc(
            N=self.mc.dataset_max_length
        )
        
        self.unit_directions = torch.tensor([
            [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
        ], device=self.mc.device, dtype=self.mc.dtype_float)
        self.unit_directions = self.unit_directions / torch.sqrt(torch.sum(self.unit_directions**2, dim=-1, keepdim=True))

        self = torch.compile(self)
        self = self.to(self.mc.device)

    def define_parameters(self):
        # params
        self.param_interaction_k = nn.Parameter(
            torch.rand(
                (2, ),
                generator=self.mc.torch_generator,
                dtype=self.mc.dtype_float,
                device=self.mc.device,
            )
            * 2
            - 1,
            requires_grad=True,
        )

    def create_qc(self, N: int):
        wires_pos = list(range(2))
        wires_total = wires_pos
        dev = qml.device(self.mc.quantum_device, wires=wires_total)

        @qml.qnode(device=dev, interface="torch", diff_method=self.mc.diff_method)
        def circuit(
            interaction_k,
        ):
            # lattice
            qml.Hadamard(wires=wires_pos[0])
            qml.Hadamard(wires=wires_pos[1])
            
            for j in range(2):
                qml.RY(
                    interaction_k[j],
                    wires=wires_pos[j],
                )
                
            return [qml.expval(qml.PauliZ(wires=wires_pos[i])) for i in range(2)]

        circuit = qml.simplify(circuit)
        # circuit = qml.compile(circuit)
        circuit = torch.compiler.disable(circuit)
        return circuit

    @staticmethod
    def sphere_to_cartesian(arr: Tensor) -> Tensor:
        """sphere to cartesian"""
        rho, theta, phi = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        t = rho * torch.sin(phi)
        x = t * torch.cos(theta)
        y = t * torch.sin(theta)
        z = rho * torch.cos(phi)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def cartesian_to_sphere(arr: Tensor) -> Tensor:
        """cartesian to sphere"""
        x, y, z = arr[:, :, :, 0], arr[:, :, :, 1], arr[:, :, :, 2]
        rho = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
        theta = torch.atan2(y, x + 1e-16)
        # phi = torch.acos(z / rho)  # fix bug when acos(theta) theta==1 or -1
        phi = torch.acos(
            torch.clip(
                z / rho,
                min=-1 + 1e-7,
                max=1 - 1e-7,
            )
        )
        return torch.stack([rho, theta, phi], dim=-1)

    def generate_random_coords(self, N: int, rids: Tensor, batch: int):
        """generate random coords - relative coords"""
        sphere_coords = torch.zeros(
            (batch, N - 1, 3),
            dtype=self.mc.dtype_float,
            device=self.mc.device,
        )
        for i in range(1, N):
            mu, sigma = self.mc.get_distribution(
                rids[i - 1].item(), rids[i].item()
            )

            bone_length = torch.normal(
                mu,
                sigma,
                (batch,),
                generator=self.mc.torch_generator,
                device=self.mc.device,
            )
            bone_length = torch.clamp(
                bone_length,
                self.mc.bone_length_range[0],
                self.mc.bone_length_range[1],
            )

            angle = torch.rand(
                (batch, 2),
                dtype=self.mc.dtype_float,
                device=self.mc.device,
                generator=self.mc.torch_generator,
            )
            sphere_coords[:, i - 1, 0] = bone_length  # rho>=0
            sphere_coords[:, i - 1, 1] = angle[:, 0] * pi  # 0<=theta<=pi
            sphere_coords[:, i - 1, 2] = angle[:, 1] * pi * 2  # 0<=phi<=2*pi
        return sphere_coords

    @staticmethod
    def fix_xyz(x: Tensor) -> Tensor:
        dtype, device = x.dtype, x.device
        batch = x.shape[0]
        one_tensor = torch.ones((batch,), dtype=dtype, device=device)
        zero_tensor = torch.zeros((batch,), dtype=dtype, device=device)
        # first at [0,0,0]
        x = x - x[:, 0, :].unsqueeze(1)
        # second at [0,0,z]
        theta_z = torch.atan2(x[:, 1, 0], x[:, 1, 1] + 1e-8)
        theta_x = torch.atan2(torch.sqrt(x[:, 1, 0] ** 2 + x[:, 1, 1] ** 2 + 1e-8), x[:, 1, 2])
        cos_theta_z = torch.cos(theta_z)
        sin_theta_z = torch.sin(theta_z)
        rz = torch.stack(
            [
                torch.stack([cos_theta_z, -sin_theta_z, zero_tensor], dim=-1),
                torch.stack([sin_theta_z, cos_theta_z, zero_tensor], dim=-1),
                torch.stack([zero_tensor, zero_tensor, one_tensor], dim=-1),
            ],
            dim=1,
        )
        cos_theta_x = torch.cos(theta_x)
        sin_theta_x = torch.sin(theta_x)
        rx = torch.stack(
            [
                torch.stack([one_tensor, zero_tensor, zero_tensor], dim=-1),
                torch.stack([zero_tensor, cos_theta_x, -sin_theta_x], dim=-1),
                torch.stack([zero_tensor, sin_theta_x, cos_theta_x], dim=-1),
            ],
            dim=1,
        )
        rot_mat = torch.matmul(rx, rz)
        x = torch.matmul(rot_mat, x.transpose(1, 2)).transpose(1, 2)
        # third at [0, y, z]
        theta_z = torch.atan2(x[:, 2, 0], x[:, 2, 1])
        cos_theta_z = torch.cos(theta_z)
        sin_theta_z = torch.sin(theta_z)
        rz = torch.stack(
            [
                torch.stack([cos_theta_z, -sin_theta_z, zero_tensor], dim=-1),
                torch.stack([sin_theta_z, cos_theta_z, zero_tensor], dim=-1),
                torch.stack([zero_tensor, zero_tensor, one_tensor], dim=-1),
            ],
            dim=1,
        )
        x = torch.matmul(rz, x.transpose(1, 2)).transpose(1, 2)
        return x

    def forward(self, protein_sequence: str):
        """forward"""
        # preprocess
        batch = 1
        N = len(protein_sequence)
        zero_tensor = torch.zeros(
            (batch, 1, 3), dtype=self.mc.dtype_float, device=self.mc.device
        )
        # mask_without_diag = ~mask_with_diag

        # qc run
        result = torch.zeros(
            (batch, N - 1, 3),
            dtype=self.mc.dtype_float,
            device=self.mc.device,
        )
        for i in range(1, N):
            out = self.qc_layer(
                interaction_k=self.param_interaction_k,
            )
            out = torch.stack(out, dim=-1).unsqueeze(0)
            out = out.real
            out_dir = []
            for j in range(batch):
                if out[j, 0] < 0:
                    if out[j, 1] < 0:
                        out_dir.append(self.unit_directions[0, :])
                    else:
                        out_dir.append(self.unit_directions[1, :])
                else:
                    if out[j, 1] < 0:
                        out_dir.append(self.unit_directions[2, :])
                    else:
                        out_dir.append(self.unit_directions[3, :])
            out = torch.stack(out_dir, dim=0) * 3.8
            result[:, i - 1, :] = out
        result = torch.cumsum(torch.concat([zero_tensor, result], dim=1), dim=1)
        result = self.fix_xyz(result).squeeze(0)
        return result

# from config import model_config
# model_config.init(0)
# net = Model(model_config)
# out = net('LAASLACAL')
# print(out)