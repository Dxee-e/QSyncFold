"""Model Define
"""
import math

import pennylane as qml
import torch
from torch import Tensor, nn, pi, tensor
from torch.nn import functional as F
import numpy as np

from AnyRYGate import AnyRY
from config import ModelConfig


class Model(nn.Module):
    """define model"""

    def __init__(self, mc: ModelConfig):
        super().__init__()
        self.mc = mc
        self.define_parameters()

        self.qc_layer, self.num_state_full_interaction = self.create_qc(
            N=self.mc.dataset_max_length
        )

        self = torch.compile(self)
        self = self.to(self.mc.device)

    def define_parameters(self):
        # params
        self.param_aa_interaction = nn.Parameter(
            torch.rand(
                (self.mc.num_amino_acids,),
                generator=self.mc.torch_generator,
                dtype=self.mc.dtype_float,
                device=self.mc.device,
            )
            * 2
            - 1,
            requires_grad=True,
        )
        self.param_interaction_sigma = nn.Parameter(
            tensor(1.0, dtype=self.mc.dtype_float, device=self.mc.device),
            requires_grad=True,
        )
        self.param_interaction_k = nn.Parameter(
            torch.rand(
                (self.mc.qc_k_num, 3),
                generator=self.mc.torch_generator,
                dtype=self.mc.dtype_float,
                device=self.mc.device,
            )
            * 2
            - 1,
            requires_grad=True,
        )
        self.interaction_strength_k = nn.Parameter(
            torch.rand(
                (self.mc.qc_k_num,),
                generator=self.mc.torch_generator,
                dtype=self.mc.dtype_float,
                device=self.mc.device,
            )
            * 2
            - 1,
            requires_grad=True,
        )

    def create_qc(self, N: int):
        num_wires_state_interaction = math.ceil(math.log2(N))
        num_state_full_interaction = 2**num_wires_state_interaction
        wires_interaction = list(range(num_wires_state_interaction))
        wires_total = wires_interaction

        dev = qml.device(self.mc.quantum_device, wires=wires_total)

        @qml.qnode(device=dev, interface="torch", diff_method=self.mc.diff_method)
        def circuit(
            total_num,
            cur_r_idx,
            interaction_strength,
            interaction_strength_k,
        ):
            # init interaction state
            AnyRY(torch.pi, 0, cur_r_idx, wires=wires_interaction)
            for i in range(total_num):
                if i != cur_r_idx:
                    k_idx = min(abs(i - cur_r_idx), self.mc.qc_k_num - 1)
                    AnyRY(
                        interaction_strength[i] * interaction_strength_k[k_idx],
                        cur_r_idx,
                        i,
                        wires=wires_interaction,
                    )
            return qml.state()

        circuit = qml.simplify(circuit)
        # circuit = qml.compile(circuit)
        circuit = torch.compiler.disable(circuit)
        return circuit, num_state_full_interaction

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

    @staticmethod
    def fix_xyz(x: Tensor) -> Tensor:
        dtype, device = x.dtype, x.device
        batch = x.shape[0]
        one_tensor = torch.ones((batch,), dtype=dtype, device=device)
        zero_tensor = torch.zeros((batch,), dtype=dtype, device=device)
        # first at [0,0,0]
        x = x - x[:, 0, :].unsqueeze(1)
        # second at [0,0,z]
        theta_z = torch.atan2(x[:, 1, 0], x[:, 1, 1])
        theta_x = torch.atan2(torch.sqrt(x[:, 1, 0] ** 2 + x[:, 1, 1] ** 2), x[:, 1, 2])
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

    def forward(self, pdb_id):
        """forward"""
        # preprocess
        max_N = self.mc.dataset_max_length

        reference = np.load(f'../../../../data/FormatData/{pdb_id}.npz')
        abs_coords = torch.tensor(reference['coord'], dtype=torch.float32, device=self.mc.device)
        protein_sequence = reference['sequence']
        
        N = len(protein_sequence)
        rids = tensor(
            self.mc.trans_aa_name2idx(protein_sequence),
            dtype=torch.long,
            device=self.mc.device,
        )
        mask_with_diag = torch.eye(N, N, dtype=torch.bool, device=self.mc.device)
        diff = abs_coords.unsqueeze(0) - abs_coords.unsqueeze(1)
        distance = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-18)
        distance[distance < 1e-8] = 1.0

        # qc init interaction state
        aa_interaction = self.param_aa_interaction.take(rids)
        print(self.param_aa_interaction)
        pair_interaction = aa_interaction.unsqueeze(0) * aa_interaction.unsqueeze(1)
        interaction_strength = (
            pair_interaction / distance * self.param_interaction_sigma
        )
        interaction_strength_min = torch.min(
            interaction_strength, dim=-1, keepdim=True
        ).values
        interaction_strength_max = torch.max(
            interaction_strength, dim=-1, keepdim=True
        ).values
        interaction_strength = (interaction_strength - interaction_strength_min) / (
            interaction_strength_max - interaction_strength_min
        )
        interaction_strength[mask_with_diag] = self.mc.init_self_strength
        interaction_strength = F.pad(
            interaction_strength,
            (0, self.num_state_full_interaction - N),
            mode="constant",
            value=0,
        )
        # interaction_strength = interaction_strength / torch.sqrt(torch.sum(interaction_strength**2, dim=-1, keepdim=True) + 1e-16)

        # qc run
        result = []
        for i in range(1, N):
            out = self.qc_layer(
                total_num=N,
                cur_r_idx=i,
                interaction_strength=interaction_strength[i, :],
                interaction_strength_k=self.interaction_strength_k,
            )
            out = out.real
            result.append(out)
        result = torch.stack(result, dim=0)
        result = result[:, :N]
        return result

from config import model_config
from tqdm import tqdm
result = {}
test_pdb_ids = np.load('./results.npz', allow_pickle=True).files
for pdb_id in tqdm(test_pdb_ids):
    result[pdb_id] = {}
    for k in range(model_config.all_kfold):
        model_config.__init__()
        model_config.init(k)
        net = Model(model_config)
        out = net(pdb_id).cpu().detach().numpy()
        temp = np.zeros((out.shape[0]+1, out.shape[1]))
        temp[1:, :] = out
        result[pdb_id][k] = temp
# np.savez('./interactions.npz', **result)

