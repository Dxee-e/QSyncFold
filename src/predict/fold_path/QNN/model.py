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
        wires_pos = list(
            range(num_wires_state_interaction, num_wires_state_interaction + 3)
        )
        wires_total = wires_interaction + wires_pos

        dev = qml.device(self.mc.quantum_device, wires=wires_total)

        @qml.qnode(device=dev, interface="torch", diff_method=self.mc.diff_method)
        def circuit(
            total_num,
            cur_r_idx,
            interaction_strength,
            interaction_strength_k,
            interaction_direct,
            interaction_k,
            init_rho,
            init_theta,
            init_phi,
            work_interaction, # list[int]
        ):
            # init interaction state
            AnyRY(torch.pi, 0, cur_r_idx, wires=wires_interaction)
            for i in range(total_num):
                if i != cur_r_idx:
                    k_idx = min(abs(i - cur_r_idx), self.mc.qc_k_num - 1)
                    AnyRY(
                        interaction_strength[:, i] * interaction_strength_k[k_idx],
                        cur_r_idx,
                        i,
                        wires=wires_interaction,
                    )
            # qml.MottonenStatePreparation(interaction_strength, wires=wires_interaction)

            # init state - wires [rho, theta, phi]
            qml.Hadamard(wires=wires_pos[0])
            qml.Hadamard(wires=wires_pos[1])
            qml.Hadamard(wires=wires_pos[2])
            qml.RY(init_rho, wires=wires_pos[0])
            qml.RY(init_theta, wires=wires_pos[1])
            qml.RY(init_phi, wires=wires_pos[2])

            for i in range(total_num):
                if i not in work_interaction:
                    continue

                bool_control_value = [
                    bit == "1" for bit in bin(i)[2:].zfill(num_wires_state_interaction)
                ]
                k_idx = min(abs(i - cur_r_idx), self.mc.qc_k_num - 1)
                for j in range(3):
                    qml.ctrl(
                        qml.RY(
                            interaction_direct[:, i, j] * interaction_k[k_idx, j],
                            wires=wires_pos[j],
                        ),
                        control=wires_interaction,
                        control_values=bool_control_value,
                    )

            return [qml.expval(qml.PauliZ(wires=wires_pos[i])) for i in range(3)]

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

    def forward(self, protein_sequence: str, work_interaction):
        """forward"""
        # preprocess
        batch = self.mc.num_try_random
        N = len(protein_sequence)
        max_N = self.mc.dataset_max_length
        rids = tensor(
            self.mc.trans_aa_name2idx(protein_sequence),
            dtype=torch.long,
            device=self.mc.device,
        )
        mask_with_diag = torch.eye(N, N, dtype=torch.bool, device=self.mc.device)
        zero_tensor = torch.zeros(
            (batch, 1, 3), dtype=self.mc.dtype_float, device=self.mc.device
        )
        # mask_without_diag = ~mask_with_diag

        # random init coords
        rel_sphere_coords = self.generate_random_coords(N, rids, batch)
        rel_coords = self.sphere_to_cartesian(rel_sphere_coords)
        abs_coords = torch.cumsum(torch.concat([zero_tensor, rel_coords], dim=1), dim=0)

        # init qc pos
        init_rho = (
            torch.acos(
                (rel_sphere_coords[:, :, 0] - self.mc.bone_length_range[0])
                / self.mc.bone_length_interval
                * 2
                - 1
            )
            - pi / 2
        )
        init_theta = torch.acos(rel_sphere_coords[:, :, 1] / torch.pi * 2 - 1) - pi / 2
        init_phi = torch.acos(rel_sphere_coords[:, :, 2] / torch.pi - 1) - pi / 2

        diff = abs_coords.unsqueeze(1) - abs_coords.unsqueeze(2)
        distance = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-18)
        distance[distance < 1e-8] = 1.0
        direct = diff / distance.unsqueeze(-1)
        direct[:, 0, 0, :] = direct[:, 1, 0, :]
        for i in range(1, N - 1):
            direct[:, i, i, :] = direct[:, i - 1, i, :] + direct[:, i + 1, i, :]
        direct[:, N - 1, N - 1, :] = direct[:, N - 2, N - 1, :]
        direct = self.cartesian_to_sphere(direct)

        # qc init interaction state
        aa_interaction = self.param_aa_interaction.take(rids)
        pair_interaction = aa_interaction.unsqueeze(0) * aa_interaction.unsqueeze(1)
        interaction_strength = (
            pair_interaction.unsqueeze(0) / distance * self.param_interaction_sigma
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
        interaction_strength[:, mask_with_diag] = self.mc.init_self_strength
        interaction_strength = F.pad(
            interaction_strength,
            (0, self.num_state_full_interaction - N),
            mode="constant",
            value=0,
        )
        # interaction_strength = interaction_strength / torch.sqrt(torch.sum(interaction_strength**2, dim=-1, keepdim=True) + 1e-16)

        # qc run
        result = torch.zeros(
            (batch, N - 1, 3),
            dtype=self.mc.dtype_float,
            device=self.mc.device,
        )
        for i in range(1, N):
            out = self.qc_layer(
                total_num=N,
                cur_r_idx=i,
                interaction_strength=interaction_strength[:, i, :],
                interaction_strength_k=self.interaction_strength_k,
                interaction_direct=direct[:, :, i, :],
                interaction_k=self.param_interaction_k,
                init_rho=init_rho[:, i - 1],
                init_theta=init_theta[:, i - 1],
                init_phi=init_phi[:, i - 1],
                work_interaction=work_interaction,
            )
            out = torch.stack(out, dim=-1).type(self.mc.dtype_float)
            out[:, 0] = (
                (out[:, 0] + 1) / 2 * self.mc.bone_length_interval
                + self.mc.bone_length_range[0]
            )
            out[:, 1] = (out[:, 1] + 1) * torch.pi / 2
            out[:, 2] = (out[:, 2] + 1) * torch.pi
            out = self.sphere_to_cartesian(out.unsqueeze(0)).squeeze(0)
            result[:, i - 1, :] = out
        result = torch.cumsum(torch.concat([zero_tensor, result], dim=1), dim=1)
        result = self.fix_xyz(result)
        rel_sphere = self.cartesian_to_sphere(
            (result[:, 1:, :] - result[:, :-1, :]).unsqueeze(-2)
        ).squeeze(-2)
        # rel_sphere = torch.mean(rel_sphere, dim=0)
        rel_sphere = (
            torch.sort(rel_sphere, dim=0)
            .values[int(batch * 0.25) : int(batch * 0.75), :, :]
            .mean(dim=0)
        )
        rel_cart = self.sphere_to_cartesian(rel_sphere.unsqueeze(0)).squeeze(0)
        abs_coord = torch.cumsum(
            torch.concat(
                [
                    torch.zeros(
                        (1, 3),
                        dtype=self.mc.dtype_float,
                        device=self.mc.device,
                    ),
                    rel_cart,
                ],
                dim=0,
            ),
            dim=0,
        )
        abs_coord = self.fix_xyz(abs_coord.unsqueeze(0)).squeeze(0)
        return abs_coord

# from config import model_config
# model_config.init(0)
# net = Model(model_config)
# out = net('LAASLACAL', [1, 5])
# print(out)