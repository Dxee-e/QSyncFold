from torch import Tensor
import torch

def l2_loss(
        target: Tensor, source: Tensor
) -> Tensor:
    loss = torch.mean(torch.sum((target - source) ** 2, dim=-1))
    return loss

@torch.compile
def fix_xyz(x: Tensor) -> Tensor:
    dtype, device = x.dtype, x.device
    # first at [0,0,0]
    x = x - x[0, :]
    # second at [0,0,z]
    theta_z = torch.atan2(x[1, 0], x[1, 1])
    theta_x = torch.atan2(torch.sqrt(x[1,0]**2+x[1,1]**2), x[1, 2])
    rz = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0],
                             [torch.sin(theta_z), torch.cos(theta_z), 0],
                             [0, 0, 1]], dtype=dtype, device=device)
    rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(theta_x), -torch.sin(theta_x)],
                       [0, torch.sin(theta_x), torch.cos(theta_x)]], dtype=dtype, device=device)
    rot_mat = torch.matmul(rx, rz)
    x = torch.matmul(rot_mat, x.T).T
    # third at [0, y, z]
    theta_z = torch.atan2(x[2, 0], x[2, 1])
    rz = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0],
                          [torch.sin(theta_z), torch.cos(theta_z), 0],
                          [0, 0, 1]], dtype=dtype, device=device)
    x = torch.matmul(rz, x.T).T
    return x

class fix_xyz_wrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return fix_xyz(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def l2_loss_fix_xyz(target: Tensor, source: Tensor, fix_xyz_grad:bool = True)->Tensor:
    if fix_xyz_grad:
        target = fix_xyz(target)
        source = fix_xyz(source)
    else:
        target = fix_xyz_wrapper.apply(target)
        source = fix_xyz_wrapper.apply(source)
    return l2_loss(target, source)
