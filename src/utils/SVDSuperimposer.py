from typing import Callable

import numpy as np
import torch
from torch.linalg import det, svd
from torch import Tensor

# @torch.compile
def svd_superimposer(source: Tensor, target: Tensor) -> Tensor:
    """Superimposer by SVD"""
    assert source.shape == target.shape
    assert source.shape[1] == 3
    assert target.shape[1] == 3

    n = source.shape[0]

    # center on centroid
    cc_tar = sum(target) / n
    cc_src = sum(source) / n
    c_target = target - cc_tar
    c_source = source - cc_src
    # correlation matrix
    a = c_target.T @ c_source
    u, _, vt = svd(a, full_matrices=False)
    rot = (vt.T @ u.T).T
    rot = rot * det(rot)
    if det(rot) < 0:
        vt[2, :] *= -1
        rot = (vt.T @ u.T).T
    tran = cc_src - (cc_tar @ rot)
    # transform
    transformed_tar = (target @ rot) + tran
    return transformed_tar

class svd_superimposer_without_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, target):
        result = svd_superimposer(source, target)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output

# @torch.compile
def l2_loss_with_superimposer(
        target: Tensor, source: Tensor, svd_grad: bool = True
) -> Tensor:
    """Loss with superimposer"""
    if svd_grad:
        target = svd_superimposer(source, target)
    else:
        target = svd_superimposer_without_grad.apply(source, target)
    loss = torch.mean(torch.sum((target - source) ** 2, dim=-1))
    return loss
