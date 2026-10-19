"""Forge GroupNorm using custom CUDA kernels.

Same signature as torch.nn.GroupNorm. Drop-in replacement.
Uses Forge's warp-level reduction for mean/variance computation
with one block per (sample, group) pair.
"""
import torch
import torch.nn as nn
from forge import _forge_C as _C


class _GroupNormFn(torch.autograd.Function):
    """Forward/backward for standalone GroupNorm (no activation fusion)."""

    @staticmethod
    def forward(ctx, input, gamma, beta, num_groups, eps):
        N = input.shape[0]
        C = input.shape[1]
        spatial = 1
        for s in input.shape[2:]:
            spatial *= s

        y = torch.empty_like(input)
        mean = torch.empty(N, num_groups, device=input.device,
                           dtype=input.dtype)
        rstd = torch.empty(N, num_groups, device=input.device,
                           dtype=input.dtype)

        _C.group_norm_forward(
            input.contiguous(), gamma.contiguous(), beta.contiguous(),
            y, mean, rstd,
            N, C, spatial, num_groups, eps)

        ctx.save_for_backward(input, gamma, mean, rstd)
        ctx.num_groups = num_groups
        ctx.spatial = spatial
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, mean, rstd = ctx.saved_tensors
        N = input.shape[0]
        C = input.shape[1]

        grad_input = torch.empty_like(input)
        grad_gamma = torch.zeros(C, device=input.device, dtype=input.dtype)
        grad_beta = torch.zeros(C, device=input.device, dtype=input.dtype)

        _C.group_norm_backward(
            grad_output.contiguous(), input.contiguous(),
            gamma.contiguous(),
            mean, rstd,
            grad_input, grad_gamma, grad_beta,
            N, C, ctx.spatial, ctx.num_groups)

        return grad_input, grad_gamma, grad_beta, None, None


class GroupNorm(nn.Module):
    """Group Normalization using Forge CUDA kernels.

    Same signature as torch.nn.GroupNorm. Drop-in replacement.

    Args:
        num_groups: Number of groups to separate the channels into.
        num_channels: Number of channels expected in input.
        eps: Value added to the denominator for numerical stability.
        affine: If True, has learnable per-channel affine parameters.
    """

    def __init__(self, num_groups: int, num_channels: int,
                 eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible "
                f"by num_groups ({num_groups})")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.affine:
            gamma = torch.ones(self.num_channels, device=input.device,
                               dtype=input.dtype)
            beta = torch.zeros(self.num_channels, device=input.device,
                               dtype=input.dtype)
        else:
            gamma = self.weight
            beta = self.bias
        return _GroupNormFn.apply(
            input, gamma, beta, self.num_groups, self.eps)

    def extra_repr(self) -> str:
        return (f"num_groups={self.num_groups}, "
                f"num_channels={self.num_channels}, "
                f"eps={self.eps}, "
                f"affine={self.affine}")
