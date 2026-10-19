"""Forge activation layers with custom backward kernels.

When used standalone, these use Forge CUDA kernels for backward.
When inside forge.nn.Sequential, they're absorbed into the fused MLP kernel.

Each activation exposes `forge_activation_name` and `forge_activation_param`
so that Sequential can detect and fuse them.
"""
import torch
import torch.nn as nn
from forge import _forge_C as _C


# ---- Autograd functions for standalone backward kernels ----

class _LeakyReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, slope):
        output = torch.where(z > 0, z, slope * z)
        ctx.save_for_backward(z)
        ctx.slope = slope
        return output

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        dz = torch.empty_like(z)
        _C.leaky_relu_backward(z, grad_output.contiguous(), dz, ctx.slope)
        return dz, None


class _SiLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        s = torch.sigmoid(z)
        ctx.save_for_backward(z)
        return z * s

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        dz = torch.empty_like(z)
        _C.silu_backward(z, grad_output.contiguous(), dz)
        return dz


class _CELUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, alpha):
        output = torch.where(z > 0, z, alpha * (torch.exp(z / alpha) - 1))
        ctx.save_for_backward(z)
        ctx.alpha = alpha
        return output

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        dz = torch.empty_like(z)
        _C.celu_backward(z, grad_output.contiguous(), dz, ctx.alpha)
        return dz, None


# ---- Module classes ----

class CELU(nn.Module):
    """CELU activation with Forge backward kernel.

    celu(x) = max(0,x) + min(0, alpha*(exp(x/alpha)-1))
    """
    forge_activation_name = 'celu'

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    @property
    def forge_activation_param(self) -> float:
        return self.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _CELUFn.apply(x, self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class GELU(nn.Module):
    """GELU activation (tanh approximation).

    gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    """
    forge_activation_name = 'gelu'
    forge_activation_param = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate='tanh')


class ReLU(nn.Module):
    """ReLU activation."""
    forge_activation_name = 'relu'
    forge_activation_param = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class LeakyReLU(nn.Module):
    """Leaky ReLU with Forge backward kernel."""
    forge_activation_name = 'relu'  # fused MLP uses ReLU; standalone uses custom backward

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    @property
    def forge_activation_param(self) -> float:
        return self.negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LeakyReLUFn.apply(x, self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class SiLU(nn.Module):
    """SiLU (Swish) activation with Forge backward kernel.

    silu(x) = x * sigmoid(x)
    """
    forge_activation_name = 'silu'
    forge_activation_param = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _SiLUFn.apply(x)


class Sigmoid(nn.Module):
    """Sigmoid activation."""
    forge_activation_name = 'sigmoid'
    forge_activation_param = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class Tanh(nn.Module):
    """Tanh activation with Forge backward kernel."""
    forge_activation_name = 'tanh'
    forge_activation_param = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
