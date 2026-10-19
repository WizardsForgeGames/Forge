"""Forge Conv2d using custom im2col + GEMM kernels.

Same signature as torch.nn.Conv2d. Drop-in replacement.
Uses Forge's FP32 tiled GEMM via the im2col approach for
forward, backward-data, and backward-weight.
"""
import math
import torch
import torch.nn as nn
from forge import _forge_C as _C


class _Conv2dFn(torch.autograd.Function):
    """Forward/backward for Conv2D via im2col + GEMM."""

    @staticmethod
    def forward(ctx, input, weight, bias,
                padH, padW, strideH, strideW,
                dilH, dilW, groups):
        N, C_in, H, W = input.shape
        C_out, _, kH, kW = weight.shape

        H_out = (H + 2 * padH - dilH * (kH - 1) - 1) // strideH + 1
        W_out = (W + 2 * padW - dilW * (kW - 1) - 1) // strideW + 1

        output = torch.empty(N, C_out, H_out, W_out,
                             device=input.device, dtype=input.dtype)

        _C.conv2d_forward(
            input.contiguous(), weight.contiguous(),
            bias.contiguous() if bias is not None else bias,
            output,
            N, C_in, H, W, C_out, kH, kW,
            padH, padW, strideH, strideW,
            dilH, dilW, groups)

        ctx.save_for_backward(input, weight)
        ctx.conv_params = (N, C_in, H, W, C_out, kH, kW,
                           padH, padW, strideH, strideW,
                           dilH, dilW, groups)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        (N, C_in, H, W, C_out, kH, kW,
         padH, padW, strideH, strideW,
         dilH, dilW, groups) = ctx.conv_params

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.empty_like(input)
            _C.conv2d_backward_data(
                grad_output.contiguous(), weight.contiguous(),
                grad_input,
                N, C_in, H, W, C_out, kH, kW,
                padH, padW, strideH, strideW,
                dilH, dilW, groups)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.empty_like(weight)
            if ctx.has_bias:
                grad_bias = torch.empty(C_out, device=input.device,
                                        dtype=input.dtype)
            _C.conv2d_backward_weight(
                grad_output.contiguous(), input.contiguous(),
                grad_weight,
                grad_bias if grad_bias is not None else grad_bias,
                N, C_in, H, W, C_out, kH, kW,
                padH, padW, strideH, strideW,
                dilH, dilW, groups)

        # Return: one gradient per forward() argument
        return (grad_input, grad_weight, grad_bias,
                None, None, None, None, None, None, None)


class Conv2d(nn.Module):
    """2D convolution using Forge im2col + GEMM kernels.

    Same signature as torch.nn.Conv2d. Drop-in replacement.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel (int or tuple).
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of blocked connections. Default: 1.
        bias: If True, adds a learnable bias. Default: True.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple,
                 stride: int | tuple = 1,
                 padding: int | tuple = 0,
                 dilation: int | tuple = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups,
                        kernel_size[0], kernel_size[1]))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming uniform initialization (matches PyTorch nn.Conv2d)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = (self.in_channels // self.groups
                      * self.kernel_size[0] * self.kernel_size[1])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _Conv2dFn.apply(
            input, self.weight, self.bias,
            self.padding[0], self.padding[1],
            self.stride[0], self.stride[1],
            self.dilation[0], self.dilation[1],
            self.groups)

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding}, "
                f"dilation={self.dilation}, "
                f"groups={self.groups}, "
                f"bias={self.bias is not None}")
