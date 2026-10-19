"""Forge linear layers using custom GEMM kernels.

FP32 Linear: Same signature as torch.nn.Linear. Uses Forge tiled GEMM.
QuantizedLinear: INT8 weight quantization with Tensor Core GEMM.
SparseLinear: 2:4 structured sparsity with INT8 Tensor Core GEMM.
"""
import math
import torch
import torch.nn as nn
from forge import _forge_C as _C


# ============================== FP32 Linear ==============================

class _LinearFn(torch.autograd.Function):
    """Forward: output = input @ weight^T + bias.  Uses Forge FP32 GEMM."""

    @staticmethod
    def forward(ctx, input, weight, bias):
        M = input.shape[0]
        K = input.shape[1]
        N = weight.shape[0]

        output = torch.empty(M, N, device=input.device, dtype=input.dtype)
        # C = 1.0 * input(M,K) @ weight^T(K,N) + 0.0 * C
        # input is (M,K) row-major, weight is (N,K) row-major
        # We need: output[M,N] = input[M,K] @ weight[N,K]^T
        # fp32_gemm: C = alpha * op(A) * op(B) + beta * C
        # A=input (N), B=weight (T): op(A)=input[M,K], op(B)=weight^T[K,N]
        _C.fp32_gemm(
            "N", "T", M, N, K, 1.0,
            input.contiguous(), K,
            weight.contiguous(), K,
            0.0, output, N,
            bias)

        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        M, K = input.shape
        N = weight.shape[0]

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_input[M,K] = grad_output[M,N] @ weight[N,K]
            grad_input = torch.empty(M, K, device=input.device, dtype=input.dtype)
            _C.fp32_gemm(
                "N", "N", M, K, N, 1.0,
                grad_output.contiguous(), N,
                weight.contiguous(), K,
                0.0, grad_input, K)

        if ctx.needs_input_grad[1]:
            # grad_weight[N,K] = grad_output^T[N,M] @ input[M,K]
            grad_weight = torch.empty(N, K, device=input.device, dtype=input.dtype)
            _C.fp32_gemm(
                "T", "N", N, K, M, 1.0,
                grad_output.contiguous(), N,
                input.contiguous(), K,
                0.0, grad_weight, K)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    """FP32 linear layer using Forge tiled GEMM.

    Same signature as torch.nn.Linear. Drop-in replacement.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming uniform initialization (matches PyTorch nn.Linear default)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _LinearFn.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ======================== Quantized / Sparse Linear ========================
# (kept from existing implementation for backward compatibility)

class _Int8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_int8, w_scale, bias, scratch_a, scratch_s):
        output = _C.int8_linear(x, w_int8, w_scale, bias, scratch_a, scratch_s)
        ctx.save_for_backward(x, w_int8, w_scale)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w_int8, w_scale = ctx.saved_tensors
        w_fp16 = w_int8.to(torch.float16) * w_scale.unsqueeze(1)
        grad_x = grad_output.to(torch.float16) @ w_fp16
        grad_w = grad_output.to(torch.float16).t() @ x.to(torch.float16)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.sum(dim=0).to(torch.float16)
        return grad_x, None, None, grad_bias, None, None


class QuantizedLinear(nn.Module):
    """INT8 quantized linear layer using Forge Tensor Core GEMM."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "w_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.w_scale = nn.Parameter(
            torch.ones(out_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)
        self._scratch_a = None
        self._scratch_s = None

    def _ensure_scratch(self, M, K, device):
        if (self._scratch_a is None or self._scratch_a.shape[0] < M
                or self._scratch_a.device != device):
            self._scratch_a = torch.empty(M, K, dtype=torch.int8, device=device)
            self._scratch_s = torch.empty(M, dtype=torch.float32, device=device)

    def forward(self, x):
        M, K = x.shape
        self._ensure_scratch(M, K, x.device)
        return _Int8LinearFn.apply(
            x, self.w_int8, self.w_scale, self.bias,
            self._scratch_a, self._scratch_s)

    @torch.no_grad()
    def quantize_weight(self, weight_fp):
        w = weight_fp.to(torch.float32)
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        self.w_int8.copy_(
            (w / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8))
        self.w_scale.data.copy_(scale.to(torch.float16))

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


class _SparseInt8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W_comp, w_meta, w_scale, bias,
                scratch_x_int8, scratch_x_scale):
        output = _C.sparse_int8_linear(
            x, W_comp, w_meta, w_scale, bias,
            scratch_x_int8, scratch_x_scale, None, 1)
        ctx.save_for_backward(x, W_comp, w_meta, w_scale)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "SparseLinear backward not yet implemented.")


class SparseLinear(nn.Module):
    """2:4 sparse INT8 linear layer using Forge sparse Tensor Core GEMM."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        assert in_features % 64 == 0
        assert out_features % 16 == 0
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "W_comp", torch.zeros(out_features, in_features // 2, dtype=torch.int8))
        self.register_buffer(
            "w_meta", torch.zeros(out_features, in_features // 32, dtype=torch.int32))
        self.w_scale = nn.Parameter(
            torch.ones(out_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)
        self._scratch_x_int8 = None
        self._scratch_x_scale = None

    def _ensure_scratch(self, M, K, device):
        if (self._scratch_x_int8 is None or self._scratch_x_int8.shape[0] < M
                or self._scratch_x_int8.device != device):
            self._scratch_x_int8 = torch.empty(M, K, dtype=torch.int8, device=device)
            self._scratch_x_scale = torch.empty(M, dtype=torch.float32, device=device)

    def forward(self, x):
        M, K = x.shape
        self._ensure_scratch(M, K, x.device)
        return _SparseInt8LinearFn.apply(
            x, self.W_comp, self.w_meta, self.w_scale, self.bias,
            self._scratch_x_int8, self._scratch_x_scale)

    @torch.no_grad()
    def from_dense_weight(self, weight_fp):
        w = weight_fp.to(torch.float32)
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        w_int8 = (w / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        N, K = w_int8.shape
        w_int8_dev = w_int8.cuda()
        comp = torch.zeros(N, K // 2, dtype=torch.int8, device="cuda")
        meta_raw = torch.zeros(N * K // 4, dtype=torch.uint8, device="cuda")
        meta_packed = torch.zeros(N, K // 32, dtype=torch.int32, device="cuda")
        _C.prune_compress_24(w_int8_dev, comp, meta_raw, N, K)
        _C.pack_metadata(meta_raw, meta_packed, N, K)
        torch.cuda.synchronize()
        self.W_comp.copy_(comp)
        self.w_meta.copy_(meta_packed)
        self.w_scale.data.copy_(scale.to(torch.float16))

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, sparse=2:4")
