"""Fused MLP module: entire multi-layer perceptron in a single CUDA kernel.

Designed for small-to-medium hidden widths where framework overhead and memory
bandwidth dominate actual compute. A fused kernel eliminates all kernel launch
overhead and intermediate DRAM round-trips.

For H <= 128: weights cached in shared memory (fastest path).
For H > 128 (up to 256): weights read from L2 cache, activations still flow
through shared memory. Slightly lower throughput but still fully fused.

TF32 Tensor Core path (use_tf32=True, requires sm_80+ and hidden % 16 == 0):
  Uses wmma 16x16x8 TF32 fragments for hidden-layer matmuls.
  TF32 truncates FP32 inputs to 10-bit mantissa; accumulation is full FP32.
  Falls back to scalar path when dimensions don't align or GPU < sm_80.

Auto-dispatch: For large batch sizes (B > unfused_threshold), automatically
falls back to cuBLAS via torch.nn.functional.linear, which is faster for
large matmuls. The fused kernel excels at small batches where launch overhead
and memory bandwidth dominate; cuBLAS excels at large batches where its
optimized GEMM implementations win on raw throughput.

Usage:
    model = FusedMLP(n_inputs=9, n_outputs=2, hidden=64, n_layers=5,
                     activation='celu', output_scale=0.01,
                     normalize=True, lower_bounds=lb, upper_bounds=ub)
    output = model(input)  # single fused kernel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from forge import _forge_C as _C

_ACTIVATIONS = {
    'celu': lambda x, p: F.celu(x, alpha=p),
    'gelu': lambda x, _: F.gelu(x),
    'relu': lambda x, _: F.relu(x),
    'sigmoid': lambda x, _: torch.sigmoid(x),
    'silu': lambda x, _: F.silu(x),
    'tanh': lambda x, _: torch.tanh(x),
}


class _FusedMLPFn(torch.autograd.Function):
    """Custom autograd function wrapping Forge fused MLP forward/backward."""

    @staticmethod
    def forward(ctx, input, weights, n_inputs, n_outputs, hidden, n_layers,
                activation, act_param, output_scale, normalize, use_tf32):
        B = input.shape[0]
        output = torch.empty(B, n_outputs, device=input.device)

        # Allocate acts_save for backward
        n_acts = _C.mlp_acts_save_count(
            n_inputs, n_outputs, hidden, n_layers,
            activation, act_param, output_scale, normalize, B)
        acts_save = torch.empty(n_acts, device=input.device)

        _C.fused_mlp_forward(
            n_inputs, n_outputs, hidden, n_layers,
            activation, act_param, output_scale, normalize, use_tf32,
            input.contiguous(), weights, output, B, acts_save)

        ctx.save_for_backward(input, weights, acts_save)
        ctx.desc_args = (n_inputs, n_outputs, hidden, n_layers,
                         activation, act_param, output_scale, normalize,
                         use_tf32)
        ctx.B = B
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, acts_save = ctx.saved_tensors
        n_inputs, n_outputs, hidden, n_layers, \
            activation, act_param, output_scale, normalize, \
            use_tf32 = ctx.desc_args
        B = ctx.B

        n_w = weights.numel()
        d_weights = torch.zeros(n_w, device=input.device)

        # Check if we need input gradient
        d_input = torch.empty_like(input) if ctx.needs_input_grad[0] else None

        # Optionally use workspace for per-block gradients (no atomicAdd).
        # MUST be zero-initialized: when normalize=True, mlp_weight_count
        # includes normalization bounds, but the backward kernel only writes
        # gradients for actual weights/biases. Without zeroing, uninitialized
        # workspace positions for norm bounds produce garbage after reduction.
        ws_count = _C.fused_mlp_backward_workspace_count(
            n_inputs, n_outputs, hidden, n_layers,
            activation, act_param, output_scale, normalize, use_tf32, B)
        workspace = torch.zeros(ws_count, device=input.device) if ws_count > 0 else None

        _C.fused_mlp_backward(
            n_inputs, n_outputs, hidden, n_layers,
            activation, act_param, output_scale, normalize, use_tf32,
            input.contiguous(), weights, acts_save,
            grad_output.contiguous(), d_input, d_weights,
            B, 0.0, workspace)

        # Gradients: input, weights, then None for all the integer/string desc args
        # Note: when normalize=True, the weights tensor is the output of
        # torch.cat([trainable_weights, norm_bounds]). Autograd's cat backward
        # splits d_weights and only propagates the trainable portion -- norm
        # bound gradients are silently dropped (buffer, no requires_grad).
        return d_input, d_weights, None, None, None, None, None, None, None, None, None


class FusedMLP(nn.Module):
    """Fused multi-layer perceptron using Forge's single-kernel forward/backward.

    The entire network (all hidden layers + output layer) executes in one CUDA
    kernel per pass. Activations flow through shared memory -- no global memory
    between layers.

    Args:
        n_inputs: Input dimension.
        n_outputs: Output dimension.
        hidden: Hidden layer width (max 256; shared memory path for H<=128).
        n_layers: Number of hidden layers.
        activation: Activation function ('celu', 'gelu', 'relu', 'sigmoid',
                     'silu', 'tanh'). Default: 'celu'.
        act_param: Activation parameter (alpha for CELU). Default: 1.0.
        output_scale: Scale applied to final output. Default: 1.0.
        normalize: Apply min-max normalization to inputs. Default: False.
        lower_bounds: Host tensor [n_inputs] for normalization lower bounds.
        upper_bounds: Host tensor [n_inputs] for normalization upper bounds.
        seed: RNG seed for weight initialization. Default: 42.
        use_tf32: Use TF32 Tensor Cores when available (sm_80+, hidden % 16 == 0).
                  Falls back to scalar path when not supported. Default: False.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden: int = 64,
        n_layers: int = 3,
        activation: str = 'celu',
        act_param: float = 1.0,
        output_scale: float = 1.0,
        normalize: bool = False,
        lower_bounds: torch.Tensor | None = None,
        upper_bounds: torch.Tensor | None = None,
        seed: int = 42,
        use_tf32: bool = False,
        unfused_threshold: int = 8192,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden = hidden
        self.n_layers = n_layers
        self.activation = activation
        self.act_param = act_param
        self.output_scale = output_scale
        self.normalize = normalize
        self.use_tf32 = use_tf32
        self.unfused_threshold = unfused_threshold

        # Trainable weight count (weights + biases only, no norm bounds)
        self._n_w_train = _C.mlp_weight_count(
            n_inputs, n_outputs, hidden, n_layers,
            activation, act_param, output_scale, False)

        # Full count including normalization bounds (when normalize=True)
        self._n_w_full = _C.mlp_weight_count(
            n_inputs, n_outputs, hidden, n_layers,
            activation, act_param, output_scale, normalize)

        # Trainable parameters -- optimizer only sees these
        self.weights = nn.Parameter(torch.empty(self._n_w_train))

        # Normalization bounds as non-trainable buffer -- immune to weight
        # decay. Concatenated with weights in forward() via torch.cat.
        if normalize:
            self.register_buffer('norm_bounds',
                                 torch.empty(self._n_w_full - self._n_w_train))

        # Cached weight view offsets for unfused path (avoids per-call slicing)
        self._layer_offsets: list[tuple[int, int, int]] = []
        offset = 0
        for i in range(n_layers):
            in_dim = n_inputs if i == 0 else hidden
            w_start = offset
            offset += hidden * in_dim
            b_start = offset
            offset += hidden
            self._layer_offsets.append((w_start, hidden * in_dim, b_start))
        self._out_w_start = offset
        self._out_w_size = n_outputs * hidden
        self._out_b_start = offset + n_outputs * hidden

        # Initialize weights
        self.reset_parameters(seed, lower_bounds, upper_bounds)

    def reset_parameters(
        self,
        seed: int = 42,
        lower_bounds: torch.Tensor | None = None,
        upper_bounds: torch.Tensor | None = None,
    ):
        """Kaiming uniform initialization + optional normalization bounds."""
        # C++ init writes to a single contiguous CUDA buffer of n_w_full size
        tmp = torch.empty(self._n_w_full, device='cuda')
        _C.mlp_init_weights(
            self.n_inputs, self.n_outputs, self.hidden, self.n_layers,
            self.activation, self.act_param, self.output_scale,
            self.normalize, tmp, seed,
            lower_bounds, upper_bounds)

        # Split into trainable weights and frozen norm bounds
        device = self.weights.device
        src_w = tmp[:self._n_w_train]
        self.weights.data.copy_(src_w if device.type == 'cuda' else src_w.cpu())
        if self.normalize:
            src_b = tmp[self._n_w_train:]
            self.norm_bounds.copy_(src_b if device.type == 'cuda' else src_b.cpu())

    @torch.compiler.disable
    def _forward_fused(self, input: torch.Tensor) -> torch.Tensor:
        """Fused single-kernel path — optimal for small batch sizes."""
        if self.normalize:
            weights = torch.cat([self.weights, self.norm_bounds])
        else:
            weights = self.weights
        return _FusedMLPFn.apply(
            input, weights,
            self.n_inputs, self.n_outputs, self.hidden, self.n_layers,
            self.activation, self.act_param, self.output_scale,
            self.normalize, self.use_tf32)

    def _forward_unfused(self, input: torch.Tensor) -> torch.Tensor:
        """cuBLAS path via F.linear — optimal for large batch sizes.

        Uses pre-computed offsets to slice the contiguous Forge weight buffer
        into per-layer (W, b) views. Views share storage with self.weights
        so gradients flow back correctly.
        """
        act_fn = _ACTIVATIONS[self.activation]
        w = self.weights
        H = self.hidden

        # Normalization: 2 * (x - lb) / (ub - lb) - 1  (maps to [-1, 1])
        if self.normalize:
            lb = self.norm_bounds[:self.n_inputs]
            ub = self.norm_bounds[self.n_inputs:]
            x = 2.0 * (input - lb) / (ub - lb) - 1.0
        else:
            x = input

        # Hidden layers (pre-computed offsets)
        for w_start, w_size, b_start in self._layer_offsets:
            in_dim = w_size // H
            x = act_fn(F.linear(x, w[w_start:w_start + w_size].view(H, in_dim),
                                w[b_start:b_start + H]), self.act_param)

        # Output layer
        x = F.linear(x, w[self._out_w_start:self._out_w_start + self._out_w_size]
                      .view(self.n_outputs, H),
                      w[self._out_b_start:self._out_b_start + self.n_outputs])

        if self.output_scale != 1.0:
            x = x * self.output_scale

        return x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B = input.shape[0]
        if B >= self.unfused_threshold:
            return self._forward_unfused(input)
        return self._forward_fused(input)

    def extra_repr(self) -> str:
        tf32_str = ", tf32=True" if self.use_tf32 else ""
        return (f"{self.n_inputs}->{self.hidden}x{self.n_layers}->"
                f"{self.n_outputs}, act={self.activation}, "
                f"scale={self.output_scale}, norm={self.normalize}"
                f"{tf32_str}, unfused_threshold={self.unfused_threshold}")
