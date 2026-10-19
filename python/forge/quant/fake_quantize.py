"""Fake quantization modules for quantization-aware training (QAT).

In QAT, the forward pass simulates quantization (round + clip to INT8 range
then immediately dequantize back to float), so the model learns to be robust
to quantization noise. The backward pass uses straight-through estimator (STE):
gradients flow as if the rounding never happened.
"""
import torch
import torch.nn as nn
from forge.quant.observers import MinMaxObserver


class FakeQuantize(nn.Module):
    """Fake quantize: x -> round(x / scale) * scale, clamped to [-127, 127].

    During calibration (enabled=False), just runs the observer.
    During QAT (enabled=True), applies fake quantization with STE backward.

    Args:
        observer: Observer module for tracking statistics. If None, uses MinMaxObserver.
        per_channel: Use per-channel quantization. Default: False.
        ch_axis: Channel axis for per-channel mode. Default: 0.
    """

    def __init__(self, observer: nn.Module | None = None,
                 per_channel: bool = False, ch_axis: int = 0):
        super().__init__()
        self.observer = observer or MinMaxObserver(per_channel, ch_axis)
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.enabled = False
        self.register_buffer("scale", torch.tensor(1.0))
        self._calibrated = False

    def calibrate(self):
        """Switch to calibration mode (observe only, no fake quantization)."""
        self.enabled = False

    def quantize(self):
        """Switch to QAT mode (fake quantize with STE)."""
        if not self._calibrated:
            self.update_scale()
        self.enabled = True

    def update_scale(self):
        """Recompute scale from observer statistics."""
        self.scale = self.observer.compute_scale()
        self._calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always observe (even during QAT, to track distribution drift)
        self.observer(x)

        if not self.enabled:
            return x

        # Reshape scale for broadcasting in per-channel mode.
        # e.g. weight (N, K) with ch_axis=0: scale (N,) -> (N, 1)
        scale = self.scale
        if self.per_channel and scale.ndim == 1 and x.ndim > 1:
            shape = [1] * x.ndim
            shape[self.ch_axis] = -1
            scale = scale.view(shape)

        return _FakeQuantizeFn.apply(x, scale)


class _FakeQuantizeFn(torch.autograd.Function):
    """STE fake quantize: forward rounds, backward passes gradients through."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Quantize then dequantize
        inv_scale = 1.0 / scale
        x_q = (x * inv_scale).round().clamp(-127, 127)
        x_dq = x_q * scale
        # Save mask for STE: gradient is 1 where x is in quantizable range,
        # 0 where it was clipped
        mask = (x.abs() <= scale * 127).float()
        ctx.save_for_backward(mask)
        return x_dq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mask, = ctx.saved_tensors
        # STE: pass gradients through where not clipped
        return grad_output * mask, None
