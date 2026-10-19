"""Forge fused softmax + cross-entropy module."""
import torch
import torch.nn as nn
from forge import _forge_C as _C


class _SoftmaxCEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, V, B):
        probs, losses, correct = _C.softmax_ce_forward(logits, targets, V, B)
        ctx.save_for_backward(probs, targets)
        ctx.V = V
        ctx.B = B
        return probs, losses, correct

    @staticmethod
    def backward(ctx, grad_probs, grad_losses, grad_correct):
        probs, targets = ctx.saved_tensors
        d_logits = torch.empty_like(probs)
        _C.softmax_ce_backward(probs, targets, d_logits, ctx.V, ctx.B)
        # Scale by upstream loss gradient (usually 1/B for mean reduction)
        if grad_losses is not None:
            d_logits *= grad_losses.unsqueeze(0)
        return d_logits, None, None, None


class SoftmaxCrossEntropy(nn.Module):
    """Fused softmax + cross-entropy using Forge kernels.

    Takes [V, B] col-major logits and [B] targets.
    Returns (probs, losses, correct).

    Optionally uses the fully-fused GEMM+softmax+CE kernel that never
    materializes logits in global memory (set fused_gemm=True and pass
    W, X, bias instead of logits).
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: [V, B] col-major float32 logits.
            targets: [B] int32 target class indices.

        Returns:
            probs: [V, B] softmax probabilities.
            losses: [B] per-sample cross-entropy losses.
            correct: [B] int32 correct-prediction flags.
        """
        V, B = logits.shape
        return _SoftmaxCEFn.apply(logits.contiguous(), targets.contiguous(), V, B)
