"""Forge L1 (MAE) loss using custom CUDA reduction kernel."""
import torch
import torch.nn as nn
from forge import _forge_C as _C


class _L1Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target):
        loss = torch.empty(1, device=pred.device)
        _C.l1_forward(pred.contiguous(), target.contiguous(), loss)
        ctx.save_for_backward(pred, target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        d_pred = torch.empty_like(pred)
        _C.l1_backward(pred, target, d_pred, 1.0)
        if grad_output.numel() == 1:
            d_pred = d_pred * grad_output
        return d_pred, None


class L1Loss(nn.Module):
    """Mean Absolute Error loss using Forge parallel reduction kernel.

    Computes: loss = (1/n) * sum(|pred - target|)

    Same interface as torch.nn.L1Loss(reduction='mean').
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _L1Fn.apply(pred.view(-1), target.view(-1))
