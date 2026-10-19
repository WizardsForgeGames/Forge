"""Forge MSE loss using custom CUDA reduction kernel."""
import torch
import torch.nn as nn
from forge import _forge_C as _C


class _MSEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target):
        loss = torch.empty(1, device=pred.device)
        _C.mse_forward(pred.contiguous(), target.contiguous(), loss)
        ctx.save_for_backward(pred, target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        d_pred = torch.empty_like(pred)
        # Compute gradient with scale=1.0, then apply chain rule via multiply.
        # Using tensor multiply (not .item()) so this is CUDA Graph compatible.
        _C.mse_backward(pred, target, d_pred, 1.0)
        if grad_output.numel() == 1:
            d_pred = d_pred * grad_output
        return d_pred, None


class MSELoss(nn.Module):
    """Mean Squared Error loss using Forge parallel reduction kernel.

    Computes: loss = (1/n) * sum((pred - target)^2)

    Same interface as torch.nn.MSELoss(reduction='mean').
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _MSEFn.apply(pred.view(-1), target.view(-1))
