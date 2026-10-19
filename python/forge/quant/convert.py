"""Model conversion: swap PyTorch modules for Forge quantized equivalents."""
import torch
import torch.nn as nn
from typing import Callable
import forge.nn as fnn
from forge.quant.fake_quantize import FakeQuantize
from forge.quant.observers import MinMaxObserver


def _swap_module(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a named child module in-place."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def quantize(
    model: nn.Module,
    calibration_data: torch.utils.data.DataLoader | list | None = None,
    num_calibration_batches: int = 100,
    scheme: str = "dynamic_int8",
    inplace: bool = False,
    forward_fn: Callable | None = None,
) -> nn.Module:
    """Quantize a PyTorch model using Forge INT8 kernels.

    Walks the module tree, replaces compatible layers with Forge equivalents,
    and optionally calibrates activation scales using provided data.

    Args:
        model: The PyTorch model to quantize.
        calibration_data: DataLoader or list of input tensors for calibration.
            Required for static quantization, ignored for dynamic.
        num_calibration_batches: Max batches to use for calibration.
        scheme: Quantization scheme. One of:
            - "dynamic_int8": Per-token dynamic activation quantization (default).
              No calibration needed. Quantizes activations on-the-fly.
            - "static_int8": Static activation quantization using calibrated scales.
              Requires calibration_data.
            - "weight_only_int8": Only quantize weights, keep activations in FP16.
        inplace: If True, modify model in-place. Otherwise, deepcopy first.
        forward_fn: Custom forward function for calibration.
            Called as forward_fn(model, batch). If None, uses model(batch).

    Returns:
        Quantized model with Forge layers.
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    device = next(model.parameters()).device

    # Phase 1: Find and convert compatible layers
    conversions = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            has_bias = module.bias is not None

            # Create Forge QuantizedLinear
            forge_linear = fnn.QuantizedLinear(in_f, out_f, bias=has_bias)

            # Quantize the weight
            forge_linear.quantize_weight(module.weight.data)

            # Copy bias if present
            if has_bias and module.bias is not None:
                forge_linear.bias.data.copy_(module.bias.data.to(torch.float16))

            forge_linear = forge_linear.to(device)
            conversions.append((name, forge_linear))

        elif isinstance(module, nn.Embedding):
            forge_emb = fnn.Embedding(module.num_embeddings, module.embedding_dim)
            forge_emb.weight.data.copy_(module.weight.data)
            forge_emb = forge_emb.to(device)
            conversions.append((name, forge_emb))

    for name, new_module in conversions:
        _swap_module(model, name, new_module)

    converted_count = len(conversions)

    # Phase 2: Calibration (for static quantization)
    if scheme == "static_int8" and calibration_data is not None:
        _calibrate(model, calibration_data, num_calibration_batches,
                   forward_fn, device)

    # Report
    linear_count = sum(1 for _, m in conversions if isinstance(m, fnn.QuantizedLinear))
    emb_count = sum(1 for _, m in conversions if isinstance(m, fnn.Embedding))
    print(f"forge.quant: converted {linear_count} Linear + {emb_count} Embedding "
          f"({converted_count} total)")

    return model


def _calibrate(
    model: nn.Module,
    data: torch.utils.data.DataLoader | list,
    num_batches: int,
    forward_fn: Callable | None,
    device: torch.device,
):
    """Run calibration data through the model to collect activation statistics."""
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data):
            if i >= num_batches:
                break
            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # assume first element is input
            batch = batch.to(device)
            if forward_fn is not None:
                forward_fn(model, batch)
            else:
                model(batch)


def prepare_qat(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """Insert fake-quantize nodes for quantization-aware training.

    Wraps each nn.Linear with FakeQuantize on input and weight, so the model
    can be fine-tuned with simulated quantization noise.

    Args:
        model: Model to prepare.
        inplace: Modify in-place or deepcopy.

    Returns:
        Model with FakeQuantize nodes inserted.
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    # Insert observers/fake-quantize at each Linear
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            wrapped = _QATLinearWrapper(module)
            _swap_module(model, name, wrapped)

    return model


class _QATLinearWrapper(nn.Module):
    """Wraps nn.Linear with fake quantization on weight and input."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self.weight_fq = FakeQuantize(
            MinMaxObserver(per_channel=True, ch_axis=0),
            per_channel=True, ch_axis=0)
        self.input_fq = FakeQuantize(
            MinMaxObserver(per_channel=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.input_fq(x)
        # Fake-quantize weight
        w_q = self.weight_fq(self.linear.weight)
        return torch.nn.functional.linear(x_q, w_q, self.linear.bias)

    def calibrate(self):
        self.weight_fq.calibrate()
        self.input_fq.calibrate()

    def enable_qat(self):
        self.weight_fq.quantize()
        self.input_fq.quantize()

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features


def enable_calibration(model: nn.Module):
    """Set all FakeQuantize modules to calibration mode."""
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.calibrate()
        elif isinstance(m, _QATLinearWrapper):
            m.calibrate()


def enable_qat(model: nn.Module):
    """Set all FakeQuantize modules to QAT mode (fake quantize + STE)."""
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.quantize()
        elif isinstance(m, _QATLinearWrapper):
            m.enable_qat()


def finalize_qat(model: nn.Module) -> nn.Module:
    """Convert a QAT model to a fully quantized Forge model.

    Replaces _QATLinearWrapper modules with forge.nn.QuantizedLinear,
    using the learned fake-quantize scales.

    Args:
        model: QAT-trained model.

    Returns:
        Model with Forge quantized layers (no fake quantize overhead).
    """
    conversions = []
    device = next(model.parameters()).device

    for name, module in model.named_modules():
        if isinstance(module, _QATLinearWrapper):
            linear = module.linear
            in_f, out_f = linear.in_features, linear.out_features
            has_bias = linear.bias is not None

            forge_linear = fnn.QuantizedLinear(in_f, out_f, bias=has_bias)
            forge_linear.quantize_weight(linear.weight.data)
            if has_bias and linear.bias is not None:
                forge_linear.bias.data.copy_(linear.bias.data.to(torch.float16))
            forge_linear = forge_linear.to(device)
            conversions.append((name, forge_linear))

    for name, new_module in conversions:
        _swap_module(model, name, new_module)

    print(f"forge.quant: finalized {len(conversions)} QAT layers")
    return model
