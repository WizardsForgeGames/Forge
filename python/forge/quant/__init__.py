"""Forge quantization toolkit.

Provides tools for converting PyTorch models to use Forge INT8 kernels:

    # Post-training quantization (one line):
    quantized_model = forge.quant.quantize(model, dataloader)

    # Quantization-aware training:
    qat_model = forge.quant.prepare_qat(model)
    train(qat_model, ...)  # train with fake quantization noise
    final_model = forge.quant.finalize_qat(qat_model)
"""
from forge.quant.convert import (  # noqa: F401
    quantize,
    prepare_qat,
    enable_calibration,
    enable_qat,
    finalize_qat,
)
from forge.quant.observers import MinMaxObserver, PercentileObserver  # noqa: F401
from forge.quant.fake_quantize import FakeQuantize  # noqa: F401
