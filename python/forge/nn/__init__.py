"""Forge neural network modules -- PyTorch drop-in replacements.

Drop-in replacement for torch.nn. Change your import from `torch.nn` to
`forge.nn` and get fused CUDA kernels automatically.

Auto-fusing: forge.nn.Sequential detects [Linear, Activation]^N + Linear
patterns and silently creates a fused MLP kernel. No code changes needed.
"""
# Core modules
from forge.nn.linear import Linear, QuantizedLinear, SparseLinear  # noqa: F401
from forge.nn.fused_mlp import FusedMLP  # noqa: F401
from forge.nn.sequential import Sequential  # noqa: F401

# Activations
from forge.nn.activation import (  # noqa: F401
    CELU, GELU, ReLU, LeakyReLU, SiLU, Sigmoid, Tanh,
)

# Loss functions
from forge.nn.mse import MSELoss  # noqa: F401
from forge.nn.l1 import L1Loss  # noqa: F401
from forge.nn.softmax_ce import SoftmaxCrossEntropy  # noqa: F401

# Convolution
from forge.nn.conv2d import Conv2d  # noqa: F401

# Normalization
from forge.nn.group_norm import GroupNorm  # noqa: F401

# Embedding
from forge.nn.embedding import Embedding  # noqa: F401
