"""Forge GPU Compute Library."""
__version__ = "0.1.0"

import torch  # must be imported first to set up CUDA DLL paths on Windows
from forge import _forge_C as _C  # noqa: F401
from forge import nn  # noqa: F401
from forge import optim  # noqa: F401
from forge import quant  # noqa: F401
from forge import compile  # noqa: F401  # registers torch.compile backend
from forge import training  # noqa: F401
from forge.training import Trainer  # noqa: F401
