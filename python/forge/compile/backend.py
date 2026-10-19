"""Forge backend for torch.compile.

Registers as a custom Dynamo backend that intercepts FX graphs, pattern-matches
nn.Linear / F.linear (and nn.Linear followed by ReLU/LeakyReLU), and replaces them
with calls to Forge's FP32 GEMM. Everything else in the graph runs eagerly as
ordinary PyTorch ops; this backend does not hand the rest of the graph to Inductor.

Inference-only for now: the Forge GEMM call has no autograd formula, so gradients
do not flow through rewritten Linear layers.
"""
import torch
import torch._dynamo
import torch.fx as fx
from typing import Callable
from forge.compile.patterns import apply_forge_patterns


def _forge_compiler(gm: fx.GraphModule, example_inputs: list[torch.Tensor]
                    ) -> Callable:
    """Forge compiler: pattern-match FX graph nodes to Forge kernels.

    This is the core of the backend. It receives an FX GraphModule from
    Dynamo's tracing, applies pattern rewrites, and returns a callable.
    """
    # Apply Forge-specific pattern rewrites
    gm = apply_forge_patterns(gm)

    # Recompile the modified graph
    gm.recompile()

    # Return the graph module itself as the callable
    # (the pattern replacements have already inserted Forge kernel calls)
    return gm


# Register with torch._dynamo
@torch._dynamo.register_backend
def forge(gm: fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
    """Forge backend for torch.compile.

    Usage:
        model = torch.compile(model, backend="forge")
    """
    return _forge_compiler(gm, example_inputs)


# Also export under a more explicit name
forge_backend = forge
