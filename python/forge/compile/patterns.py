"""FX graph pattern matching and rewriting for Forge kernels.

Each pattern defines:
1. A match condition (which FX nodes to look for)
2. A replacement (what to insert instead)

Patterns are applied greedily in priority order: Linear+ReLU/LeakyReLU is
checked before the single-op Linear patterns.
"""
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from forge import _forge_C as _C


# ========================================================================
# Forge replacement functions (called from the rewritten FX graph)
# ========================================================================

def _forge_fp32_linear(x: torch.Tensor, weight: torch.Tensor,
                       bias: torch.Tensor | None) -> torch.Tensor:
    """FP32 Linear via Forge tiled GEMM."""
    M = x.shape[0]
    K = x.shape[1]
    N = weight.shape[0]
    C = torch.empty(M, N, dtype=torch.float32, device=x.device)
    # weight is (N, K), x is (M, K). We need x @ weight^T = (M, K) @ (K, N).
    # Forge fp32_gemm: C = alpha * op(A) * op(B) + beta * C
    # A = x (M, K), transA = N, lda = K
    # B = weight (N, K), transB = T, ldb = K  ->  op(B) = (K, N)
    # NOTE: Forge GEMM bias is per-row [M], but nn.Linear needs per-column [N].
    # Apply bias separately after the GEMM.
    _C.fp32_gemm("N", "T", M, N, K, 1.0,
                 x.contiguous(), K,
                 weight.contiguous(), K,
                 0.0, C, N,
                 None)
    if bias is not None:
        C += bias  # broadcasts (M, N) + (N,)
    return C


def _forge_fp32_linear_relu(x: torch.Tensor, weight: torch.Tensor,
                            bias: torch.Tensor | None,
                            negative_slope: float = 0.0) -> torch.Tensor:
    """Linear + ReLU/LeakyReLU as a Forge GEMM followed by an elementwise activation.

    This is two kernel launches (GEMM, then activation), not a single fused kernel.
    """
    out = _forge_fp32_linear(x, weight, bias)
    if negative_slope == 0.0:
        return F.relu(out)
    else:
        return F.leaky_relu(out, negative_slope)


# ========================================================================
# Pattern matching
# ========================================================================

def apply_forge_patterns(gm: fx.GraphModule) -> fx.GraphModule:
    """Walk the FX graph, find matching patterns, rewrite with Forge ops.

    Current patterns (in priority order):
    1. nn.Linear + nn.ReLU/LeakyReLU -> Forge GEMM + activation (two kernels)
    2. nn.Linear -> Forge fp32_gemm
    3. F.linear -> Forge fp32_gemm
    """
    graph = gm.graph
    modules = dict(gm.named_modules())
    replacements = 0

    # Pattern 1: Linear followed by ReLU/LeakyReLU
    for node in list(graph.nodes):
        if _is_relu_or_leaky(node, modules):
            linear_node = node.args[0] if isinstance(node.args[0], fx.Node) else None
            if linear_node and _is_linear(linear_node, modules):
                slope = _get_negative_slope(node, modules)
                linear_mod = modules[linear_node.target]
                with graph.inserting_before(node):
                    fused = graph.call_function(
                        _forge_fp32_linear_relu,
                        args=(linear_node.args[0],
                              linear_mod.weight,
                              linear_mod.bias,
                              slope))
                node.replace_all_uses_with(fused)
                graph.erase_node(node)
                if len(linear_node.users) == 0:
                    graph.erase_node(linear_node)
                replacements += 1

    # Pattern 2: Standalone nn.Linear -> Forge fp32_gemm
    for node in list(graph.nodes):
        if _is_linear(node, modules) and len(node.users) > 0:
            linear_mod = modules[node.target]
            with graph.inserting_before(node):
                forge_node = graph.call_function(
                    _forge_fp32_linear,
                    args=(node.args[0],
                          linear_mod.weight,
                          linear_mod.bias))
            node.replace_all_uses_with(forge_node)
            graph.erase_node(node)
            replacements += 1

    # Pattern 3: F.linear function calls
    for node in list(graph.nodes):
        if (node.op == "call_function" and node.target is F.linear
                and len(node.users) > 0):
            x, weight = node.args[0], node.args[1]
            bias = node.args[2] if len(node.args) > 2 else None
            if bias is None and "bias" in node.kwargs:
                bias = node.kwargs["bias"]
            with graph.inserting_before(node):
                forge_node = graph.call_function(
                    _forge_fp32_linear,
                    args=(x, weight, bias))
            node.replace_all_uses_with(forge_node)
            graph.erase_node(node)
            replacements += 1

    if replacements > 0:
        graph.lint()
        gm.recompile()

    return gm


# ========================================================================
# Helper predicates
# ========================================================================

def _is_linear(node: fx.Node, modules: dict) -> bool:
    """Check if node is a call to an nn.Linear module."""
    if node.op != "call_module":
        return False
    mod = modules.get(node.target)
    return isinstance(mod, nn.Linear)


def _is_relu_or_leaky(node: fx.Node, modules: dict) -> bool:
    """Check if node is ReLU, LeakyReLU, or F.relu/F.leaky_relu."""
    if node.op == "call_module":
        mod = modules.get(node.target)
        return isinstance(mod, (nn.ReLU, nn.LeakyReLU))
    if node.op == "call_function":
        return node.target in (F.relu, F.leaky_relu, torch.relu)
    return False


def _get_negative_slope(node: fx.Node, modules: dict) -> float:
    """Extract negative_slope from a ReLU/LeakyReLU node."""
    if node.op == "call_module":
        mod = modules.get(node.target)
        if isinstance(mod, nn.LeakyReLU):
            return mod.negative_slope
        return 0.0
    if node.op == "call_function":
        if node.target is F.leaky_relu:
            # F.leaky_relu(input, negative_slope=0.01)
            if len(node.args) > 1:
                return node.args[1]
            return node.kwargs.get("negative_slope", 0.01)
        return 0.0
    return 0.0
