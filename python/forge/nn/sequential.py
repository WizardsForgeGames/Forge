"""Auto-fusing Sequential container.

When constructed, inspects its children to detect fuseable MLP patterns.
If the module list matches [Linear, Activation]^N + Linear where all hidden
dims are equal and <= 256, it silently creates a FusedMLP internally.

The user writes standard code:
    model = forge.nn.Sequential(
        forge.nn.Linear(16, 64), forge.nn.CELU(),
        forge.nn.Linear(64, 64), forge.nn.CELU(),
        forge.nn.Linear(64, 64), forge.nn.CELU(),
        forge.nn.Linear(64, 4),
    )

And gets a single fused CUDA kernel. No changes to their code needed.
"""
import torch
import torch.nn as nn
from forge.nn.fused_mlp import FusedMLP
from forge.nn.linear import Linear


def _is_forge_activation(mod):
    """Check if a module is a Forge activation (has forge_activation_name)."""
    return hasattr(mod, 'forge_activation_name')


def _try_fuse(modules: list) -> FusedMLP | None:
    """Detect [Linear, Act, Linear, Act, ..., Linear] and create FusedMLP.

    Returns a FusedMLP if the pattern matches, None otherwise.

    Requirements for fusion:
    - Pattern: (Linear, Act)* Linear  (alternating Linear+Activation, ending with Linear)
    - All hidden Linear layers have the same in_features == out_features
    - Hidden width <= 256
    - All activations are the same type
    - At least one hidden layer (minimum: Linear, Act, Linear)
    """
    if len(modules) < 3:
        return None

    # Must end with a Linear
    if not isinstance(modules[-1], Linear):
        return None

    # Must have odd length (pairs of Linear+Act, plus final Linear)
    if len(modules) % 2 == 0:
        return None

    # Check alternating pattern: Linear, Act, Linear, Act, ..., Linear
    linears = []
    activations = []
    for i, mod in enumerate(modules):
        if i % 2 == 0:
            if not isinstance(mod, Linear):
                return None
            linears.append(mod)
        else:
            if not _is_forge_activation(mod):
                return None
            activations.append(mod)

    if len(linears) < 2 or len(activations) < 1:
        return None

    # Check all activations are the same type
    act_name = activations[0].forge_activation_name
    act_param = activations[0].forge_activation_param
    for act in activations[1:]:
        if act.forge_activation_name != act_name:
            return None
        if act.forge_activation_param != act_param:
            return None

    # Extract dimensions
    first_linear = linears[0]
    last_linear = linears[-1]
    n_inputs = first_linear.in_features
    n_outputs = last_linear.out_features

    # All hidden linears (index 0..N-1) must have the same out_features (hidden width)
    # and all middle linears (index 1..N-1) must have in_features == hidden
    hidden = first_linear.out_features
    if hidden > 256:
        return None

    for lin in linears[1:-1]:
        if lin.in_features != hidden or lin.out_features != hidden:
            return None

    # Last linear's in_features must match hidden
    if last_linear.in_features != hidden:
        return None

    n_layers = len(activations)  # number of hidden layers = number of activations

    # Check none of the linears have bias=False (fused MLP always has bias)
    for lin in linears:
        if lin.bias is None:
            return None

    # Auto-enable TF32 when hidden dimension is aligned for Tensor Cores
    use_tf32 = (hidden % 16 == 0)

    # Create FusedMLP with matching architecture
    fused = FusedMLP(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden=hidden,
        n_layers=n_layers,
        activation=act_name,
        act_param=act_param,
        output_scale=1.0,
        normalize=False,
        seed=0,  # placeholder -- we'll copy weights in
        use_tf32=use_tf32,
    )

    # Copy weights from individual Linear layers into the packed buffer
    _pack_weights_from_linears(fused, linears)

    return fused


def _pack_weights_from_linears(fused: FusedMLP, linears: list):
    """Copy weights from individual Linear layers into the FusedMLP packed buffer.

    Weight packing layout (contiguous float buffer):
      Layer 0: W0[hidden, n_inputs] + b0[hidden]
      Layer 1: W1[hidden, hidden]   + b1[hidden]
      ...
      Layer L: WL[hidden, hidden]   + bL[hidden]
      Output:  Wout[n_outputs, hidden] + bout[n_outputs]
    """
    with torch.no_grad():
        offset = 0
        buf = fused.weights.data

        for i, lin in enumerate(linears):
            w = lin.weight.data  # [out, in]
            b = lin.bias.data    # [out]
            n_w = w.numel()
            n_b = b.numel()

            buf[offset:offset + n_w].copy_(w.reshape(-1))
            offset += n_w
            buf[offset:offset + n_b].copy_(b.reshape(-1))
            offset += n_b


class Sequential(nn.Module):
    """Auto-fusing sequential container.

    If the module list matches a fuseable MLP pattern, creates an internal
    FusedMLP and routes forward/backward through the single-kernel path.
    Otherwise, falls back to running modules sequentially.

    Usage:
        # This auto-fuses into a single kernel:
        model = forge.nn.Sequential(
            forge.nn.Linear(16, 64), forge.nn.CELU(),
            forge.nn.Linear(64, 64), forge.nn.CELU(),
            forge.nn.Linear(64, 4),
        )

        # This falls back to sequential execution:
        model = forge.nn.Sequential(
            forge.nn.Linear(10, 20),
            forge.nn.Linear(20, 30),  # no activation between
        )
    """

    def __init__(self, *modules):
        super().__init__()
        self._modules_list = nn.ModuleList(modules)
        self._fused: FusedMLP | None = _try_fuse(list(modules))

        if self._fused is not None:
            # Register the fused MLP as a submodule for proper state_dict handling
            self._fused_mlp = self._fused
        else:
            self._fused_mlp = None

    @property
    def is_fused(self) -> bool:
        """True if the module list was auto-fused into a single kernel."""
        return self._fused is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused is not None:
            return self._fused(x)
        for mod in self._modules_list:
            x = mod(x)
        return x

    def extra_repr(self) -> str:
        if self._fused is not None:
            return f"(auto-fused: {self._fused.extra_repr()})"
        return ""
