"""Forge Adam optimizer with on-device step scalars (CUDA Graph compatible).

All step computations happen on-device (no host-device sync), making this
compatible with CUDA Graphs. Built-in LR scheduling: linear warmup + cosine
annealing with optional SGDR warm restarts.

Usage:
    optimizer = forge.optim.Adam(model.parameters(), lr=1e-3,
                                  warmup_steps=100, total_steps=5000)
    for step in range(5000):
        loss = model(x).sum()
        loss.backward()
        optimizer.step(zero_grad=True)  # fused step + zero_grad
"""
import torch
from forge import _forge_C as _C


class Adam:
    """Adam optimizer using Forge kernels.

    All step computations happen on-device (no host-device sync), making this
    compatible with CUDA Graphs.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate. Default: 1e-3.
        betas: Coefficients for running averages. Default: (0.9, 0.999).
        eps: Term for numerical stability. Default: 1e-8.
        weight_decay: L2 penalty. Default: 0.
        warmup_steps: Linear warmup steps. Default: 0.
        total_steps: Total training steps (for cosine decay). Default: 1.
        min_lr_fraction: Minimum LR as fraction of base. Default: 0.01.
        max_grad_norm: Gradient clipping threshold. Default: 0 (disabled).
        num_restarts: SGDR cosine cycles (1 = no restarts). Default: 1.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        total_steps: int = 1,
        min_lr_fraction: float = 0.01,
        max_grad_norm: float = 0.0,
        num_restarts: int = 1,
    ):
        self.param_groups = [{"params": list(params)}]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_fraction = min_lr_fraction
        self.max_grad_norm = max_grad_norm
        self.num_restarts = num_restarts

        self._step_count = 0
        self._state: dict[torch.Tensor, dict] = {}

        # On-device scalars: [lr_scale, bias_correction1, bias_correction2]
        self._step_scalars: torch.Tensor | None = None
        # On-device scratch for gradient clipping
        self._norm_sq: torch.Tensor | None = None
        self._clip_scale: torch.Tensor | None = None

    def _lazy_init(self, device: torch.device):
        if self._step_scalars is None:
            self._step_scalars = torch.zeros(3, dtype=torch.float32, device=device)
            if self.max_grad_norm > 0:
                self._norm_sq = torch.zeros(1, dtype=torch.float32, device=device)
                self._clip_scale = torch.zeros(1, dtype=torch.float32, device=device)

    def _init_param_state(self, p: torch.Tensor) -> dict:
        state = {
            "m": torch.zeros_like(p),
            "v": torch.zeros_like(p),
        }
        self._state[p] = state
        return state

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, zero_grad: bool = False):
        """Full optimizer step: update LR schedule + Adam update.

        Args:
            zero_grad: If True, zeros gradients after consuming them.
                Eliminates the need for a separate optimizer.zero_grad() call.
                The write is essentially free (cache line already loaded).
        """
        self._step_count += 1
        all_params = [p for g in self.param_groups for p in g["params"]
                      if p.grad is not None]
        if not all_params:
            return

        device = all_params[0].device
        self._lazy_init(device)

        # Update step scalars on device (LR schedule + bias correction)
        _C.update_step_scalars(
            self._step_scalars, self._step_count,
            self.warmup_steps, self.total_steps, self.min_lr_fraction,
            self.beta1, self.beta2, self.num_restarts)

        # NaN/Inf guard: replace with 0 (skip bad updates gracefully)
        for p in all_params:
            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        # Gradient clipping (if enabled)
        grad_scale_ptr = None
        if self.max_grad_norm > 0:
            self._norm_sq.zero_()
            for p in all_params:
                _C.grad_norm_sq(p.grad, self._norm_sq)

            _C.compute_clip_scale(
                self._clip_scale, self._norm_sq,
                self.max_grad_norm, 1.0)
            grad_scale_ptr = self._clip_scale

        # Adam step for each parameter
        for p in all_params:
            if p not in self._state:
                self._init_param_state(p)
            state = self._state[p]

            _C.adam_step(
                p, p.grad, state["m"], state["v"],
                self.lr, self._step_scalars,
                self.beta1, self.beta2, self.eps, self.weight_decay,
                grad_scale_ptr, False, 0.0, 0.0, zero_grad)

            # Safety net: clean NaN from weights and momentum states
            p.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            state["m"].nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            state["v"].nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Trainer-internal methods (CUDA Graph compatible) ----

    def _update_scalars(self, device: torch.device | None = None):
        """Increment step and update step_scalars on device.

        Call this BEFORE graph replay. This is the non-graph-captured part
        of the optimizer step: it updates the device-side LR schedule and
        bias corrections that the graph-captured adam_step kernel reads from.
        """
        self._step_count += 1
        if device is not None:
            self._lazy_init(device)
        _C.update_step_scalars(
            self._step_scalars, self._step_count,
            self.warmup_steps, self.total_steps, self.min_lr_fraction,
            self.beta1, self.beta2, self.num_restarts)

    @torch.no_grad()
    def _apply_updates(self, zero_grad: bool = True):
        """Apply Adam parameter updates. Graph-compatible.

        Call this INSIDE graph capture. Reads step_scalars from device memory
        (written by _update_scalars before graph replay). Does NOT update
        step_count or step_scalars — those are handled by _update_scalars.

        Includes gradient clipping when max_grad_norm > 0. The _norm_sq and
        _clip_scale tensors have fixed device addresses (allocated once in
        _lazy_init), so the clipping kernels are graph-capturable.

        NaN guard: replaces NaN/Inf gradients with 0 before the Adam step.
        This prevents a single bad forward/backward pass from permanently
        corrupting model weights. The nan_to_num_ call is in-place and
        graph-compatible (fixed tensor addresses).
        """
        all_params = [p for g in self.param_groups for p in g["params"]
                      if p.grad is not None]

        # NaN/Inf guard: replace with 0 (skip bad updates gracefully)
        for p in all_params:
            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        grad_scale_ptr = None
        if self.max_grad_norm > 0 and self._norm_sq is not None:
            self._norm_sq.zero_()
            for p in all_params:
                _C.grad_norm_sq(p.grad, self._norm_sq)
            _C.compute_clip_scale(
                self._clip_scale, self._norm_sq,
                self.max_grad_norm, 1.0)
            grad_scale_ptr = self._clip_scale

        for p in all_params:
            if p not in self._state:
                self._init_param_state(p)
            state = self._state[p]

            _C.adam_step(
                p, p.grad, state["m"], state["v"],
                self.lr, self._step_scalars,
                self.beta1, self.beta2, self.eps, self.weight_decay,
                grad_scale_ptr, False, 0.0, 0.0, zero_grad)

            # Safety net: if adam_step produced NaN (from corrupted m/v),
            # replace with 0 to prevent permanent model corruption.
            # Also clean m/v to prevent one-way NaN poisoning of momentum.
            p.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            state["m"].nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            state["v"].nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    def state_dict(self) -> dict:
        return {
            "step_count": self._step_count,
            "param_states": {
                id(p): {k: v.cpu() for k, v in s.items()}
                for p, s in self._state.items()
            },
        }

    def load_state_dict(self, state_dict: dict):
        self._step_count = state_dict["step_count"]
