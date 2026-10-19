"""Forge GPU-native Trainer with CUDA Graph support.

Handles data upload, mini-batching, CUDA Graph capture, and training loop.
Combined with forge.nn and forge.optim, the whole training step is replayed
as a single CUDA Graph, removing per-step Python and kernel-launch overhead.

Usage (supervised):
    model = forge.nn.FusedMLP(16, 4, hidden=64, n_layers=5, activation='celu')
    optimizer = forge.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = forge.nn.MSELoss()

    trainer = forge.Trainer(model, optimizer, loss_fn, X_train, y_train,
                            val_data=X_val, val_targets=y_val,
                            epochs=5000, batch_size=400)
    history = trainer.fit()

Usage (custom step function):
    trainer = forge.Trainer(model, optimizer)
    history = trainer.fit_custom(
        step_fn=my_step_fn,       # (model) -> scalar loss
        iterations=50000,
        internal=5,
        pre_step=fill_rng,        # called before each graph replay
        post_iteration=walk_state, # called after each internal block
    )
"""
import time
import torch


class Trainer:
    """GPU-native training loop with CUDA Graph capture.

    Supports two modes:

    **Supervised mode** (via ``fit()``):
      Uploads data to GPU once, captures the training step as a CUDA Graph on
      the first batch, and replays the graph for all subsequent batches.

    **Custom step mode** (via ``fit_custom()``):
      User provides a ``step_fn(model) -> loss`` that defines the training
      step using pure tensor ops. The Trainer handles CUDA Graph capture
      around the step function, with optional pre_step (RNG fill) and
      post_iteration (state walk, parameter redraw) callbacks.

    Args:
        model: A forge.nn module (FusedMLP, Sequential, Linear, etc.).
        optimizer: A forge.optim.Adam optimizer.
        loss_fn: A loss module (for supervised mode). Optional.
        train_data: Training inputs [N, n_inputs]. Optional.
        train_targets: Training targets [N] or [N, n_outputs]. Optional.
        val_data: Optional validation inputs.
        val_targets: Optional validation targets.
        batch_size: Mini-batch size. Default: 256.
        epochs: Number of training epochs. Default: 1000.
        graph_capture: Enable CUDA Graph capture. Default: True.
        print_every: Print loss every N epochs. 0 = silent. Default: 100.
        shuffle: Shuffle training data each epoch. Default: True.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn=None,
        train_data: torch.Tensor | None = None,
        train_targets: torch.Tensor | None = None,
        val_data: torch.Tensor | None = None,
        val_targets: torch.Tensor | None = None,
        batch_size: int = 256,
        epochs: int = 1000,
        graph_capture: bool = True,
        print_every: int = 100,
        shuffle: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.graph_capture = graph_capture
        self.print_every = print_every
        self.shuffle = shuffle

        # Move data to GPU (no-op if already there, skip if not provided)
        if train_data is not None and train_targets is not None:
            self.train_x = train_data.cuda().float().contiguous()
            self.train_y = train_targets.cuda().float().contiguous()
            self.n_train = self.train_x.shape[0]
        else:
            self.train_x = self.train_y = None
            self.n_train = 0

        if val_data is not None and val_targets is not None:
            self.val_x = val_data.cuda().float().contiguous()
            self.val_y = val_targets.cuda().float().contiguous()
            self.n_val = self.val_x.shape[0]
        else:
            self.val_x = self.val_y = None
            self.n_val = 0

    def _validate(self) -> float:
        """Compute validation loss (mean over all samples)."""
        if self.val_x is None:
            return float('nan')

        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            n_samples = 0
            for start in range(0, self.n_val, self.batch_size):
                end = min(start + self.batch_size, self.n_val)
                bs = end - start
                pred = self.model(self.val_x[start:end])
                loss = self.loss_fn(pred, self.val_y[start:end])
                total_loss += loss.item() * bs
                n_samples += bs
            val_loss = total_loss / n_samples
        self.model.train()
        return val_loss

    def fit(self) -> dict:
        """Execute supervised training. Returns dict with loss history.

        Requires loss_fn, train_data, and train_targets to be set.

        Returns:
            dict with keys: 'train_loss', 'val_loss', 'epoch', 'time'
            Each is a list of values recorded at print_every intervals.
        """
        if self.train_x is None or self.loss_fn is None:
            raise ValueError(
                "fit() requires loss_fn, train_data, and train_targets. "
                "Use fit_custom(step_fn=...) for custom training steps.")

        batches_per_epoch = max(1, (self.n_train + self.batch_size - 1) // self.batch_size)
        total_steps = self.epochs * batches_per_epoch
        self.optimizer.total_steps = total_steps

        history = {"train_loss": [], "val_loss": [], "epoch": [], "time": []}

        if self.graph_capture:
            return self._fit_graph(history, total_steps)
        else:
            return self._fit_eager(history)

    def fit_custom(
        self,
        step_fn,
        iterations: int = 1000,
        internal: int = 1,
        pre_step=None,
        post_iteration=None,
        graph_capture: bool = True,
        print_every: int = 100,
    ) -> dict:
        """Execute training with a custom step function.

        The step function defines the full forward computation. The Trainer
        handles graph capture, optimizer updates, and the training loop.

        Args:
            step_fn: Callable ``(model) -> scalar_loss``. Must use pure tensor
                ops with fixed shapes (no data-dependent control flow, no
                host-device sync). RNG should be done in pre_step instead.
            iterations: Number of outer iterations.
            internal: Number of graph replays per outer iteration.
            pre_step: Optional callable ``() -> None``, called before each
                graph replay. Use for RNG fills (e.g. shocks.normal_()).
            post_iteration: Optional callable ``(int) -> None``, called after
                each internal block with the iteration index. Use for
                parameter redraws, state walks, etc.
            graph_capture: Enable CUDA Graph capture. Default: True.
            print_every: Print loss every N iterations. 0 = silent.

        Returns:
            dict with keys: 'loss', 'iteration', 'time'
        """
        total_steps = iterations * internal
        self.optimizer.total_steps = total_steps

        history = {"loss": [], "iteration": [], "time": []}

        if graph_capture:
            return self._fit_graph_custom(
                step_fn, iterations, internal, pre_step, post_iteration,
                print_every, history, total_steps)
        else:
            return self._fit_eager_custom(
                step_fn, iterations, internal, pre_step, post_iteration,
                print_every, history)

    def _fit_eager(self, history: dict) -> dict:
        """Standard training loop without CUDA Graph."""
        t0 = time.perf_counter()

        for epoch in range(self.epochs):
            if self.shuffle:
                perm = torch.randperm(self.n_train, device=self.train_x.device)
            else:
                perm = torch.arange(self.n_train, device=self.train_x.device)

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, self.n_train, self.batch_size):
                end = min(start + self.batch_size, self.n_train)
                idx = perm[start:end]
                x_batch = self.train_x[idx]
                y_batch = self.train_y[idx]

                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)
                loss.backward()
                self.optimizer.step(zero_grad=True)

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= n_batches

            if self.print_every > 0 and epoch % self.print_every == 0:
                val_loss = self._validate()
                elapsed = time.perf_counter() - t0
                history["train_loss"].append(epoch_loss)
                history["val_loss"].append(val_loss)
                history["epoch"].append(epoch)
                history["time"].append(elapsed)
                print(f"  epoch {epoch:5d} | train_mse {epoch_loss:.6e} "
                      f"| val_mse {val_loss:.6e} | {elapsed:.1f}s")

        return history

    def _fit_graph(self, history: dict, total_steps: int) -> dict:
        """Training loop with CUDA Graph capture."""
        t0 = time.perf_counter()
        device = torch.device('cuda')

        # ---- Static staging buffers (fixed device pointers for graph) ----
        n_in = self.train_x.shape[1]
        target_shape = (self.batch_size,) + self.train_y.shape[1:]
        x_static = torch.empty(self.batch_size, n_in, device=device)
        y_static = torch.empty(target_shape, device=device)

        # ---- Eagerly initialize optimizer state ----
        self.optimizer._lazy_init(device)
        for p in self.model.parameters():
            if p.requires_grad and p not in self.optimizer._state:
                self.optimizer._init_param_state(p)

        # ---- Save state before warmup ----
        weight_backup = {id(p): p.data.clone() for p in self.model.parameters()}
        opt_state_backup = {
            id(p): {'m': s['m'].clone(), 'v': s['v'].clone()}
            for p, s in self.optimizer._state.items()
        }
        step_backup = self.optimizer._step_count

        # ---- Warmup (settles CUDA memory allocator) ----
        # Set valid step_scalars for warmup iterations
        self.optimizer._step_count = 0
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                pred = self.model(x_static)
                loss = self.loss_fn(pred, y_static)
                loss.backward()
                self.optimizer.step(zero_grad=True)
        torch.cuda.current_stream().wait_stream(side)

        # ---- Restore state ----
        for p in self.model.parameters():
            p.data.copy_(weight_backup[id(p)])
        for p, state in self.optimizer._state.items():
            backup = opt_state_backup[id(p)]
            state['m'].copy_(backup['m'])
            state['v'].copy_(backup['v'])
        self.optimizer._step_count = step_backup

        # ---- Capture graph ----
        # The graph captures: forward -> loss -> backward -> adam_step(zero_grad)
        # step_scalars are read from device memory (updated before replay)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pred_static = self.model(x_static)
            loss_static = self.loss_fn(pred_static, y_static)
            loss_static.backward()
            self.optimizer._apply_updates(zero_grad=True)

        # ---- Training loop ----
        for epoch in range(self.epochs):
            if self.shuffle:
                perm = torch.randperm(self.n_train, device=device)
            else:
                perm = torch.arange(self.n_train, device=device)

            for start in range(0, self.n_train, self.batch_size):
                end = min(start + self.batch_size, self.n_train)
                bs = end - start

                if bs < self.batch_size:
                    # Graph requires fixed batch size — skip irregular last batch
                    continue

                # D2D copy into staging buffers
                idx = perm[start:end]
                x_static.copy_(self.train_x[idx])
                y_static.copy_(self.train_y[idx])

                # Update LR schedule + bias corrections on device (pre-graph)
                self.optimizer._update_scalars(device=device)

                # Replay the captured training step
                graph.replay()

            # ---- Periodic logging ----
            if self.print_every > 0 and epoch % self.print_every == 0:
                torch.cuda.synchronize()
                train_loss = loss_static.item()
                val_loss = self._validate()
                elapsed = time.perf_counter() - t0
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["epoch"].append(epoch)
                history["time"].append(elapsed)
                print(f"  epoch {epoch:5d} | train_mse {train_loss:.6e} "
                      f"| val_mse {val_loss:.6e} | {elapsed:.1f}s")

        return history

    def _fit_eager_custom(self, step_fn, iterations, internal, pre_step,
                          post_iteration, print_every, history):
        """Custom training loop without CUDA Graph."""
        t0 = time.perf_counter()

        for iteration in range(iterations):
            for o in range(internal):
                if pre_step is not None:
                    pre_step()
                self.optimizer.zero_grad()
                loss = step_fn(self.model)
                loss.backward()
                self.optimizer.step(zero_grad=True)

            if post_iteration is not None:
                post_iteration(iteration)

            if print_every > 0 and iteration % print_every == 0:
                torch.cuda.synchronize()
                loss_val = loss.item()
                elapsed = time.perf_counter() - t0
                history["loss"].append(loss_val)
                history["iteration"].append(iteration)
                history["time"].append(elapsed)
                print(f"  iter {iteration:5d} | loss {loss_val:.6e} | {elapsed:.1f}s")

        return history

    def _fit_graph_custom(self, step_fn, iterations, internal, pre_step,
                          post_iteration, print_every, history, total_steps):
        """Custom training loop with CUDA Graph capture."""
        t0 = time.perf_counter()
        device = torch.device('cuda')

        # ---- Eagerly initialize optimizer state ----
        self.optimizer._lazy_init(device)
        for p in self.model.parameters():
            if p.requires_grad and p not in self.optimizer._state:
                self.optimizer._init_param_state(p)

        # ---- Save state before warmup ----
        weight_backup = {id(p): p.data.clone() for p in self.model.parameters()}
        opt_state_backup = {
            id(p): {'m': s['m'].clone(), 'v': s['v'].clone()}
            for p, s in self.optimizer._state.items()
        }
        step_backup = self.optimizer._step_count

        # ---- Warmup (settles CUDA memory allocator) ----
        self.optimizer._step_count = 0
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                if pre_step is not None:
                    pre_step()
                loss = step_fn(self.model)
                loss.backward()
                self.optimizer.step(zero_grad=True)
        torch.cuda.current_stream().wait_stream(side)

        # ---- Restore state ----
        for p in self.model.parameters():
            p.data.copy_(weight_backup[id(p)])
        for p, state in self.optimizer._state.items():
            backup = opt_state_backup[id(p)]
            state['m'].copy_(backup['m'])
            state['v'].copy_(backup['v'])
        self.optimizer._step_count = step_backup

        # ---- Capture graph ----
        # The graph captures: step_fn -> backward -> grad_clip -> adam_step
        # step_scalars and RNG buffers are updated before replay.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss_static = step_fn(self.model)
            loss_static.backward()
            self.optimizer._apply_updates(zero_grad=True)

        # ---- Training loop ----
        for iteration in range(iterations):
            for o in range(internal):
                if pre_step is not None:
                    pre_step()
                self.optimizer._update_scalars(device=device)
                graph.replay()

            if post_iteration is not None:
                post_iteration(iteration)

            if print_every > 0 and iteration % print_every == 0:
                torch.cuda.synchronize()
                loss_val = loss_static.item()
                elapsed = time.perf_counter() - t0
                history["loss"].append(loss_val)
                history["iteration"].append(iteration)
                history["time"].append(elapsed)
                print(f"  iter {iteration:5d} | loss {loss_val:.6e} | {elapsed:.1f}s")

        return history
