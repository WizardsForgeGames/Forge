# Forge Python Training Utilities

Python-side training tools that complement Forge's C++/CUDA kernels. Pure Python (no external dependencies beyond PyTorch for Trainer), designed to work with any training loop.

| Module | Purpose | Dependencies |
|--------|---------|-------------|
| `forge.training.Trainer` | GPU-native training loop with CUDA Graph capture | PyTorch |
| `forge.training.ConvergencePredictor` | Predicts where training is heading, recommends LR schedules | None (pure Python + math) |
| `forge.training.cosine_S` | Compute total LR exposure for cosine schedules | None |
| `forge.training.predict_from` | Standalone convergence prediction (no predictor instance) | None |

---

## ConvergencePredictor

Predicts training convergence using the **harmonic decay model** and recommends optimal learning rate schedules. Model-agnostic — works with any training loop that has a learning rate and a decreasing validation metric.

### The Model

Training loss/MAE follows a harmonic decay when plotted against cumulative LR exposure:

```
dv/dS = -k · (v - B)²

=> v(S) = B + (v₀ - B) / (1 + k · (v₀ - B) · S)
```

Where:
- **v** = running-best validation metric (loss, MAE, RMSE, etc.)
- **S** = cumulative LR exposure: `∫ lr(t) dt` over training
- **B** = basin floor (the local minimum this trajectory converges toward)
- **k** = learning efficiency (how fast the model converts LR budget into improvement)

**Why S instead of epochs?** LR exposure `S` is the schedule-agnostic "learning budget" currency. Two runs with different LR schedules but the same total S produce comparable improvement. This lets the predictor compare across schedules.

**Why harmonic, not exponential?** The harmonic model `1/(1 + kS)` consistently fits real training trajectories better than exponential `exp(-kS)`. The key difference: harmonic decay has a long tail — improvement slows but never truly stops, which matches how gradient descent behaves near local minima.

### Quick Start

```python
from forge.training import ConvergencePredictor

predictor = ConvergencePredictor()

for epoch in range(total_epochs):
    lr = scheduler.get_lr()
    train(...)
    val_metric = validate(...)

    predictor.observe(epoch, lr, val_metric)

    if predictor.is_fitted:
        remaining = total_epochs - epoch
        predicted = predictor.predict(remaining)
        print(f"Predicted final: {predicted:.4f}")

        if predictor.should_stop(remaining):
            print("Diminishing returns — stopping early")
            break

# Schedule recommendations for future runs
print(predictor.format_report(epochs_remaining=600))
```

### How It Works

1. **Accumulates S** via trapezoidal integration of the learning rate over epochs
2. **Tracks running-best** validation metric (handles noisy trajectories)
3. **Fits k and B** by grid-searching basin floor candidates and computing trajectory MSE
4. **Predicts** where the current trajectory will end up given remaining LR budget
5. **Recommends** which peak LR will produce the best final metric, accounting for cross-LR basin depth tradeoffs

### API Reference

#### `ConvergencePredictor(k_prior, B_prior, min_observations, window, peak_lr)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_prior` | 50.0 | Prior for learning efficiency. Deep basins ~50-65, shallow ~7-15 |
| `B_prior` | 0.0 | Prior for basin floor. 0 = no prior knowledge (conservative) |
| `min_observations` | 5 | Validation observations required before making predictions |
| `window` | 20 | Recent observations used for fitting. Larger = more stable, slower to adapt |
| `peak_lr` | None | Peak LR of the current schedule. Required for cross-LR predictions |

#### `observe(epoch, lr, metric=None)`

Record a training observation. Call every epoch — for non-validation epochs, pass `metric=None` to still track S.

```python
# Every epoch
predictor.observe(epoch, current_lr, val_loss)

# Non-validation epochs (still accumulates S)
predictor.observe(epoch, current_lr, None)
```

#### `predict(epochs_remaining, lr_avg=None)`

Predict the final metric after training for more epochs.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs_remaining` | — | How many more epochs will be trained |
| `lr_avg` | None | Average LR over remaining epochs. None = use last observed LR |

Returns predicted final metric value. If not fitted, returns current best.

#### `should_stop(epochs_remaining, threshold=0.005, lr_avg=None)`

Returns `True` if predicted improvement over remaining epochs is below threshold.

```python
# Stop if less than 0.01 improvement expected
if predictor.should_stop(remaining_epochs, threshold=0.01):
    break
```

#### `recommend(epochs_budget, peak_candidates=None, lr_min=5e-6)`

Compare different peak LR candidates and predict which produces the best final metric.

```python
rec = predictor.recommend(
    epochs_budget=600,
    peak_candidates=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
)

print(f"Best peak LR: {rec['best_peak']:.1e}")
print(f"Predicted final: {rec['candidates'][rec['best_peak']]['predicted']:.4f}")
print(f"Recommendation: {rec['action']}")
```

Returns dict:

| Key | Description |
|-----|-------------|
| `best_peak` | Recommended peak LR |
| `candidates` | Per-candidate details (S, k_adjusted, predicted, improvement) |
| `action` | Human-readable recommendation (STOP / OPTIONAL / CONTINUE) |
| `current` | Current best metric |
| `k` | Fitted learning efficiency |
| `B` | Fitted basin floor |

#### `status()` / `format_report(epochs_remaining, ...)`

```python
# One-line status
print(predictor.status())
# [Convergence] k=52.3, B=0.0700, best=0.4680, S=0.07500 (120 obs)

# Full multi-line report
print(predictor.format_report(epochs_remaining=300))
# [Convergence] k=52.3, B=0.0700, best=0.4680, S=0.07500 (120 obs)
#   Remaining 300 epochs:
#     peak=1.0e-05: S=0.00153 k=2615 -> 0.4614 (-0.0066)
#     peak=5.0e-05: S=0.00753 k=523 -> 0.4212 (-0.0468)
#     peak=1.0e-04: S=0.01503 k=262 -> 0.3886 (-0.0794) <-- BEST
#     peak=5.0e-04: S=0.07515 k=52 -> 0.3940 (-0.0740)
#     peak=1.0e-03: S=0.15030 k=26 -> 0.4088 (-0.0592)
#   CONTINUE at peak=1.0e-04 (predicted 0.0794 improvement)
```

#### Properties

| Property | Description |
|----------|-------------|
| `is_fitted` | True after `min_observations` validation observations |
| `S` | Cumulative LR exposure so far |
| `best` | Running-best validation metric |
| `n_observations` | Number of validation observations recorded |
| `k` | Fitted learning efficiency |
| `B` | Fitted basin floor |

### Standalone Utilities

#### `cosine_S(epochs, peak_lr, lr_min=5e-6)`

Compute total LR exposure for a cosine decay schedule: `S = T × (peak + lr_min) / 2`

```python
from forge.training import cosine_S

# 300 epochs, peak=5e-4, min=5e-6
S = cosine_S(300, 5e-4, 5e-6)  # 0.07575
```

#### `predict_from(v_current, k, B, S_remaining)`

Standalone prediction without a predictor instance. Useful for quick calculations or analysis scripts.

```python
from forge.training import predict_from

# Current best = 1.0, k = 50, basin floor = 0.1, remaining S = 0.01
predicted = predict_from(1.0, 50.0, 0.1, 0.01)  # 0.7207
```

### Cross-LR Scaling

The predictor accounts for the tradeoff between LR budget and basin depth:

- **Higher peak LR** → more LR budget (S) per epoch → shallower basins (higher B)
- **Lower peak LR** → less LR budget per epoch → deeper basins (lower B)

This is captured by scaling the learning efficiency:

```
k_eff = k_fitted × (current_peak / candidate_peak)
```

When `peak_lr` is provided to the constructor, `recommend()` uses this scaling automatically.

### Integration with Training Loops

The predictor has no dependencies on PyTorch, Forge, or any framework. It only needs `(epoch, lr, metric)` tuples.

**With Forge Trainer:**
```python
trainer = forge.Trainer(model, optimizer, loss_fn, X_train, y_train,
                        val_data=X_val, val_targets=y_val, epochs=5000)
predictor = ConvergencePredictor(peak_lr=1e-3)

# Use Trainer's post_epoch callback
def on_epoch(epoch, metrics):
    predictor.observe(epoch, optimizer.lr, metrics.get('val_loss'))
    if predictor.is_fitted and predictor.should_stop(5000 - epoch):
        return 'stop'  # signal early stopping

trainer.fit(post_epoch=on_epoch)
```

**With PyTorch:**
```python
predictor = ConvergencePredictor(peak_lr=args.lr)

for epoch in range(args.epochs):
    train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    lr = optimizer.param_groups[0]['lr']

    predictor.observe(epoch, lr, val_loss)

    if predictor.is_fitted:
        print(predictor.status())
        if predictor.should_stop(args.epochs - epoch):
            break
```

**Non-validation epochs:**
```python
for epoch in range(args.epochs):
    lr = scheduler.get_lr()
    train_one_epoch(...)

    if epoch % val_every == 0:
        val_loss = validate(...)
        predictor.observe(epoch, lr, val_loss)
    else:
        predictor.observe(epoch, lr, None)  # track S, no metric
```

### Calibration Data

The harmonic decay model was validated against 7 controlled sweep experiments on a thermal conduction ROM:

| Run | Peak LR | Restarts | Epochs | Final Best | Fitted k | Fitted B |
|-----|---------|----------|--------|-----------|----------|----------|
| S1 | 5e-4 | 0 | 300 | 0.912K | 7.5 | 0.91 |
| S2 | 1e-4 | 0 | 300 | 1.251K | 64.8 | 0.07 |
| S3 | 5e-5 | 0 | 300 | 4.155K | 14.5 | 3.27 |
| S4 | 5e-4 | 1 (T0=75) | 300 | 1.109K | 11.3 | 0.82 |
| S5 | 5e-4 | 1 (T0=150) | 300 | 0.795K | 8.4 | 0.57 |
| v50 | 5e-4 | adaptive | 600 | 0.468K | 52.4 | 0.13 |
| v51 | 5e-4 | adaptive+warmup | 600 | 0.580K | 49.1 | 0.24 |

**Key findings:**
- **k is stable within a run** (CV 31-41%) but varies between basins
- **Harmonic beats exponential** on trajectory MSE, R², and cross-validation
- **Two basin types**: deep (k≈50-65, low B) vs shallow (k≈7-15, high B)
- **Lower peak LR can find deeper basins**: S2 at 1e-4 found B≈0.07 vs S1 at 5e-4 found B≈0.91
- **Half-trajectory prediction accuracy**: ~7% error predicting the second half from the first half

### Limitations

- **Cold start**: Needs ~5 validation observations before making predictions. Prior to that, returns current best.
- **Regime changes**: After a warm restart or LR jump, the predictor may take `min_observations` epochs to recalibrate. Use a smaller `window` if your schedule has frequent restarts.
- **Cross-LR scaling is approximate**: The `k ∝ 1/peak_lr` scaling was calibrated from limited data. It works well for 10x LR range but may be inaccurate for 100x+ differences.
- **Basin floor is local**: B represents the basin the current trajectory is heading toward, not the global minimum. Different LR schedules can find different basins with different B values.

---

## Trainer

GPU-native training loop with CUDA Graph capture. Combined with `forge.nn` and `forge.optim`, each training step is replayed as one CUDA Graph, which removes per-step Python and kernel-launch overhead (most significant for small models where that overhead dominates).

Two modes:
- **Supervised** (`fit()`): Upload data, auto-batch, auto-graph-capture
- **Custom step** (`fit_custom()`): User provides `step_fn(model) -> loss`, Trainer handles the CUDA Graph

```python
# Supervised
trainer = forge.Trainer(model, optimizer, loss_fn, X_train, y_train,
                        val_data=X_val, val_targets=y_val,
                        epochs=5000, batch_size=400)
history = trainer.fit()

# Custom step
history = trainer.fit_custom(
    step_fn=my_step_fn,       # (model) -> scalar loss
    iterations=50000,
    internal=5,               # graph replays per iteration
    pre_step=fill_rng,        # called before each graph replay
    post_iteration=walk_state # called after each internal block
)
```

See `forge/training/trainer.py` for the full API.
