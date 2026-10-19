"""Convergence predictor for training optimization.

Predicts where a training run is heading and recommends optimal LR schedules.
Based on the harmonic decay model:

    dv/dS = -k * (v - B)^2

    => v(S) = B + (v0 - B) / (1 + k * (v0 - B) * S)

Where:
    v  = running-best validation metric (loss, MAE, etc.)
    S  = cumulative LR exposure: integral(lr * dt) over training
    B  = basin floor (the local minimum this trajectory converges toward)
    k  = learning efficiency (how fast the model converts LR budget into improvement)

Key insight: learning efficiency k is roughly constant within a basin but
varies between basins. Higher peak LR explores more aggressively but lands
in shallower basins (higher B). This tradeoff is captured by the
LR-adjusted k scaling.

Usage:
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

            # Early stopping
            if predictor.should_stop(remaining):
                break

    # Schedule recommendations for future runs
    print(predictor.recommend(epochs_budget=600))

Works with any decreasing metric (loss, MAE, RMSE, etc.). The metric does
NOT need to be smooth — the predictor uses running-best and tolerates noise.

No external dependencies (pure Python + math).
"""
import math


class ConvergencePredictor:
    """Predicts training convergence using the harmonic decay model.

    The predictor tracks cumulative LR exposure (S) and running-best
    validation metric (v), then fits a harmonic decay curve to predict
    where training is heading.

    This is model-agnostic: works with any training loop that has a
    learning rate and a validation metric that decreases over time.

    Args:
        k_prior: Prior for learning efficiency. Higher = faster convergence
            per unit of LR budget. Default 50 works well for deep basins.
            Shallow basins typically have k ~10.
        B_prior: Prior for basin floor. 0 = no prior knowledge (conservative).
        min_observations: Number of validation observations required before
            making predictions. Default 5.
        window: Number of recent observations to use for fitting. Larger
            windows are more stable but slower to adapt to regime changes
            (e.g., after a learning rate restart). Default 20.
        peak_lr: The peak LR of the current schedule. Required for cross-LR
            predictions (comparing different peak LR candidates). If not
            provided, cross-LR scaling is disabled.
    """

    def __init__(self, k_prior=50.0, B_prior=0.0, min_observations=5,
                 window=20, peak_lr=None):
        self._k_prior = k_prior
        self._B_prior = B_prior
        self._min_obs = min_observations
        self._window = window
        self._peak_lr = peak_lr

        # Observation state
        self._observations = []  # [(epoch, S, metric, best_metric)]
        self._S = 0.0            # cumulative LR exposure
        self._prev_epoch = 0
        self._prev_lr = 0.0
        self._best = float('inf')

        # Fitted model parameters
        self.k = k_prior     # learning efficiency
        self.B = B_prior     # basin floor
        self._fitted = False

    # ── Core API ──────────────────────────────────────────────────────

    def observe(self, epoch, lr, metric=None):
        """Record a training observation.

        Call this every epoch (or at least every validation epoch).
        For non-validation epochs, pass metric=None to still track S.

        Args:
            epoch: Current epoch number.
            lr: Current learning rate.
            metric: Validation metric (loss, MAE, etc.), or None if no
                validation was done this epoch.
        """
        # Accumulate S via trapezoidal rule
        if self._prev_epoch > 0:
            dt = epoch - self._prev_epoch
            if dt > 0:
                self._S += 0.5 * (self._prev_lr + lr) * dt

        self._prev_epoch = epoch
        self._prev_lr = lr

        if metric is None:
            return

        # Update running best
        if metric < self._best:
            self._best = metric

        self._observations.append((epoch, self._S, metric, self._best))

        # Re-fit if enough data
        if len(self._observations) >= self._min_obs:
            self._fit()

    def predict(self, epochs_remaining, lr_avg=None):
        """Predict the final metric after training for more epochs.

        Args:
            epochs_remaining: How many more epochs will be trained.
            lr_avg: Average LR over remaining epochs. If None, uses
                the last observed LR (conservative for decaying schedules).

        Returns:
            Predicted final metric value, or current best if not fitted.
        """
        if not self._fitted or self._best <= self.B:
            return self._best

        S_remaining = self._estimate_S(epochs_remaining, lr_avg)
        gap = self._best - self.B
        return self.B + gap / (1.0 + self.k * gap * S_remaining)

    def should_stop(self, epochs_remaining, threshold=0.005, lr_avg=None):
        """Check if continuing training would yield meaningful improvement.

        Args:
            epochs_remaining: How many more epochs are planned.
            threshold: Minimum predicted improvement to justify continuing.
            lr_avg: Average LR over remaining epochs.

        Returns:
            True if predicted improvement is below threshold.
        """
        if not self._fitted:
            return False  # Can't predict yet, don't stop

        predicted = self.predict(epochs_remaining, lr_avg)
        improvement = self._best - predicted
        return improvement < threshold

    def recommend(self, epochs_budget, peak_candidates=None, lr_min=5e-6):
        """Recommend the best LR schedule for a given epoch budget.

        Compares different peak LR candidates and predicts which one
        will reach the lowest final metric, accounting for the tradeoff
        between LR budget (S) and basin depth.

        Args:
            epochs_budget: Total epochs available for the next training run.
            peak_candidates: List of peak LRs to evaluate.
                Default: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
            lr_min: Minimum LR for cosine schedule S calculation.

        Returns:
            dict with keys:
                - best_peak: Recommended peak LR
                - candidates: Per-candidate details
                - action: Human-readable recommendation
                - current: Current best metric
                - k: Fitted learning efficiency
                - B: Fitted basin floor
        """
        if peak_candidates is None:
            peak_candidates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

        candidates = {}
        for peak in peak_candidates:
            S = cosine_S(epochs_budget, peak, lr_min)
            k_adj = self._k_for_peak(peak)
            v = self._best

            if v <= self.B:
                v_pred = self.B
            else:
                gap = v - self.B
                v_pred = self.B + gap / (1.0 + k_adj * gap * S)

            candidates[peak] = {
                'S': S,
                'k_adjusted': k_adj,
                'predicted': v_pred,
                'improvement': self._best - v_pred,
            }

        best_peak = min(candidates, key=lambda p: candidates[p]['predicted'])
        max_improvement = max(c['improvement'] for c in candidates.values())

        if max_improvement < 0.005:
            action = 'STOP: diminishing returns (max improvement < 0.005)'
        elif max_improvement < 0.02:
            action = 'OPTIONAL: marginal gains (%.4f)' % max_improvement
        else:
            action = 'CONTINUE at peak=%.1e (predicted %.4f improvement)' % (
                best_peak, candidates[best_peak]['improvement'])

        return {
            'best_peak': best_peak,
            'candidates': candidates,
            'action': action,
            'current': self._best,
            'k': self.k,
            'B': self.B,
        }

    # ── Properties ────────────────────────────────────────────────────

    @property
    def is_fitted(self):
        """True if the model has been fitted (enough observations)."""
        return self._fitted

    @property
    def S(self):
        """Cumulative LR exposure so far."""
        return self._S

    @property
    def best(self):
        """Running-best validation metric."""
        return self._best

    @property
    def n_observations(self):
        """Number of validation observations recorded."""
        return len(self._observations)

    # ── Display ───────────────────────────────────────────────────────

    def status(self):
        """One-line status string."""
        if not self._fitted:
            return '[Convergence] Not fitted (%d/%d observations)' % (
                len(self._observations), self._min_obs)
        return '[Convergence] k=%.1f, B=%.4f, best=%.4f, S=%.5f (%d obs)' % (
            self.k, self.B, self._best, self._S, len(self._observations))

    def format_report(self, epochs_remaining, peak_candidates=None,
                      lr_min=5e-6):
        """Formatted multi-line report with schedule recommendations."""
        rec = self.recommend(epochs_remaining, peak_candidates, lr_min)
        lines = [self.status()]
        lines.append('  Remaining %d epochs:' % epochs_remaining)
        for peak in sorted(rec['candidates'].keys()):
            c = rec['candidates'][peak]
            marker = ' <-- BEST' if peak == rec['best_peak'] else ''
            lines.append('    peak=%.1e: S=%.5f k=%.0f -> %.4f (%+.4f)%s' % (
                peak, c['S'], c['k_adjusted'], c['predicted'],
                -c['improvement'], marker))
        lines.append('  %s' % rec['action'])
        return '\n'.join(lines)

    def __repr__(self):
        if not self._fitted:
            return 'ConvergencePredictor(not fitted, %d obs)' % len(self._observations)
        return ('ConvergencePredictor(k=%.1f, B=%.4f, best=%.4f, S=%.5f)' %
                (self.k, self.B, self._best, self._S))

    # ── Internal ──────────────────────────────────────────────────────

    def _fit(self):
        """Fit k and B from recent observations."""
        obs = self._observations
        if len(obs) > self._window:
            obs = obs[-self._window:]

        # Extract improving trajectory (skip flat regions)
        improving = []
        prev = None
        for _, S, _, v_best in obs:
            if prev is None or v_best < prev - 1e-8:
                improving.append((S, v_best))
                prev = v_best

        if len(improving) < 2:
            return

        # Grid search B: find basin floor that gives best trajectory fit
        v_min = improving[-1][1]
        best_B = self._B_prior
        best_score = float('inf')
        best_k = self._k_prior

        for B_pct in range(0, 85, 5):
            B_cand = v_min * B_pct / 100.0
            if B_cand >= v_min:
                continue

            # Compute k from first and last improving points
            S0, v0 = improving[0]
            S1, v1 = improving[-1]
            dS = S1 - S0
            if dS < 1e-10 or v0 <= B_cand or v1 <= B_cand:
                continue

            k_cand = (1.0 / (v1 - B_cand) - 1.0 / (v0 - B_cand)) / dS
            if k_cand <= 0:
                continue

            # Score: trajectory MSE
            gap0 = v0 - B_cand
            mse = 0.0
            n = 0
            for Si, vi in improving[1:]:
                ds = Si - S0
                if ds <= 0:
                    continue
                v_pred = B_cand + gap0 / (1.0 + k_cand * gap0 * ds)
                mse += (v_pred - vi) ** 2
                n += 1

            if n > 0:
                mse /= n
                if mse < best_score:
                    best_score = mse
                    best_B = B_cand
                    best_k = k_cand

        self.B = best_B
        self.k = best_k
        self._fitted = True

    def _estimate_S(self, epochs_remaining, lr_avg=None):
        """Estimate remaining S for a given number of epochs."""
        if lr_avg is not None:
            return epochs_remaining * lr_avg
        # Conservative: use last observed LR (decaying schedule → lower bound)
        return epochs_remaining * self._prev_lr

    def _k_for_peak(self, candidate_peak):
        """Scale k for a different peak LR (cross-schedule prediction).

        Higher peak LR → shallower basins → lower effective k.
        Scaling: k_eff = k_fitted * (current_peak / candidate_peak)
        """
        if self._peak_lr is None or self._peak_lr <= 0 or candidate_peak <= 0:
            return self.k
        return self.k * (self._peak_lr / candidate_peak)


# ── Utility functions ─────────────────────────────────────────────────

def cosine_S(epochs, peak_lr, lr_min=5e-6):
    """Compute total LR exposure S for a cosine decay schedule.

    For cosine decay from peak_lr to lr_min over T epochs:
        S ≈ T * (peak_lr + lr_min) / 2

    Args:
        epochs: Number of epochs.
        peak_lr: Peak learning rate.
        lr_min: Minimum learning rate. Default: 5e-6.

    Returns:
        Total LR exposure S.
    """
    return epochs * (peak_lr + lr_min) / 2.0


def predict_from(v_current, k, B, S_remaining):
    """Standalone prediction without a predictor instance.

    Useful for quick calculations or analysis scripts.

    Args:
        v_current: Current best metric value.
        k: Learning efficiency.
        B: Basin floor.
        S_remaining: Remaining LR exposure budget.

    Returns:
        Predicted final metric.
    """
    if v_current <= B:
        return B
    gap = v_current - B
    return B + gap / (1.0 + k * gap * S_remaining)
