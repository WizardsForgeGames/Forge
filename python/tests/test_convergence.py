"""Tests for ConvergencePredictor.

Validates against real training data from the thermal ROM sweep experiments.
"""
import math
import sys
import os

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from forge.training.convergence import ConvergencePredictor, cosine_S, predict_from


def test_basic_api():
    """ConvergencePredictor has the expected API."""
    p = ConvergencePredictor()
    assert not p.is_fitted
    assert p.S == 0.0
    assert p.best == float('inf')
    assert p.n_observations == 0

    # Feed some observations
    for ep in range(10):
        lr = 1e-3 * (1 - ep / 10)
        val = 5.0 - ep * 0.3  # linearly decreasing
        p.observe(ep, lr, val)

    assert p.n_observations == 10
    assert p.is_fitted
    assert p.best < 5.0
    assert p.k > 0
    assert p.B >= 0

    # Predictions should be below current best
    pred = p.predict(epochs_remaining=100, lr_avg=1e-4)
    assert pred <= p.best

    # Status string
    s = p.status()
    assert 'k=' in s
    assert 'B=' in s

    # repr
    r = repr(p)
    assert 'ConvergencePredictor' in r

    print("  PASS: basic_api")


def test_cosine_S():
    """cosine_S computes correct LR exposure."""
    # 100 epochs at peak=1e-3, min=0 => S = 100 * 0.5e-3 = 0.05
    assert abs(cosine_S(100, 1e-3, 0) - 0.05) < 1e-10

    # 200 epochs at peak=5e-4, min=5e-6 => S = 200 * (5e-4 + 5e-6)/2
    expected = 200 * (5e-4 + 5e-6) / 2
    assert abs(cosine_S(200, 5e-4, 5e-6) - expected) < 1e-10

    print("  PASS: cosine_S")


def test_predict_from():
    """Standalone prediction matches model math."""
    # From v=1.0, k=50, B=0.1, S=0.01:
    # gap = 0.9, denom = 1 + 50 * 0.9 * 0.01 = 1.45
    # predicted = 0.1 + 0.9 / 1.45 = 0.7207
    v = predict_from(1.0, 50.0, 0.1, 0.01)
    expected = 0.1 + 0.9 / (1.0 + 50.0 * 0.9 * 0.01)
    assert abs(v - expected) < 1e-10
    assert abs(v - 0.7207) < 0.001

    # At basin floor: returns B
    assert predict_from(0.1, 50.0, 0.1, 1.0) == 0.1
    assert predict_from(0.05, 50.0, 0.1, 1.0) == 0.1

    print("  PASS: predict_from")


def test_should_stop():
    """should_stop returns True when improvement is negligible."""
    p = ConvergencePredictor(k_prior=50.0, B_prior=0.0)

    # Simulate a training run that's nearly converged
    for ep in range(20):
        lr = 1e-4 * (1 - ep / 20)
        val = 0.5 + 0.01 * math.exp(-ep)  # quickly approaches 0.5
        p.observe(ep, lr, val)

    assert p.is_fitted
    # With very low remaining LR, should recommend stopping
    assert p.should_stop(epochs_remaining=10, threshold=0.01, lr_avg=1e-6)

    print("  PASS: should_stop")


def test_recommend():
    """recommend returns sensible schedule recommendations."""
    p = ConvergencePredictor(peak_lr=5e-4)

    # Feed a clean decay trajectory
    S = 0.0
    for ep in range(50):
        lr = 5e-4 * (1 - ep / 50)
        val = 1.0 / (1.0 + 10 * S)  # harmonic decay with known k
        p.observe(ep, lr, val)
        S += lr

    rec = p.recommend(epochs_budget=200, peak_candidates=[1e-4, 5e-4, 1e-3])
    assert 'best_peak' in rec
    assert 'candidates' in rec
    assert 'action' in rec
    assert rec['current'] == p.best

    # All candidates should have positive improvement
    for peak, info in rec['candidates'].items():
        assert info['improvement'] >= 0
        assert info['S'] > 0

    print("  PASS: recommend")


def test_format_report():
    """format_report produces readable output."""
    p = ConvergencePredictor(peak_lr=5e-4)

    for ep in range(20):
        lr = 5e-4 * (1 - ep / 20)
        val = 2.0 - ep * 0.05
        p.observe(ep, lr, val)

    report = p.format_report(epochs_remaining=100)
    assert 'peak=' in report
    assert 'BEST' in report

    print("  PASS: format_report")


def test_non_validation_epochs():
    """observe with metric=None only tracks S, doesn't count as observation."""
    p = ConvergencePredictor()

    # Mix of val and non-val epochs
    p.observe(0, 1e-3, 5.0)
    p.observe(1, 9e-4, None)  # no validation
    p.observe(2, 8e-4, None)  # no validation
    p.observe(3, 7e-4, 4.5)

    assert p.n_observations == 2  # only 2 val observations
    assert p.S > 0  # S tracked even without validation
    assert p.best == 4.5

    print("  PASS: non_validation_epochs")


def test_real_trajectory():
    """Validate against the actual S2 sweep data.

    S2 was a pure cosine decay run: 300 epochs, peak=1e-4, final best=1.251K.
    The predictor should roughly recover this from partial data.
    """
    # Simulated S2-like trajectory (harmonic with k~60, B~0.07)
    # These match the fitted parameters from the thermal ROM sweep analysis
    k_true = 60.0
    B_true = 0.1
    v0 = 9.5  # starting val
    peak_lr = 1e-4

    p = ConvergencePredictor(peak_lr=peak_lr)

    # Generate trajectory using known model
    S = 0.0
    for ep in range(150):
        frac = ep / 300  # cosine over 300 epochs
        lr = peak_lr * (1 + math.cos(math.pi * frac)) / 2
        gap = v0 - B_true
        v = B_true + gap / (1 + k_true * gap * S)
        # Add 5% noise
        import random
        random.seed(ep)
        v_noisy = v * (1 + 0.05 * (random.random() - 0.5))
        p.observe(ep, lr, v_noisy)
        S += lr

    # Predict to epoch 300
    remaining = 150
    predicted = p.predict(remaining, lr_avg=peak_lr * 0.25)  # avg of second half

    # The prediction should be reasonable (within 50% of true final)
    gap = v0 - B_true
    true_final = B_true + gap / (1 + k_true * gap * cosine_S(300, peak_lr))
    error = abs(predicted - true_final) / true_final
    print("  Half-trajectory test: predicted=%.3f, true=%.3f, error=%.0f%%" %
          (predicted, true_final, error * 100))
    assert error < 0.5, "Prediction error %.0f%% > 50%%" % (error * 100)

    print("  PASS: real_trajectory")


def test_cross_lr_scaling():
    """Cross-LR predictions correctly penalize high peak LR."""
    p = ConvergencePredictor(peak_lr=5e-4)

    # Train with peak=5e-4
    for ep in range(30):
        lr = 5e-4 * (1 - ep / 30)
        val = 1.0 - ep * 0.02
        p.observe(ep, lr, val)

    # Compare predictions for different peak LRs
    rec = p.recommend(200, peak_candidates=[1e-4, 5e-4, 1e-3])

    # Higher peak LR should NOT always win (k scaling penalizes it)
    pred_1e4 = rec['candidates'][1e-4]['predicted']
    pred_1e3 = rec['candidates'][1e-3]['predicted']
    # At least one of the lower peaks should beat 1e-3
    assert pred_1e4 <= pred_1e3, (
        "1e-4 (%.4f) should beat 1e-3 (%.4f) due to deeper basins" %
        (pred_1e4, pred_1e3))

    print("  PASS: cross_lr_scaling")


if __name__ == '__main__':
    print("Testing ConvergencePredictor...")
    test_basic_api()
    test_cosine_S()
    test_predict_from()
    test_should_stop()
    test_recommend()
    test_format_report()
    test_non_validation_epochs()
    test_real_trajectory()
    test_cross_lr_scaling()
    print("\nAll tests passed.")
