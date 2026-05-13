"""Sanity checks on the model-mismatch sweep.

These pin the qualitative behaviour visible in results/mismatch_*.png so a
regression in any estimator (or in the augmented truth simulator) surfaces
immediately.
"""

import numpy as np
import pytest

from src.scenarios.mismatch import EFFECTS, run_sweep, simulate_truth


def test_truth_simulator_decreasing():
    t = np.arange(0, 3.0, 0.01)
    v, s, p = simulate_truth(28.0, t, effect="grade", intensity=0.0)
    assert v.shape == t.shape
    assert v[-1] < v[0]
    assert np.all(s >= 0.0)
    # brake-pressure factor in [0, 1]
    assert np.all((p >= 0.0) & (p <= 1.0 + 1e-9))


def test_grade_adds_deceleration():
    t = np.arange(0, 3.0, 0.01)
    v_flat,  _, _ = simulate_truth(28.0, t, effect="grade", intensity=0.0)
    v_steep, _, _ = simulate_truth(28.0, t, effect="grade", intensity=0.10)
    assert v_steep[-1] < v_flat[-1] + 1e-6, "uphill should decelerate more"


def test_headwind_adds_drag():
    t = np.arange(0, 3.0, 0.01)
    v_calm, _, _ = simulate_truth(28.0, t, effect="headwind", intensity=0.0)
    v_wind, _, _ = simulate_truth(28.0, t, effect="headwind", intensity=10.0)
    assert v_wind[-1] < v_calm[-1] + 1e-6


@pytest.mark.slow
def test_sweep_qualitative_ranking():
    """Batch should be the most robust, NN the most out-of-distribution."""
    cells = run_sweep()
    by_method = {}
    for c in cells:
        by_method.setdefault(c.method, []).append(c.rmse_norm)
    means = {m: float(np.mean(v)) for m, v in by_method.items()}

    # Batch has the lowest mean RMSE.
    assert means["Batch (SciPy)"] == min(means.values()), means
    # NN is consistently mismatched (>10% RMSE/v0 on average).
    assert means["NN (FrictionNet)"] > 0.10
    # Either PINN beats EKF on average.
    assert means["PINN"] < means["EKF"]
