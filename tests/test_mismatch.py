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
    v, s, p, n = simulate_truth(28.0, t, effect="grade", intensity=0.0)
    assert v.shape == t.shape
    assert v[-1] < v[0]
    assert np.all(s >= 0.0)
    # brake-pressure factor in [0, 1]
    assert np.all((p >= 0.0) & (p <= 1.0 + 1e-9))
    # lateral utilisation in [0, 1); zero when cornering is not the effect
    assert np.all((n >= 0.0) & (n < 1.0))
    assert np.all(n == 0.0)


def test_grade_adds_deceleration():
    t = np.arange(0, 3.0, 0.01)
    v_flat,  _, _, _ = simulate_truth(28.0, t, effect="grade", intensity=0.0)
    v_steep, _, _, _ = simulate_truth(28.0, t, effect="grade", intensity=0.10)
    assert v_steep[-1] < v_flat[-1] + 1e-6, "uphill should decelerate more"


def test_headwind_adds_drag():
    t = np.arange(0, 3.0, 0.01)
    v_calm, _, _, _ = simulate_truth(28.0, t, effect="headwind", intensity=0.0)
    v_wind, _, _, _ = simulate_truth(28.0, t, effect="headwind", intensity=10.0)
    assert v_wind[-1] < v_calm[-1] + 1e-6


def test_cornering_costs_longitudinal_grip():
    """Spending the friction budget laterally must lengthen the stop.

    The physical invariant behind the friction ellipse: a tire cornering at
    n = 0.6 has sqrt(1 - 0.36) = 80% of its longitudinal capacity left, so the
    same braking input has to take longer. Asserting the ordering rather than
    a number, so it stays a physics check and not a pinned output.
    """
    t = np.arange(0, 3.0, 0.01)
    v_straight, _, _, n0 = simulate_truth(28.0, t, effect="corner", intensity=0.0)
    v_corner,   _, _, n6 = simulate_truth(28.0, t, effect="corner", intensity=0.6)
    assert np.all(n0 == 0.0) and np.allclose(n6, 0.6)
    assert v_corner[-1] > v_straight[-1], (
        "cornering must leave the car FASTER at the end of the same braking "
        "event, because less longitudinal grip is available"
    )


@pytest.mark.slow
def test_no_method_is_broken_at_nominal():
    """Every estimator must be near the noise floor when no effect is active.

    This is the regression guard that was missing. A constant-mu fitter whose
    drag coefficient disagrees with the truth simulator shows a large, roughly
    constant error at EVERY intensity including the nominal one -- which is
    exactly how the 75x drag mismatch between `model.py`'s callers and
    `wheel.py` hid for so long. The old ranking test asserted the NN's mean
    error was ABOVE 10%, so it pinned that bug in place instead of catching it.
    """
    cells = run_sweep()
    nominal = [c for c in cells
               if (c.effect == "grade" and c.intensity == 0.0)
               or (c.effect == "brake" and c.intensity == 0.01)
               or (c.effect == "corner" and c.intensity == 0.0)]
    assert nominal, "no nominal cells found"
    for c in nominal:
        assert c.rmse_norm < 0.05, (
            f"{c.method} is at {100 * c.rmse_norm:.1f}% RMSE/v0 with effect "
            f"'{c.effect}' at its nominal intensity -- it should be near the "
            f"noise floor. Suspect a model/parameter inconsistency, not "
            f"mismatch sensitivity."
        )


@pytest.mark.slow
def test_sweep_qualitative_ranking():
    """Batch is most robust; the brake ramp is what separates the rest."""
    cells = run_sweep()
    by_method = {}
    for c in cells:
        by_method.setdefault(c.method, []).append(c.rmse_norm)
    means = {m: float(np.mean(v)) for m, v in by_method.items()}

    # The offline batch fit carries k_drag as a free parameter, so it absorbs
    # structural error the online methods have to eat. It should win overall.
    assert means["Batch (SciPy)"] == min(means.values()), means

    # The headline finding: on the brake ramp -- time-varying friction that a
    # slip-only model cannot represent -- the brake-aware PINN holds while
    # every other online method degrades badly.
    worst_brake = {}
    for c in cells:
        if c.effect == "brake":
            worst_brake[c.method] = max(worst_brake.get(c.method, 0.0), c.rmse_norm)
    assert worst_brake["PINN-B (brake-aware)"] < 0.10, worst_brake
    for other in ("EKF", "NN (FrictionNet)", "PINN"):
        assert worst_brake[other] > 2 * worst_brake["PINN-B (brake-aware)"], worst_brake
