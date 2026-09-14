"""On noiseless synthetic data the SciPy batch estimator must recover (mu, k)
within tight tolerance. If this regresses, the loss surface or the optimiser
configuration has drifted.
"""

import numpy as np

from src.estimation.optimize import estimate
from src.simulation.run_sim import simulate
from src.physics.wheel import K_DRAG


def test_recovers_parameters_noiseless():
    # true_k is deliberately OFF the optimiser's seed (estimate() starts k at
    # K_DRAG): with the truth equal to the seed the k half of "recovery" started
    # at the answer and a broken k-search could still pass (D-04). At 1.5*K_DRAG
    # the simplex must actually travel to find it.
    true_mu, true_k = 0.7, 1.5 * K_DRAG
    v0, dt = 30.0, 0.01
    t = np.arange(0, 4, dt)
    v = simulate([true_mu, true_k], v0, t, dt)

    mu_est, k_est = estimate(v, v0, t, dt)

    assert abs(mu_est - true_mu) < 1e-3, f"mu off: got {mu_est}, want {true_mu}"
    # Relative, not absolute: K_DRAG is ~2.7e-4, so the absolute 1e-3 bound this
    # replaces was ~4x the value it guards and would have passed for k_est = 0.
    # The bound is 2e-2 (not tighter) because Nelder-Mead's default xatol=1e-4 is
    # an ABSOLUTE simplex tolerance, so at this scale it stops refining k while
    # still a few e-3 out; mu, being O(1), lands far tighter.
    assert abs(k_est - true_k) / true_k < 2e-2, f"k off: got {k_est}, want {true_k}"


def test_recovers_mu_under_modest_noise():
    rng = np.random.default_rng(0)
    true_mu, true_k = 0.7, K_DRAG
    v0, dt = 30.0, 0.01
    t = np.arange(0, 4, dt)
    v = simulate([true_mu, true_k], v0, t, dt) + rng.normal(0, 0.1, size=len(t))

    mu_est, _ = estimate(v, v0, t, dt)
    assert abs(mu_est - true_mu) < 0.02
