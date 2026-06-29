"""On noiseless synthetic data the SciPy batch estimator must recover (mu, k)
within tight tolerance. If this regresses, the loss surface or the optimiser
configuration has drifted.
"""

import numpy as np

from src.estimation.optimize import estimate
from src.simulation.run_sim import simulate


def test_recovers_parameters_noiseless():
    true_mu, true_k = 0.7, 0.02
    v0, dt = 30.0, 0.01
    t = np.arange(0, 4, dt)
    v = simulate([true_mu, true_k], v0, t, dt)

    mu_est, k_est = estimate(v, v0, t, dt)

    assert abs(mu_est - true_mu) < 1e-3, f"mu off: got {mu_est}, want {true_mu}"
    assert abs(k_est - true_k) < 1e-3, f"k off: got {k_est}, want {true_k}"


def test_recovers_mu_under_modest_noise():
    rng = np.random.default_rng(0)
    true_mu, true_k = 0.7, 0.02
    v0, dt = 30.0, 0.01
    t = np.arange(0, 4, dt)
    v = simulate([true_mu, true_k], v0, t, dt) + rng.normal(0, 0.1, size=len(t))

    mu_est, _ = estimate(v, v0, t, dt)
    assert abs(mu_est - true_mu) < 0.02
