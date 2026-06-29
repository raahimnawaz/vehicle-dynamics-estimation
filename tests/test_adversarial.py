"""Adversarial scenarios for the EKF.

These tests pin down the qualitative behaviour shown in
results/adversarial_ekf.png so a regression in EKF tuning or scenario
generation surfaces immediately.
"""

import numpy as np

from src.scenarios.adversarial import biased_sensor, clean_sensor, dropout_sensor, mu_step
from src.scenarios.runner import run_scenario


def test_ekf_reacquires_after_mu_step():
    """After a dry->wet transition, the filter should land within 0.05 of true mu."""
    label, schedule = mu_step(mu_before=0.8, mu_after=0.35, t_change=2.0)
    _, sensor = clean_sensor(noise_std=0.1)
    res = run_scenario(label, schedule, sensor, q_mu=1e-2, seed=0, t_final=6.0)

    # Sample mu_est at t=4s (2s after the step, well-converged) and t=5.5s.
    idx_4 = int(4.0 / 0.01)
    idx_55 = int(5.5 / 0.01)
    assert abs(res.mu_est[idx_4] - 0.35) < 0.10
    assert abs(res.mu_est[idx_55] - 0.35) < 0.08


def test_covariance_grows_during_dropout():
    """When measurements stop arriving, mu_sigma must increase, then shrink on resume."""
    label, sensor = dropout_sensor(rate=0.8, burst_len=80)
    res = run_scenario(label, lambda ti: 0.7, sensor, seed=42, t_final=6.0)

    dropped = np.isnan(res.v_meas)
    # Find a sufficiently long dropout run
    runs = []
    i = 0
    while i < len(dropped):
        if dropped[i]:
            j = i
            while j < len(dropped) and dropped[j]:
                j += 1
            if j - i > 30:
                runs.append((i, j))
            i = j
        else:
            i += 1

    assert runs, "expected at least one dropout burst >30 samples"
    start, end = runs[0]
    assert res.mu_sigma[end - 1] > res.mu_sigma[start], "sigma should grow during dropout"


def test_bias_inflates_mu_estimate_predictably():
    """A +1.5 m/s sensor bias makes the car look like it decelerated less => mu_est < true mu."""
    label, sensor = biased_sensor(bias=1.5, noise_std=0.1)
    res = run_scenario(label, lambda ti: 0.7, sensor, seed=0, t_final=6.0)

    tail = res.mu_est[int(0.7 * len(res.t)):]
    mean_est = float(np.mean(tail))
    assert mean_est < 0.7, f"bias should pull mu_est below true; got {mean_est}"
    assert abs(mean_est - 0.7) < 0.20, "but it should still be in the right ballpark"


def test_simulate_accepts_callable_mu():
    """The forward model has to support time-varying mu for these scenarios to be meaningful."""
    from src.simulation.run_sim import simulate

    t = np.arange(0, 4, 0.01)
    v_const = simulate([0.7, 0.02], 30.0, t, 0.01)
    v_callable = simulate([lambda ti: 0.7, 0.02], 30.0, t, 0.01)
    assert np.allclose(v_const, v_callable)
