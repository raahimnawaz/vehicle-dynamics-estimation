"""Run the EKF through an adversarial scenario and record state + covariance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.estimation.kalman import VehicleEKF
from src.simulation.run_sim import simulate


@dataclass
class ScenarioResult:
    label: str
    t: np.ndarray
    v_true: np.ndarray
    v_meas: np.ndarray            # NaN where dropped
    mu_true: np.ndarray
    mu_est: np.ndarray
    mu_sigma: np.ndarray          # sqrt(P[1,1]) at each step


def run_scenario(
    label: str,
    mu_schedule,
    sensor_fn,
    *,
    v0: float = 30.0,
    dt: float = 0.01,
    t_final: float = 6.0,
    k: float = 0.02,
    q_mu: float = 1e-3,
    seed: int = 0,
) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_final, dt)

    v_true = simulate([mu_schedule, k], v0, t, dt)
    mu_true = np.array([mu_schedule(ti) for ti in t])

    ekf = VehicleEKF(mu_init=0.5, v_init=v0)
    ekf.Q[1, 1] = q_mu  # let mu adapt; baseline 1e-4 is too cold for step changes

    v_meas = np.full_like(t, np.nan)
    mu_est = np.zeros_like(t)
    mu_sigma = np.zeros_like(t)

    for i, vi in enumerate(v_true):
        ekf.predict(dt)
        z = sensor_fn(i, vi, rng)
        if z is not None and vi > 1.0:
            ekf.update(z)
            v_meas[i] = z
        mu_est[i] = ekf.x[1]
        mu_sigma[i] = float(np.sqrt(max(ekf.P[1, 1], 0.0)))

    return ScenarioResult(
        label=label,
        t=t,
        v_true=v_true,
        v_meas=v_meas,
        mu_true=mu_true,
        mu_est=mu_est,
        mu_sigma=mu_sigma,
    )
