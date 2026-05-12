"""Adversarial scenarios for the EKF: cases that break a naively-tuned filter.

Each helper returns a (description, mu_schedule, sensor_fn) tuple, where:
  - mu_schedule(t) -> mu at time t (used by `simulate(...)`),
  - sensor_fn(t_idx, v_true_i, rng) -> measurement or None for dropouts.

The driver runs the EKF, feeding only the non-None measurements through
`update()` while still calling `predict()` every step. This mirrors how a
real online filter handles missing samples.
"""

from __future__ import annotations

import numpy as np


def mu_step(mu_before: float = 0.8, mu_after: float = 0.35, t_change: float = 2.0):
    """Dry-to-wet road transition at `t_change` seconds."""
    def schedule(ti: float) -> float:
        return mu_before if ti < t_change else mu_after
    return f"mu step {mu_before}->{mu_after} at t={t_change}s", schedule


def dropout_sensor(rate: float = 0.4, burst_len: int = 25, noise_std: float = 0.25):
    """Sensor that drops bursts of measurements (GPS occlusion, tunnel, etc.)."""
    def sensor(i: int, v_true_i: float, rng: np.random.Generator):
        return _burst_dropout(i, v_true_i, rng, rate=rate, burst_len=burst_len, noise_std=noise_std)
    return f"dropout rate={rate} burst={burst_len}", sensor


def biased_sensor(bias: float = 1.5, noise_std: float = 0.25):
    """Sensor with a constant additive bias (e.g. miscalibrated wheel-speed)."""
    def sensor(i: int, v_true_i: float, rng: np.random.Generator):
        return float(v_true_i + bias + rng.normal(0.0, noise_std))
    return f"bias={bias} m/s", sensor


# --- internals ---------------------------------------------------------------

_DROPOUT_STATE: dict[int, int] = {}


def _burst_dropout(i, v_true_i, rng, rate, burst_len, noise_std):
    # Stateful across calls keyed on `id(rng)` so multiple scenarios are independent.
    key = id(rng)
    remaining = _DROPOUT_STATE.get(key, 0)
    if remaining > 0:
        _DROPOUT_STATE[key] = remaining - 1
        return None
    if rng.random() < rate / burst_len:
        _DROPOUT_STATE[key] = burst_len
        return None
    _DROPOUT_STATE[key] = 0
    return float(v_true_i + rng.normal(0.0, noise_std))


def clean_sensor(noise_std: float = 0.25):
    def sensor(i, v_true_i, rng):
        return float(v_true_i + rng.normal(0.0, noise_std))
    return f"baseline (noise σ={noise_std})", sensor
