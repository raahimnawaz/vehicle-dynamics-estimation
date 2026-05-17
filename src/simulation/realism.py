"""Sensor-noise model for synthetic benchmarks.

Used by `reproduce.py::run_synthetic` and `main.py` to corrupt clean
forward-rolled trajectories before handing them to the estimators. Default
std=0.25 m/s is the noise floor of typical fused INS / wheel-speed; raise it
to ~1.0 to mimic raw GPS.

For per-run reproducibility, seed numpy's global RNG before calling.
"""

from __future__ import annotations

import numpy as np


def add_noise(v: np.ndarray, std: float = 0.25) -> np.ndarray:
    return v + np.random.normal(0, std, size=len(v))
