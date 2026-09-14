"""Sensor-noise model for synthetic benchmarks.

Used by `reproduce.py::run_synthetic` to corrupt clean forward-rolled
trajectories before handing them to the estimators. Default std=0.25 m/s is the
noise floor of typical fused INS / wheel-speed; raise it to ~1.0 to mimic raw
GPS.

Pass an explicit `numpy.random.Generator` for reproducibility. The old version
drew from numpy's *global* legacy RNG, so results depended on global seeding
order and were inconsistent with the `default_rng(seed)` Generators used in
`src/ml/pinn.py` (formal-inspection defect D-06).
"""

from __future__ import annotations

import numpy as np


def add_noise(v: np.ndarray, std: float = 0.25,
              rng: np.random.Generator | None = None) -> np.ndarray:
    gen = np.random.default_rng() if rng is None else rng
    return v + gen.normal(0, std, size=len(v))
