"""Load real-telemetry CSV logs and reduce them to a single braking event.

Accepts CSVs with columns (time, speed) in (seconds, m/s). This is the format
produced by most OBD-II loggers (after one-line conversion from km/h) and by
the comma2k19 dataset when its `CAN/value` traces are resampled.

The braking segment is extracted automatically: contiguous samples where
speed is monotonically non-increasing and bracketed by a deceleration > 0.5 m/s^2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, comment="#")
    cols = {c.lower(): c for c in df.columns}
    if "time" not in cols or "speed" not in cols:
        raise ValueError(f"CSV must have 'time' and 'speed' columns, got {list(df.columns)}")
    t = df[cols["time"]].to_numpy(dtype=float)
    v = df[cols["speed"]].to_numpy(dtype=float)
    return t, v


def resample_uniform(t: np.ndarray, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    t_uniform = np.arange(t[0], t[-1], dt)
    v_uniform = np.interp(t_uniform, t, v)
    return t_uniform, v_uniform


def extract_braking_event(t: np.ndarray, v: np.ndarray, min_decel: float = 0.5,
                          smooth_window: int = 5):
    """Return the longest contiguous braking segment (a < -min_decel).

    Velocity is smoothed with a centred moving average before differentiation
    so that GPS quantisation noise doesn't fragment the detected segment.
    """
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        v_smooth = np.convolve(v, kernel, mode="same")
    else:
        v_smooth = v
    dt = np.diff(t)
    a = np.diff(v_smooth) / dt
    braking = a < -min_decel

    best_start, best_end, best_len = 0, 0, 0
    i = 0
    while i < len(braking):
        if braking[i]:
            j = i
            while j < len(braking) and braking[j]:
                j += 1
            if j - i > best_len:
                best_len = j - i
                best_start, best_end = i, j
            i = j
        else:
            i += 1

    if best_len == 0:
        raise ValueError("no braking event found in trace")

    sl = slice(best_start, best_end + 1)
    return t[sl] - t[best_start], v[sl]
