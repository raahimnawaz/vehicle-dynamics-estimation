"""The telemetry loader must round-trip a CSV and pull the braking event out
of a trace that includes pre-brake coast and post-brake rest.
"""

import os

import numpy as np

from src.data.telemetry import extract_braking_event, load_csv, resample_uniform


def test_load_sample_csv():
    path = os.path.join("data", "sample_braking.csv")
    t, v = load_csv(path)
    assert len(t) == len(v) > 50
    assert t[0] >= 0 and np.all(np.diff(t) > 0)


def test_extracts_braking_segment_from_synthetic_trace():
    dt = 0.1
    t = np.arange(0, 10, dt)
    v = np.full_like(t, 25.0)
    # brake from t=2 to t=5
    mask = (t >= 2) & (t <= 5)
    v[mask] = 25.0 - 6.0 * (t[mask] - 2.0)
    v[t > 5] = v[mask][-1]

    t_b, v_b = extract_braking_event(t, v)
    assert len(t_b) >= 25
    assert v_b[0] > v_b[-1]


def test_resample_uniform_preserves_endpoints():
    t = np.array([0.0, 0.1, 0.25, 0.4])
    v = np.array([10.0, 9.0, 7.5, 6.0])
    t_u, v_u = resample_uniform(t, v, 0.05)
    assert abs(t_u[0] - 0.0) < 1e-9
    assert t_u[-1] <= 0.4
    assert abs(v_u[0] - 10.0) < 1e-9
