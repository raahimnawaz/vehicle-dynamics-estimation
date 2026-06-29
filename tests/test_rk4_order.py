"""RK4 must converge at 4th order on a problem with a known closed-form solution.

We integrate dv/dt = -lambda * v from v0=1 over [0, 1] and compare the terminal
error against the analytic solution e^{-lambda}. Halving dt should drop the
error by ~16x for a 4th-order method.
"""

import math

import numpy as np

from src.solvers.rk4 import rk4_step


def _decay(v, lam):
    return -lam * v


def _integrate(dt: float, lam: float = 1.0, t_final: float = 1.0) -> float:
    v = 1.0
    n = int(round(t_final / dt))
    for _ in range(n):
        v = rk4_step(_decay, v, dt, lam)
    return v


def test_rk4_fourth_order_convergence():
    lam = 1.0
    truth = math.exp(-lam)
    dts = [0.1, 0.05, 0.025, 0.0125]
    errs = [abs(_integrate(dt, lam) - truth) for dt in dts]

    # Each halving of dt should reduce error by ~16x for a 4th-order scheme.
    # Allow some slack at the smallest dt where floating-point noise creeps in.
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    for p in orders:
        assert 3.7 < p < 4.3, f"observed order {p} not ~4 ({orders})"


def test_rk4_matches_truth_at_fine_dt():
    truth = math.exp(-1.0)
    assert abs(_integrate(1e-3) - truth) < 1e-9
