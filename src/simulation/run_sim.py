"""Forward roll-out of the constant-form longitudinal braking model.

RK4 is implemented inline here rather than calling
`src/solvers/rk4.py::rk4_step`, for the same reason `wheel.simulate` does it:
`mu` is allowed to vary with time, and the generic helper is *autonomous* --
it evaluates the right-hand side only at the start of the step. Feeding it a
time-varying input silently drops the integrator to 1st order, which is
exactly what this module used to do. Measured on a smooth dry-to-wet
transition, error fell 2.00x per halving of dt instead of 16x:

    dt        via rk4_step      correct RK4
    0.0400    8.770e-02         2.529e-07
    0.0200    4.385e-02         1.573e-08
    0.0100    2.192e-02         9.821e-10
    0.0050    1.096e-02         6.133e-11

No published figure moved, because every schedule this repo actually ships is
piecewise constant in time -- `adversarial.mu_step` and the constant-mu
benchmarks -- and a piecewise-constant mu is autonomous on each side of the
jump. The gap on the step schedule was 7.4e-3 m/s against 0.25 m/s of sensor
noise. It was a trap set for the next person to write a realistic mu(t), not
a live error, and `tests/test_rk4_order.py::test_simulate_is_fourth_order_in_a
_time_varying_mu` now fails if it is re-set.
"""

import numpy as np
from src.physics.model import dvdt


def simulate(theta, v0, t, dt, m=1500, g=9.81):
    """Forward roll-out of the longitudinal braking model.

    `theta` is `(mu, k)`. `mu` may be:
      - a scalar,
      - a callable `mu(t_seconds) -> float` (for time-varying friction), or
      - a 1-D array of the same length as `t` (precomputed schedule).

    Each RK4 stage samples `mu` at its own sub-step time (t, t+dt/2, t+dt).
    For a scalar `mu` all three samples coincide and the result is bit-identical
    to the autonomous helper this replaced.
    """
    mu_arg, k = theta

    if callable(mu_arg):
        mu_at = mu_arg
    elif np.ndim(mu_arg) == 0:
        mu_at = lambda ti, _val=float(mu_arg): _val
    else:
        mu_seq = np.asarray(mu_arg, dtype=float)
        if len(mu_seq) != len(t):
            raise ValueError("mu array must match len(t)")
        mu_at = lambda ti, _seq=mu_seq, _t=t: _seq[min(int(round((ti - _t[0]) / (_t[1] - _t[0]))), len(_seq) - 1)]

    v = v0
    out = []
    for ti in t:
        out.append(v)
        mu_a = mu_at(ti)
        mu_b = mu_at(ti + 0.5 * dt)
        mu_c = mu_at(ti + dt)
        k1 = dvdt(v, m, mu_a, g, k)
        k2 = dvdt(v + 0.5 * dt * k1, m, mu_b, g, k)
        k3 = dvdt(v + 0.5 * dt * k2, m, mu_b, g, k)
        k4 = dvdt(v + dt * k3, m, mu_c, g, k)
        v = v + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        v = max(min(v, 100), 0)

    return np.array(out)
