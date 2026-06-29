import numpy as np
from src.physics.model import dvdt
from src.solvers.rk4 import rk4_step


def simulate(theta, v0, t, dt, m=1500, g=9.81):
    """Forward roll-out of the longitudinal braking model.

    `theta` is `(mu, k)`. `mu` may be:
      - a scalar,
      - a callable `mu(t_seconds) -> float` (for time-varying friction), or
      - a 1-D array of the same length as `t` (precomputed schedule).
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
        mu_t = mu_at(ti)
        v = rk4_step(dvdt, v, dt, m, mu_t, g, k)
        v = max(min(v, 100), 0)

    return np.array(out)
