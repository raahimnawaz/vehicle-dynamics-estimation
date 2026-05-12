"""Slip-aware longitudinal braking model.

We deliberately *do not* simulate full wheel rotational dynamics, because they
are an order of magnitude stiffer than the vehicle's translational dynamics and
would force a sub-millisecond integrator. A real ABS / brake controller produces
a slip profile s(t) by modulating brake pressure; we abstract that to a
caller-supplied schedule. The PINN's job is then to recover mu(s) from observed
(v, s, dv/dt) trajectories without prescribing the tire curve's functional form.

State: x = v (longitudinal velocity). External input: s(t) in [0, ~0.3].

Dynamics:
  m dv/dt = -mu(s) m g  -  k v^2
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DEFAULTS = {
    "m": 1500.0,
    "g": 9.81,
    "k": 0.4,    # lumped 0.5*rho*Cd*A so drag = k*v^2
}


def mu_pacejka(s, mu_max: float = 0.9, C: float = 20.0):
    """Ground-truth saturating tire curve: mu(s) = mu_max * (1 - e^{-C s})."""
    s = np.asarray(s)
    return mu_max * (1.0 - np.exp(-C * np.clip(s, 0.0, None)))


def dvdt(v: float, s: float, mu_fn: Callable[[float], float], p=DEFAULTS) -> float:
    v = max(v, 0.0)
    return -mu_fn(s) * p["g"] - (p["k"] / p["m"]) * v * v


def simulate(v0: float, t: np.ndarray, s_schedule: Callable[[float], float],
             mu_fn: Callable[[float], float], p=DEFAULTS) -> np.ndarray:
    """Fixed-step RK4 over uniform `t`."""
    dt = float(t[1] - t[0])
    v = np.zeros_like(t)
    v[0] = v0
    for i in range(1, len(t)):
        ti = t[i - 1]
        s_a = s_schedule(ti)
        s_b = s_schedule(ti + 0.5 * dt)
        s_c = s_schedule(ti + dt)
        k1 = dvdt(v[i - 1], s_a, mu_fn, p)
        k2 = dvdt(v[i - 1] + 0.5 * dt * k1, s_b, mu_fn, p)
        k3 = dvdt(v[i - 1] + 0.5 * dt * k2, s_b, mu_fn, p)
        k4 = dvdt(v[i - 1] + dt * k3, s_c, mu_fn, p)
        v[i] = max(v[i - 1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
    return v


def ramp_slip(s_peak: float = 0.18, ramp: float = 0.3, hold_start: float = 0.0):
    """A simple brake-controller slip schedule: ramp to peak, then hold."""
    def schedule(ti: float) -> float:
        if ti < hold_start:
            return 0.0
        u = (ti - hold_start) / max(ramp, 1e-9)
        return s_peak * min(max(u, 0.0), 1.0)
    return schedule
