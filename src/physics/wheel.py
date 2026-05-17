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

Two tire models are provided:
  * `mu_exponential(s, mu_max, C)` -- simple saturating curve mu_max(1-e^{-Cs}).
    Cheap, monotone, no peak. Useful as a baseline.
  * `mu_pacejka(s, B, C, D, E)`    -- Pacejka 1989 / 'Magic Formula' for
    longitudinal force. Reaches a peak around s ~ 0.1-0.15 and then *falls*
    back -- the regime real ABS controllers care about. This is the default
    ground truth.

`PACEJKA_DRY` and `PACEJKA_WET` are typical parameter sets for dry asphalt and
wet asphalt respectively (peak around mu = 0.9 / 0.55).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DEFAULTS = {
    "m": 1500.0,
    "g": 9.81,
    "k": 0.4,    # lumped 0.5*rho*Cd*A so drag = k*v^2
}

# Pacejka longitudinal-force parameter sets. B is stiffness, C is shape factor,
# D is peak coefficient (= mu_max), E controls the curvature past the peak.
# Values chosen to give a textbook-clear peak in the operational slip range so
# that the PINN's recovery is testable; real tires occupy this part of the
# parameter space (Genta 2014, "Motor Vehicle Dynamics", Tab. 2.3).
PACEJKA_DRY = {"B": 10.0, "C": 1.9, "D": 0.9,  "E": 0.5}
PACEJKA_WET = {"B": 12.0, "C": 2.0, "D": 0.55, "E": 0.6}


def mu_exponential(s, mu_max: float = 0.9, C: float = 20.0):
    """Simple saturating tire model: mu(s) = mu_max * (1 - e^{-C s}).

    No peak / fall-off -- only used as a baseline for ablations against the
    full Pacejka curve.
    """
    s = np.asarray(s)
    return mu_max * (1.0 - np.exp(-C * np.clip(s, 0.0, None)))


def mu_pacejka(s, B: float = 10.0, C: float = 1.9, D: float = 0.9, E: float = 0.97):
    """Pacejka 'Magic Formula' for longitudinal friction coefficient.

        mu(s) = D * sin( C * arctan( B s - E (B s - arctan(B s)) ) )

    The curve rises steeply, peaks near s ~ 0.10-0.15, then *falls* back
    toward the sliding-friction value -- the qualitative behaviour real ABS
    controllers exploit.
    """
    s = np.asarray(s, dtype=float)
    Bs = B * np.clip(s, 0.0, None)
    inner = Bs - E * (Bs - np.arctan(Bs))
    return D * np.sin(C * np.arctan(inner))


def pacejka_peak(B: float = 10.0, C: float = 1.9, D: float = 0.9, E: float = 0.97
                 ) -> tuple[float, float]:
    """Return `(s_peak, mu_peak)` for a Pacejka curve, by dense sampling."""
    s = np.linspace(0.0, 0.4, 4001)
    mu = mu_pacejka(s, B, C, D, E)
    i = int(np.argmax(mu))
    return float(s[i]), float(mu[i])


def dvdt(v: float, s: float, mu_fn: Callable[[float], float], p=DEFAULTS) -> float:
    v = max(v, 0.0)
    return -mu_fn(s) * p["g"] - (p["k"] / p["m"]) * v * v


def simulate(v0: float, t: np.ndarray, s_schedule: Callable[[float], float],
             mu_fn: Callable[[float], float], p=DEFAULTS) -> np.ndarray:
    """Fixed-step RK4 over uniform `t`.

    Implements RK4 inline (rather than reusing `src/solvers/rk4.py::rk4_step`)
    because the right-hand side here depends on a time-varying slip schedule
    `s_schedule(ti)`. Each RK4 stage must sample the schedule at the correct
    sub-step time (t, t+dt/2, t+dt); the generic autonomous helper would only
    sample at the step start and silently drop to 1st-order accuracy.
    """
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


def sweep_slip(s_max: float = 0.35, t_total: float = 4.0, hold_start: float = 0.2):
    """A slip schedule that sweeps linearly from 0 to s_max across the braking
    event. This is the schedule the PINN training data uses: it covers the
    entire slip range so the recovered curve is identifiable everywhere,
    rather than concentrating at a single plateau.
    """
    def schedule(ti: float) -> float:
        if ti < hold_start:
            return 0.0
        u = (ti - hold_start) / max(t_total - hold_start, 1e-9)
        return s_max * min(max(u, 0.0), 1.0)
    return schedule
