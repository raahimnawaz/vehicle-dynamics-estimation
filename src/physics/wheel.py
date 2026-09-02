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
    "k": 0.4,    # lumped 0.5*rho*Cd*A (kg/m), so drag FORCE = k*v^2
}

# THE single drag coefficient for the whole repo, in the form the acceleration
# equations actually use:
#
#     dv/dt = -mu*g - K_DRAG * v^2          [K_DRAG has units 1/m]
#
# Every model here -- the slip-aware one below, the constant-mu one in
# `model.py`, the EKF, the FrictionNet training set and the C++ port -- must
# use this value. They previously did not: callers of `model.py` hardcoded
# k = 0.02, which is 75x this value and implies ~15.7 m/s^2 of aerodynamic
# deceleration at 28 m/s -- roughly twice the entire braking force available
# from a tire on dry asphalt.
#
# Sanity check: 0.4 / 1500 = 2.67e-4 gives 0.21 m/s^2 of drag at 28 m/s,
# the right order for a passenger car (rho*Cd*A ~ 0.8 kg/m).
K_DRAG = DEFAULTS["k"] / DEFAULTS["m"]

# Pacejka longitudinal-force parameter sets. B is stiffness, C is shape factor,
# D is peak coefficient (= mu_max), E controls the curvature past the peak.
# Values chosen to give a textbook-clear peak in the operational slip range so
# that the PINN's recovery is testable; real tires occupy this part of the
# parameter space (Genta 2014, "Motor Vehicle Dynamics", Tab. 2.3).
PACEJKA_DRY = {"B": 10.0, "C": 1.9, "D": 0.9,  "E": 0.5}
PACEJKA_WET = {"B": 12.0, "C": 2.0, "D": 0.55, "E": 0.6}

# The keyword defaults of `mu_pacejka`, `pacejka_peak` and `mu_combined` below
# are taken from PACEJKA_DRY rather than written out again, for the same reason
# K_DRAG exists: a second copy of a physical constant is a second chance to get
# it wrong. They previously read `E = 0.97` against PACEJKA_DRY's 0.5, so any
# caller who omitted the keyword silently got a different tire than the one the
# ground truth is generated from -- a curve whose peak sits at s = 0.180 instead
# of 0.127, off by 42% in the one quantity this repo is about, and differing by
# up to 0.133 in mu where the headline recovery error is 0.013. Every call site
# passed `**PACEJKA_DRY` explicitly, so nothing published was ever wrong; the
# defaults were a loaded gun pointed at the next caller.
# `tests/test_pinn.py::test_pacejka_defaults_match_the_ground_truth_set` fails
# if these ever drift apart again.


def mu_exponential(s, mu_max: float = 0.9, C: float = 20.0):
    """Simple saturating tire model: mu(s) = mu_max * (1 - e^{-C s}).

    No peak / fall-off -- only used as a baseline for ablations against the
    full Pacejka curve.
    """
    s = np.asarray(s)
    return mu_max * (1.0 - np.exp(-C * np.clip(s, 0.0, None)))


def mu_pacejka(s, B: float = PACEJKA_DRY["B"], C: float = PACEJKA_DRY["C"],
               D: float = PACEJKA_DRY["D"], E: float = PACEJKA_DRY["E"]):
    """Pacejka 'Magic Formula' for longitudinal friction coefficient.

        mu(s) = D * sin( C * arctan( B s - E (B s - arctan(B s)) ) )

    The curve rises steeply, peaks near s ~ 0.10-0.15, then *falls* back
    toward the sliding-friction value -- the qualitative behaviour real ABS
    controllers exploit.

    Defaults are PACEJKA_DRY, so `mu_pacejka(s)` and `mu_pacejka(s,
    **PACEJKA_DRY)` are the same curve. See the note above PACEJKA_DRY.
    """
    s = np.asarray(s, dtype=float)
    Bs = B * np.clip(s, 0.0, None)
    inner = Bs - E * (Bs - np.arctan(Bs))
    return D * np.sin(C * np.arctan(inner))


def pacejka_peak(B: float = PACEJKA_DRY["B"], C: float = PACEJKA_DRY["C"],
                 D: float = PACEJKA_DRY["D"], E: float = PACEJKA_DRY["E"]
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


# ---------------------------------------------------------------------------
# Combined slip: the friction ellipse
# ---------------------------------------------------------------------------
#
# Everything above treats the tire as though its entire friction budget were
# available longitudinally. It is not. A tire has one contact patch and one
# limit; force spent turning is not available for stopping. The standard
# lumped model is the friction ellipse,
#
#     (F_x / (mu_x N))^2 + (F_y / (mu_y N))^2 <= 1,
#
# which with mu_x = mu_y is a circle. Writing n for the fraction of the budget
# already spent laterally, n = a_y / (g * D), the longitudinal coefficient
# that remains is mu(s) * sqrt(1 - n^2).
#
# This is a lumped single-track approximation, not a combined-slip Pacejka
# model: it derates the whole mu(s) curve by one scalar rather than solving
# for the (kappa, alpha) force surface with the G_x-alpha / G_y-kappa
# weighting functions of Pacejka (2002) sec. 4.3.2. It also assumes lateral
# demand is an exogenous input rather than a state, which is what keeps the
# model 1-DOF -- see the Roadmap in the README for what dropping each of those
# assumptions costs.


def lateral_utilisation(a_y, D: float = 0.9, g: float = None):
    """Fraction of the friction budget spent laterally, n = a_y / (g * D).

    `a_y` is lateral acceleration in m/s^2, `D` the tire's peak friction
    coefficient. n = 1 means the tire is at its limit in pure cornering and
    has no longitudinal capacity left.
    """
    g = DEFAULTS["g"] if g is None else g
    return np.abs(np.asarray(a_y, dtype=float)) / max(g * D, 1e-9)


def friction_ellipse(n_lat):
    """Longitudinal derating factor sqrt(1 - n^2) from the friction ellipse.

    Exactly 1.0 at n = 0 (straight-line braking, full budget available) and
    0.0 at n = 1 (tire saturated laterally). Clipped outside [0, 1] so an
    over-demanded corner returns zero longitudinal capacity rather than a NaN.
    """
    n = np.clip(np.asarray(n_lat, dtype=float), 0.0, 1.0)
    return np.sqrt(np.clip(1.0 - n * n, 0.0, None))


def mu_combined(s, n_lat, B: float = PACEJKA_DRY["B"], C: float = PACEJKA_DRY["C"],
                D: float = PACEJKA_DRY["D"], E: float = PACEJKA_DRY["E"]):
    """Longitudinal friction under combined slip: mu(s) * sqrt(1 - n^2).

    The factorisation is the whole point for the 2D PINN in `src/ml/pinn.py`:
    the network is given (s, n) and has to split the product back into a slip
    curve and a derating factor without being told which is which.
    """
    return mu_pacejka(s, B, C, D, E) * friction_ellipse(n_lat)


def corner_demand(n_max: float = 0.6, ramp: float = 1.2, hold_start: float = 0.4):
    """Lateral-demand schedule: hold straight, ramp into a corner, then hold.

    Given its own `hold_start` and `ramp` rather than reusing the slip
    schedule's, so that lateral demand is not an exact function of slip. That
    exact case is the one that defeats the factorisation in `mu_combined`:
    with n rigidly tied to s the network can trade any amount of one factor
    against the other and still fit the data. Merely *correlated* inputs are
    fine -- see `src/ml/pinn.py::generate_dataset_combined` for the measured
    difference between the two.
    """
    def schedule(ti: float) -> float:
        if ti < hold_start:
            return 0.0
        u = (ti - hold_start) / max(ramp, 1e-9)
        return n_max * min(max(u, 0.0), 1.0)
    return schedule
