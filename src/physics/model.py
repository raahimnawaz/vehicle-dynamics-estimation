"""Minimal constant-mu longitudinal model used by the simple-model batch
estimator and EKF.

This is the *constant-friction* dynamics:

    dv/dt = -mu * g - k * v^2

where mu is a scalar. The slip-aware Pacejka model (where mu is a function of
slip ratio s) lives in `src/physics/wheel.py` and has a different signature:

    wheel.dvdt(v, s, mu_fn, p)     # slip-aware, callable mu, params dict
    model.dvdt(v, m, mu, g, k)     # constant-mu scalar, positional args

The two are deliberately distinct because the constant-mu fitters (`Batch`,
`EKF`, `FrictionNet`) operate on this simpler model — the whole point of the
model-mismatch study in `src/scenarios/mismatch.py` is to see what happens
when those constant-mu estimators meet the slip-aware Pacejka truth.

`src/simulation/run_sim.py::simulate` builds on this version. The wheel-physics
forward roll uses the inline RK4 in `wheel.simulate` because it needs to
sample a time-varying slip schedule mid-step.
"""

from __future__ import annotations


def dvdt(v: float, m: float, mu: float, g: float, k: float) -> float:
    """Constant-mu longitudinal braking dynamics. See module docstring."""
    v = max(v, 0.0)
    return -mu * g - k * v ** 2
