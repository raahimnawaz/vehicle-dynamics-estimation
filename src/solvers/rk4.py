"""Generic 4th-order Runge-Kutta integrator for autonomous ODEs.

Use this for problems where the right-hand side has no explicit time
dependence (or the time dependence is folded into a state variable). The
function signature is:

    f(v, *args) -> dv/dt

`*args` are passed through unchanged on every stage, so use them for
constants (mass, gravity, friction, etc.).

For *non-autonomous* problems where the right-hand side depends on a
time-varying external input (e.g. a slip schedule s(t) in the braking
model), use the inline RK4 in `src/physics/wheel.py::simulate` instead —
it samples the schedule at the correct intermediate times (t, t+dt/2, t+dt)
so the integrator stays 4th order. Using this generic helper with a
schedule that only evaluates at the step start silently degrades to
1st-order accuracy.

The 4th-order convergence rate is verified in `tests/test_rk4_order.py`
against the closed-form solution of dv/dt = -lambda v.
"""

from __future__ import annotations


def rk4_step(f, v, dt, *args):
    k1 = f(v, *args)
    k2 = f(v + 0.5 * dt * k1, *args)
    k3 = f(v + 0.5 * dt * k2, *args)
    k4 = f(v + dt * k3, *args)
    return v + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
