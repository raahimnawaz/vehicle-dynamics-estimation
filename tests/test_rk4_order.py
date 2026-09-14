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


def test_simulate_is_fourth_order_in_a_time_varying_mu():
    """The order guarantee has to survive a mu that actually varies with time.

    The two tests above only exercise `rk4_step`, which is autonomous by
    construction -- they cannot see the failure this guards. `run_sim.simulate`
    accepts a callable `mu(t)`, and it used to hand that to `rk4_step`, which
    samples the right-hand side once per step at the step start. Sampling a
    time-varying input at one point per step is Euler's approximation of that
    input, so the whole integrator collapses to 1st order no matter how many
    stages it has: measured 2.00x error reduction per halving of dt instead of
    16x, a factor of 2.2e7 at dt = 0.01.

    The reference is the same scheme at a much finer step. A smooth (tanh)
    dry-to-wet transition is used rather than the discontinuous `mu_step` this
    repo ships, because a step change has no high-order reference to converge
    to -- which is also why the defect stayed invisible in the published
    scenarios.

    Two things this test has to get right, both of which quietly destroy it:

      * Every dt must be compared at the SAME instant. `np.arange(0, 4, dt)`
        stops at 3.96 / 3.98 / 3.99 for dt = 0.04 / 0.02 / 0.01, so comparing
        terminal values measures where each sweep happened to stop, not how
        accurately it integrated. Here every run is evaluated at T = 3.6, an
        exact multiple of all four step sizes.
      * The step sizes must stay coarse enough that truncation error dominates
        round-off. At dt = 0.005 the error is 2.8e-13 on a value of order 10,
        which is the double-precision floor, and the apparent order collapses
        to 4.4x per halving for reasons that have nothing to do with the
        integrator.
    """
    import math

    from src.physics.wheel import K_DRAG
    from src.simulation.run_sim import simulate

    def mu_sched(ti, t0=2.0, w=0.25, a=0.8, b=0.35):
        return a + (b - a) * 0.5 * (1.0 + math.tanh((ti - t0) / w))

    v0, T = 30.0, 3.6

    def v_at_T(dt):
        n = int(round(T / dt))
        t = np.arange(0.0, T + dt, dt)[:n + 1]
        return float(simulate([mu_sched, K_DRAG], v0, t, dt)[n])

    ref = v_at_T(2e-4)
    dts = [0.12, 0.06, 0.03]
    errs = [abs(v_at_T(dt) - ref) for dt in dts]

    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    # Measured 4.15 and 4.00. The bound only has to separate 4th order from the
    # 1st order the bug produced, so it is set well clear of both.
    for p in orders:
        assert p > 3.0, (
            f"observed convergence order {p:.2f} on a time-varying mu "
            f"(orders={orders}); ~1 means the integrator is sampling mu only "
            f"at the step start -- each RK4 stage must sample it at its own "
            f"sub-step time (t, t+dt/2, t+dt)"
        )


def test_simulate_scalar_mu_is_unchanged_by_substep_sampling():
    """A constant mu must give exactly what the autonomous helper gave.

    Sub-step sampling is only supposed to affect a mu that varies. This pins
    the published constant-mu results (sections 1, 2 and 4) against any future
    change to the integrator: all three samples coincide, so every arithmetic
    operation is the same one `rk4_step` performed, in the same order.
    """
    from src.physics.model import dvdt
    from src.physics.wheel import K_DRAG
    from src.simulation.run_sim import simulate
    from src.solvers.rk4 import rk4_step

    v0, dt = 30.0, 0.01
    t = np.arange(0, 4, dt)

    v_new = simulate([0.7, K_DRAG], v0, t, dt)

    v, expected = v0, []
    for _ in t:
        expected.append(v)
        v = rk4_step(dvdt, v, dt, 0.7, 9.81, K_DRAG)
        v = max(min(v, 100), 0)

    assert np.array_equal(v_new, np.array(expected)), "scalar-mu path drifted"
