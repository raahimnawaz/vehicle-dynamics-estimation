"""With friction and drag disabled (mu = k = 0), kinetic energy is conserved.

The forward model is `dv/dt = -mu*g - k*v^2`, so zeroing both coefficients
should yield a constant velocity. Any drift exposes a bug in the integrator,
the time loop, or the physics function.
"""

import numpy as np

from src.simulation.run_sim import simulate


def test_velocity_constant_when_no_forces():
    v0 = 30.0
    dt = 0.01
    t = np.arange(0, 5, dt)

    v = simulate([0.0, 0.0], v0, t, dt)

    assert v.shape == t.shape
    assert np.allclose(v, v0, atol=1e-9), f"max drift = {np.max(np.abs(v - v0))}"


def test_velocity_monotone_decreasing_with_friction():
    v0 = 30.0
    dt = 0.01
    t = np.arange(0, 3, dt)
    v = simulate([0.7, 0.02], v0, t, dt)
    diffs = np.diff(v)
    # speed never increases while above the v=0 clamp
    assert np.all(diffs <= 1e-9), "velocity should monotonically decrease under braking"
