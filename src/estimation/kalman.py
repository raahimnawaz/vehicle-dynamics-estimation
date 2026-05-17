"""Extended Kalman Filter for joint (velocity, friction) estimation.

State: x = [v, mu]^T
Measurement: z = v + noise  (we observe velocity only, so H = [1, 0])

The mu component has no dynamics (mu_{k+1} = mu_k); it adapts purely through
the measurement-update step driven by the process-noise covariance Q[1,1].

Defaults are tuned for the synthetic benchmark in `reproduce.py::run_synthetic`,
which has constant ground-truth mu and clean velocity samples. For other
scenarios you almost certainly need to override:

  * Step-change scenarios (e.g. dry -> wet at t=2s):
        ekf = VehicleEKF(...); ekf.Q[1, 1] = 1e-2
    The default q_mu = 1e-4 is intentionally cold (keeps the estimate steady
    when mu is constant). It's too cold to track abrupt regime changes.

  * Real telemetry (noisier than the synthetic baseline):
        ekf.R[0, 0] = 0.25  # or whatever matches your sensor noise std^2
    The default R = 1.0 corresponds to a sensor std of ~1 m/s, which is
    realistic for raw GPS but overestimates wheel-speed or fused INS.

These overrides are made explicitly at the call sites in
`src/scenarios/runner.py` and `src/scenarios/mismatch.py`; see those for the
recommended values per scenario.
"""

from __future__ import annotations

import numpy as np


class VehicleEKF:
    def __init__(self, mu_init: float = 0.3, v_init: float = 30.0) -> None:
        self.x = np.array([v_init, mu_init], dtype=float)

        # Initial covariance: 1 m^2/s^2 on velocity, dimensionless 1 on mu.
        self.P = np.array([[1.0, 0.0],
                           [0.0, 1.0]])

        # Process noise: q_mu=1e-4 is COLD — see module docstring for when
        # to bump this (step changes need ~1e-2).
        self.Q = np.array([[0.1, 0.0],
                           [0.0, 1e-4]])

        # Measurement noise variance on velocity (m^2/s^2). Default ~1 m/s
        # std is realistic for raw GPS; override for cleaner sensors.
        self.R = np.array([[1.0]])

        # Observation matrix: we measure velocity only.
        self.H = np.array([[1.0, 0.0]])

        # Vehicle parameters baked into the EKF dynamics. These match the
        # Python forward-model in `src/physics/wheel.py`; the C++ port in
        # `cpp/include/vd/ekf.hpp` uses an equivalent parameterisation with
        # k_cpp = rho_cd_a / (2 * m), reconciled by tools/parity_check.py.
        self.g = 9.81
        self.m = 1500.0
        self.rho_cd_a = 0.02 * 2 * self.m

    def predict(self, dt: float) -> None:
        v = self.x[0]
        mu = self.x[1]

        a = -mu * self.g - (self.rho_cd_a / (2 * self.m)) * (v ** 2)
        v_next = v + a * dt

        self.x = np.array([v_next, mu])

        # Jacobian of the dynamics wrt (v, mu)
        df_dv = 1.0 - dt * (self.rho_cd_a / self.m) * v
        df_dmu = -self.g * dt

        F = np.array([[df_dv, df_dmu],
                      [0.0,   1.0]])

        self.P = F @ self.P @ F.T + self.Q

    def update(self, v_obs: float) -> None:
        # H = [1, 0] makes S a 1x1 scalar (P[0,0] + R[0,0]); no matrix
        # inverse needed. C++ port specialises the same way.
        S = float(self.P[0, 0] + self.R[0, 0])
        K0 = self.P[0, 0] / S
        K1 = self.P[1, 0] / S

        innov = v_obs - self.x[0]
        self.x[0] += K0 * innov
        self.x[1] += K1 * innov

        # P = (I - K H) P, expanded for H = [1, 0]
        p00, p01 = self.P[0, 0], self.P[0, 1]
        p10, p11 = self.P[1, 0], self.P[1, 1]
        self.P[0, 0] = (1.0 - K0) * p00
        self.P[0, 1] = (1.0 - K0) * p01
        self.P[1, 0] = -K1 * p00 + p10
        self.P[1, 1] = -K1 * p01 + p11
