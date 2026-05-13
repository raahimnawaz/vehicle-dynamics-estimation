"""Model-mismatch study (Option B).

Generates ground-truth braking trajectories from the *full* slip-aware model
augmented with one of three commonly-unmodeled effects, then fits each
estimator using its native (and often incorrect) model. The point is not to
crown a winner -- it is to *map* which method degrades under which mismatch,
which is the question a real robotics PI is asking when picking a stack.

The three effects:

  - **road grade** theta:    -m g sin(theta) is added to the longitudinal force,
                             which the constant-mu fitters absorb into mu.
  - **headwind** v_w:        drag becomes (1/2) rho Cd A (v + v_w)^2, which
                             biases the drag coefficient k.
  - **brake-force ramp**:    effective mu is multiplied by (1 - exp(-t/tau)),
                             so the initial portion of the trace looks like a
                             low-mu surface, then transitions to nominal.

Methods compared:
  - **Batch** (SciPy Nelder-Mead, simple constant-mu model)
  - **EKF**   (online, simple constant-mu model)
  - **NN**    (FrictionNet -- pretrained, single forward pass)
  - **PINN**  (mu_theta(s), pretrained; uses the slip-aware Pacejka model)
  - **PINN-B** (mu_theta(s)*ramp(p), pretrained on a brake-aware dataset)

Metric: trajectory RMSE between each method's *predicted* velocity and the
clean ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.estimation.kalman import VehicleEKF
from src.estimation.optimize import estimate
from src.ml.pinn import MuNet, MuNet2D
from src.ml.train import FrictionNet
from src.physics.wheel import DEFAULTS, PACEJKA_DRY, mu_pacejka, ramp_slip
from src.simulation.run_sim import simulate as simple_simulate


# ---------------------------------------------------------------------------
# Augmented forward model
# ---------------------------------------------------------------------------

@dataclass
class Effect:
    """One unmodeled effect, parameterised by a scalar `intensity`."""
    name: str
    label: str
    intensities: tuple[float, ...]
    unit: str


EFFECTS = (
    Effect("grade",    "road grade (rad)",  (0.0, 0.02, 0.05, 0.08, 0.12),       "rad"),
    Effect("headwind", "headwind (m/s)",     (0.0, 3.0, 6.0, 10.0, 15.0),         "m/s"),
    Effect("brake",    "brake ramp tau (s)", (0.01, 0.15, 0.3, 0.5, 0.8),         "s"),
)


def simulate_truth(v0: float, t: np.ndarray, *, effect: str, intensity: float,
                   s_schedule: Callable[[float], float] | None = None,
                   pacejka: dict | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slip-aware forward roll with one unmodeled effect active.

    Returns `(v_truth_clean, s_traj, p_traj)`. The simple-model fitters never
    see s_traj/p_traj, only noisy v. The PINN sees s_traj; the brake-aware
    PINN also sees p_traj (normalised brake pressure).
    """
    if s_schedule is None:
        s_schedule = ramp_slip(s_peak=0.16, ramp=0.25, hold_start=0.0)
    if pacejka is None:
        pacejka = PACEJKA_DRY

    dt = float(t[1] - t[0])
    g = DEFAULTS["g"]
    m = DEFAULTS["m"]
    k = DEFAULTS["k"]   # lumped 0.5 * rho * Cd * A

    grade = intensity if effect == "grade" else 0.0
    v_w   = intensity if effect == "headwind" else 0.0
    tau   = intensity if effect == "brake" else 1e-4   # ~instantaneous

    v = np.zeros_like(t)
    v[0] = v0
    s_traj = np.zeros_like(t)
    p_traj = np.zeros_like(t)
    for i in range(1, len(t)):
        ti = t[i - 1]
        s_traj[i - 1] = s_schedule(ti)
        p_traj[i - 1] = 1.0 - np.exp(-ti / max(tau, 1e-9))

        def rhs(vi: float, tt: float) -> float:
            si = s_schedule(tt)
            mu = mu_pacejka(si, **pacejka)
            ramp = 1.0 - np.exp(-tt / max(tau, 1e-9))
            F_fric_per_m = mu * g * np.cos(grade) * ramp
            F_drag_per_m = (k / m) * (vi + v_w) ** 2
            F_grav_per_m = g * np.sin(grade)
            return -F_fric_per_m - F_drag_per_m - F_grav_per_m

        k1 = rhs(v[i - 1],                  ti)
        k2 = rhs(v[i - 1] + 0.5 * dt * k1,  ti + 0.5 * dt)
        k3 = rhs(v[i - 1] + 0.5 * dt * k2,  ti + 0.5 * dt)
        k4 = rhs(v[i - 1] + dt * k3,        ti + dt)
        v[i] = max(v[i - 1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
    s_traj[-1] = s_schedule(t[-1])
    p_traj[-1] = 1.0 - np.exp(-t[-1] / max(tau, 1e-9))
    return v, s_traj, p_traj


# ---------------------------------------------------------------------------
# Estimators -- each takes (t, v_obs, s_traj, p_traj) and returns
# (label, v_predicted, mu_effective_scalar)
# ---------------------------------------------------------------------------

def fit_batch(t, v_obs, s_traj, p_traj):
    dt = float(t[1] - t[0])
    v0 = float(v_obs[0])
    mu_est, k_est = estimate(v_obs, v0, t, dt)
    v_pred = simple_simulate([float(mu_est), float(k_est)], v0, t, dt)
    return ("Batch (SciPy)", v_pred, float(mu_est))


def fit_ekf(t, v_obs, s_traj, p_traj):
    dt = float(t[1] - t[0])
    v0 = float(v_obs[0])
    ekf = VehicleEKF(mu_init=0.5, v_init=v0)
    ekf.Q[1, 1] = 1e-2
    for vi in v_obs:
        if vi > 1.0:
            ekf.predict(dt)
            ekf.update(vi)
    mu_est = float(ekf.x[1])
    v_pred = simple_simulate([mu_est, 0.02], v0, t, dt)
    return ("EKF", v_pred, mu_est)


def fit_nn(t, v_obs, s_traj, p_traj, weights_path: str = "models/friction_net.pth"):
    dt = float(t[1] - t[0])
    v0 = float(v_obs[0])
    window_size = 50
    net = FrictionNet(window_size)
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    net.eval()
    start = min(10, len(v_obs) - window_size - 1)
    win = torch.tensor(v_obs[start:start + window_size], dtype=torch.float32)
    with torch.no_grad():
        mu_est = float(net(win).item())
    v_pred = simple_simulate([mu_est, 0.02], v0, t, dt)
    return ("NN (FrictionNet)", v_pred, mu_est)


def fit_pinn(t, v_obs, s_traj, p_traj, weights_path: str = "models/pinn_mu.pth"):
    """1D PINN: mu_theta(s). Forward-rolls slip-aware model (no brake ramp)."""
    dt = float(t[1] - t[0])
    v0 = float(v_obs[0])
    net = MuNet()
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    net.eval()

    g = DEFAULTS["g"]
    k_over_m = DEFAULTS["k"] / DEFAULTS["m"]
    s_tensor = torch.tensor(s_traj, dtype=torch.float32)
    with torch.no_grad():
        mu_traj = net(s_tensor).numpy().astype(np.float64)

    v_pred = np.zeros_like(t)
    v_pred[0] = v0
    for i in range(1, len(t)):
        a = -mu_traj[i - 1] * g - k_over_m * v_pred[i - 1] ** 2
        v_pred[i] = max(v_pred[i - 1] + a * dt, 0.0)
    mu_eff = float(np.average(mu_traj[s_traj > 1e-4])) if np.any(s_traj > 1e-4) else 0.0
    return ("PINN", v_pred, mu_eff)


def fit_pinn_brake(t, v_obs, s_traj, p_traj,
                   weights_path: str = "models/pinn_mu_2d.pth"):
    """2D PINN: mu_theta(s) * ramp_theta(p). Forward-rolls with both inputs."""
    dt = float(t[1] - t[0])
    v0 = float(v_obs[0])
    net = MuNet2D()
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    net.eval()

    g = DEFAULTS["g"]
    k_over_m = DEFAULTS["k"] / DEFAULTS["m"]
    s_tensor = torch.tensor(s_traj, dtype=torch.float32)
    p_tensor = torch.tensor(p_traj, dtype=torch.float32)
    with torch.no_grad():
        mu_eff_traj = net(s_tensor, p_tensor).numpy().astype(np.float64)

    v_pred = np.zeros_like(t)
    v_pred[0] = v0
    for i in range(1, len(t)):
        a = -mu_eff_traj[i - 1] * g - k_over_m * v_pred[i - 1] ** 2
        v_pred[i] = max(v_pred[i - 1] + a * dt, 0.0)
    active = s_traj > 1e-4
    mu_eff = float(np.average(mu_eff_traj[active])) if np.any(active) else 0.0
    return ("PINN-B (brake-aware)", v_pred, mu_eff)


METHODS = (fit_batch, fit_ekf, fit_nn, fit_pinn, fit_pinn_brake)


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

@dataclass
class SweepCell:
    method: str
    effect: str
    intensity: float
    mu_est: float
    rmse: float
    rmse_norm: float           # rmse / v0  -- comparable across runs
    v_truth: np.ndarray
    v_obs: np.ndarray
    v_pred: np.ndarray
    t: np.ndarray


def run_sweep(*, dt: float = 0.01, t_final: float = 3.5, v0: float = 28.0,
              noise: float = 0.25, seed: int = 0,
              methods=METHODS) -> list[SweepCell]:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_final, dt)
    cells: list[SweepCell] = []

    for effect in EFFECTS:
        for intensity in effect.intensities:
            v_truth, s_traj, p_traj = simulate_truth(
                v0, t, effect=effect.name, intensity=intensity
            )
            v_obs = v_truth + rng.normal(0.0, noise, size=len(t))
            for method in methods:
                try:
                    label, v_pred, mu_est = method(t, v_obs, s_traj, p_traj)
                except FileNotFoundError:
                    continue
                rmse = float(np.sqrt(np.mean((v_pred - v_truth) ** 2)))
                cells.append(SweepCell(
                    method=label, effect=effect.name, intensity=float(intensity),
                    mu_est=float(mu_est), rmse=rmse, rmse_norm=rmse / v0,
                    v_truth=v_truth, v_obs=v_obs, v_pred=v_pred, t=t,
                ))
    return cells
