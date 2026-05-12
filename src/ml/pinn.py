"""Physics-Informed Neural Network that recovers the tire friction curve mu(s)
from braking trajectories WITHOUT prescribing the curve's functional form.

The network is a small MLP mu_theta : [0, 0.3] -> R. It is trained against the
ODE residual of the longitudinal vehicle equation at every collocation point
along observed trajectories:

    residual_i = dv_obs/dt|_i  -  ( - mu_theta(s_i) * g  -  (k/m) v_i^2 )

A monotonicity prior keeps mu non-decreasing on the operating range. No
assumption is made about the curve's functional form — the saturating Pacejka
shape (or whatever curve produced the data) is recovered, not imposed.

Pipeline:
  generate_dataset(...) : roll forward the ground truth + add sensor noise
  MuNet                 : the network
  pinn_loss             : residual + monotonicity
  train_pinn            : Adam loop
  evaluate_curve        : sample mu_theta on a grid for plotting
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.physics.wheel import DEFAULTS, mu_pacejka, ramp_slip, simulate


@dataclass
class Dataset:
    v: np.ndarray
    s: np.ndarray
    dv_dt: np.ndarray
    run_id: np.ndarray


class MuNet(nn.Module):
    """Small MLP s -> mu. 1.2*sigmoid keeps mu within a sane physical range."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = s.view(-1, 1)
        return 1.2 * torch.sigmoid(self.net(x)).squeeze(-1)


def _smooth(arr: np.ndarray, w: int = 7) -> np.ndarray:
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def generate_dataset(
    n_runs: int = 8,
    t_final: float = 4.0,
    dt: float = 0.01,
    noise_v: float = 0.15,
    seed: int = 0,
    mu_max: float = 0.9,
    C: float = 20.0,
) -> tuple[Dataset, dict]:
    """Roll the system out under several brake-controller slip schedules."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_final, dt)

    def mu_true(s):
        return mu_pacejka(s, mu_max, C)

    vs, ss, dvs, rids = [], [], [], []
    for run in range(n_runs):
        v0 = float(rng.uniform(22.0, 32.0))
        s_peak = float(rng.uniform(0.08, 0.25))
        ramp = float(rng.uniform(0.15, 0.5))
        hold_start = float(rng.uniform(0.0, 0.3))
        sched = ramp_slip(s_peak=s_peak, ramp=ramp, hold_start=hold_start)

        v_clean = simulate(v0, t, sched, mu_true)
        v_noisy = v_clean + rng.normal(0.0, noise_v, size=len(t))
        s_t = np.array([sched(ti) for ti in t])

        # finite-difference derivative on the smoothed observation
        v_s = _smooth(v_noisy, w=9)
        dv_dt = np.gradient(v_s, dt)

        # Drop samples where the car is below the noise floor or where the
        # finite-difference derivative is dominated by run-edge artefacts.
        # |dv/dt| < 12 m/s^2 covers the entire physically plausible range
        # (mu_max ~1.0 + drag < 12), so anything outside is noise.
        mask = (v_s > 8.0) & (np.abs(dv_dt) < 12.0)
        vs.append(v_s[mask])
        ss.append(s_t[mask])
        dvs.append(dv_dt[mask])
        rids.append(np.full(int(mask.sum()), run, dtype=int))

    ds = Dataset(
        v=np.concatenate(vs),
        s=np.concatenate(ss),
        dv_dt=np.concatenate(dvs),
        run_id=np.concatenate(rids),
    )
    meta = {
        "mu_max": mu_max,
        "C": C,
        "n_samples": int(len(ds.v)),
        "s_range": (float(ds.s.min()), float(ds.s.max())),
    }
    return ds, meta


def pinn_loss(net: MuNet, ds: Dataset, lam_mono: float = 0.5) -> tuple[torch.Tensor, dict]:
    p = DEFAULTS
    s = torch.tensor(ds.s, dtype=torch.float32)
    v = torch.tensor(ds.v, dtype=torch.float32)
    dv = torch.tensor(ds.dv_dt, dtype=torch.float32)

    mu = net(s)
    dv_pred = -mu * p["g"] - (p["k"] / p["m"]) * v * v
    res = dv - dv_pred
    loss_data = torch.mean(res ** 2)

    # monotonicity prior on a dense grid: d mu / d s >= 0
    s_grid = torch.linspace(0.0, 0.3, 80, requires_grad=True)
    mu_grid = net(s_grid)
    dmu_ds = torch.autograd.grad(mu_grid.sum(), s_grid, create_graph=True)[0]
    loss_mono = torch.mean(torch.relu(-dmu_ds) ** 2)

    return loss_data + lam_mono * loss_mono, {
        "data": float(loss_data.item()),
        "mono": float(loss_mono.item()),
    }


def train_pinn(ds: Dataset, *, epochs: int = 4000, lr: float = 5e-3,
               seed: int = 0, verbose: bool = False) -> tuple[MuNet, list[float]]:
    torch.manual_seed(seed)
    net = MuNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history: list[float] = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss, parts = pinn_loss(net, ds)
        loss.backward()
        opt.step()
        if epoch % 200 == 0 or epoch == epochs - 1:
            history.append(float(loss.item()))
            if verbose:
                print(f"  epoch {epoch:4d}  total={loss.item():.4f}  data={parts['data']:.4f}  mono={parts['mono']:.4f}")
    return net, history


def evaluate_curve(net: MuNet, n: int = 200, s_max: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, s_max, n)
    with torch.no_grad():
        mu = net(torch.tensor(s, dtype=torch.float32)).numpy()
    return s, mu
