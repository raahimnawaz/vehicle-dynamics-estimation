"""Physics-Informed Networks for tire-friction system identification.

Three networks are exposed, illustrating a spectrum from "function-free" to
"grey-box parametric":

  * `MuNet`       : s -> mu, free-form MLP with a concavity prior.
                    Recovers Pacejka shape (rise, peak, fall) without being
                    told the functional form.
  * `MuNet2D`     : (s, p) -> mu, factorised mu(s) * ramp(p).
                    Required when brake-force lag (time-varying mu_eff) is
                    in play -- adds normalised brake pressure as a 2nd input.
  * `PacejkaNet`  : grey-box. Four learnable scalars (B, C, D, E) that are
                    plugged into the analytic Pacejka magic formula. Same
                    ODE-residual loss; no free-form MLP. Industry-standard
                    parameter ID -- the cleanest recovery the data allows.

The structural-prior story is the point of having both `MuNet` and
`PacejkaNet` in the repo:

  * Symmetric smoothness (`mean d2mu^2`) penalises curvature in BOTH
    directions, which actively discourages the post-peak peak-and-fall shape.
    That was the original failure mode.
  * The correct shape prior for a single-arch tire curve is *concavity*:
    `mean relu(d2mu/ds2)^2`. It penalises only POSITIVE curvature, allowing
    the network to form a peak and then descend.
  * Layering `PacejkaNet` alongside `MuNet` makes the cost of "function-free"
    explicit: MuNet must rediscover the shape; PacejkaNet only fits 4 scalars
    and recovers the curve almost exactly.

Pipeline:
  generate_dataset(...)         : roll forward Pacejka truth + sensor noise
  generate_dataset_braking(...) : same, with a brake-ramp time constant tau
  MuNet / MuNet2D / PacejkaNet  : the networks
  pinn_loss / pinn_loss_2d /
      pacejka_loss              : residual + structural priors
  train_pinn / train_pinn_2d /
      train_pacejka             : Adam loops
  evaluate_curve(_2d)           : sample on a grid for plotting
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.physics.wheel import (
    DEFAULTS,
    PACEJKA_DRY,
    mu_pacejka,
    ramp_slip,
    simulate,
    sweep_slip,
)


@dataclass
class Dataset:
    v: np.ndarray
    s: np.ndarray
    dv_dt: np.ndarray
    run_id: np.ndarray


@dataclass
class BrakeDataset:
    v: np.ndarray
    s: np.ndarray
    p: np.ndarray            # normalised brake pressure in [0, 1]
    dv_dt: np.ndarray
    run_id: np.ndarray


class MuNet(nn.Module):
    """Small MLP s -> mu. 1.1*sigmoid caps mu just above 1.0 with headroom so
    the network can both reach and fall back from the Pacejka peak (~0.9) --
    a clean sigmoid at exactly 1.0 saturates the gradient and the network
    cannot represent the post-peak fall-off."""

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
        return 1.1 * torch.sigmoid(self.net(x)).squeeze(-1)


class MuNet2D(nn.Module):
    """mu_eff(s, p) = mu(s) * ramp(p). Two heads, each a small MLP."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.ramp_head = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def mu(self, s: torch.Tensor) -> torch.Tensor:
        x = s.view(-1, 1)
        return 1.2 * torch.sigmoid(self.mu_head(x)).squeeze(-1)

    def ramp(self, p: torch.Tensor) -> torch.Tensor:
        x = p.view(-1, 1)
        return torch.sigmoid(self.ramp_head(x)).squeeze(-1)

    def forward(self, s: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.mu(s) * self.ramp(p)


def _smooth(arr: np.ndarray, w: int = 7) -> np.ndarray:
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def generate_dataset(
    n_runs: int = 8,
    t_final: float = 4.0,
    dt: float = 0.01,
    noise_v: float = 0.15,
    seed: int = 0,
    pacejka: dict | None = None,
) -> tuple[Dataset, dict]:
    """Roll the system out under several brake-controller slip schedules.

    Ground truth is the Pacejka 'Magic Formula'; the PINN must recover the
    peak and post-peak fall-off without being told the functional form.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_final, dt)
    pj = pacejka or PACEJKA_DRY

    def mu_true(s):
        return mu_pacejka(s, **pj)

    vs, ss, dvs, rids = [], [], [], []
    for run in range(n_runs):
        v0 = float(rng.uniform(22.0, 32.0))
        # Use a sweep schedule (slip ramps from 0 to s_max across the whole
        # braking event) instead of a hold schedule. This is the key to making
        # the recovery problem well-posed: the data covers EVERY slip value in
        # [0, s_max] uniformly, so the post-peak fall-off is identifiable
        # rather than buried inside a single plateau.
        s_max = float(rng.uniform(0.20, 0.35))
        hold_start = float(rng.uniform(0.0, 0.3))
        sched = sweep_slip(s_max=s_max, t_total=t_final, hold_start=hold_start)

        v_clean = simulate(v0, t, sched, mu_true)
        v_noisy = v_clean + rng.normal(0.0, noise_v, size=len(t))
        s_t = np.array([sched(ti) for ti in t])

        v_s = _smooth(v_noisy, w=9)
        dv_dt = np.gradient(v_s, dt)
        # |dv/dt| < 12 m/s^2 covers everything physically plausible. Velocity
        # filter at 4 m/s keeps us above the noise floor while preserving
        # late-braking samples where slip is highest.
        mask = (v_s > 4.0) & (np.abs(dv_dt) < 12.0)
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
        "model": "pacejka",
        "params": pj,
        "n_samples": int(len(ds.v)),
        "s_range": (float(ds.s.min()), float(ds.s.max())),
    }
    return ds, meta


def generate_dataset_braking(
    n_runs: int = 12,
    t_final: float = 4.0,
    dt: float = 0.01,
    noise_v: float = 0.15,
    seed: int = 0,
    pacejka: dict | None = None,
    tau_range: tuple[float, float] = (0.05, 0.7),
) -> tuple[BrakeDataset, dict]:
    """Like generate_dataset but each run has a random brake-ramp time constant.

    The effective friction at time t is mu(s) * (1 - exp(-t/tau)). The 2D PINN
    has to factorise the two effects without being told which dimension is which.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_final, dt)
    pj = pacejka or PACEJKA_DRY
    g = DEFAULTS["g"]
    k_over_m = DEFAULTS["k"] / DEFAULTS["m"]

    vs, ss, ps, dvs, rids = [], [], [], [], []
    for run in range(n_runs):
        v0 = float(rng.uniform(22.0, 32.0))
        s_max = float(rng.uniform(0.20, 0.35))
        hold_start = float(rng.uniform(0.0, 0.3))
        tau = float(rng.uniform(*tau_range))

        sched = sweep_slip(s_max=s_max, t_total=t_final, hold_start=hold_start)

        # Forward-roll RK4 with mu_eff = mu(s) * (1 - exp(-t/tau)).
        v = np.zeros_like(t)
        v[0] = v0
        s_t = np.array([sched(ti) for ti in t])
        p_t = 1.0 - np.exp(-t / max(tau, 1e-4))   # brake pressure in [0,1]
        mu_s = mu_pacejka(s_t, **pj)
        mu_eff = mu_s * p_t
        for i in range(1, len(t)):
            # midpoint sampling for RK4
            def rhs(vi, idx):
                return -mu_eff[idx] * g - k_over_m * vi ** 2
            k1 = rhs(v[i - 1], i - 1)
            k2 = rhs(v[i - 1] + 0.5 * dt * k1, i - 1)
            k3 = rhs(v[i - 1] + 0.5 * dt * k2, i - 1)
            k4 = rhs(v[i - 1] + dt * k3, i)
            v[i] = max(v[i - 1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)

        v_noisy = v + rng.normal(0.0, noise_v, size=len(t))
        v_s = _smooth(v_noisy, w=9)
        dv_dt = np.gradient(v_s, dt)

        mask = (v_s > 8.0) & (np.abs(dv_dt) < 12.0)
        vs.append(v_s[mask])
        ss.append(s_t[mask])
        ps.append(p_t[mask])
        dvs.append(dv_dt[mask])
        rids.append(np.full(int(mask.sum()), run, dtype=int))

    ds = BrakeDataset(
        v=np.concatenate(vs),
        s=np.concatenate(ss),
        p=np.concatenate(ps),
        dv_dt=np.concatenate(dvs),
        run_id=np.concatenate(rids),
    )
    meta = {
        "model": "pacejka+brake_ramp",
        "params": pj,
        "n_samples": int(len(ds.v)),
        "s_range": (float(ds.s.min()), float(ds.s.max())),
        "p_range": (float(ds.p.min()), float(ds.p.max())),
        "tau_range": tau_range,
    }
    return ds, meta


def pinn_loss(net: MuNet, ds: Dataset, lam_concave: float = 1.0,
              lam_zero: float = 2.0) -> tuple[torch.Tensor, dict]:
    """Loss = ODE residual + concavity prior + boundary.

    The concavity prior penalises POSITIVE second derivative only -- it
    enforces a single-arch shape (rise, peak, fall) without prescribing where
    the peak sits. Symmetric d2_mu^2 priors actively discourage the post-peak
    descent (high concavity = high curvature = penalised), which was the
    fundamental failure mode of the previous loss. This penalty allows the
    descent.

    The boundary term pins mu(0) ~ 0 (no force without slip).
    """
    p = DEFAULTS
    s = torch.tensor(ds.s, dtype=torch.float32)
    v = torch.tensor(ds.v, dtype=torch.float32)
    dv = torch.tensor(ds.dv_dt, dtype=torch.float32)

    mu = net(s)
    dv_pred = -mu * p["g"] - (p["k"] / p["m"]) * v * v
    loss_data = torch.mean((dv - dv_pred) ** 2)

    # Concavity on a dense grid: d2_mu / ds2 <= 0.
    s_grid = torch.linspace(0.0, 0.3, 80, requires_grad=True)
    mu_grid = net(s_grid)
    dmu_ds = torch.autograd.grad(mu_grid.sum(), s_grid, create_graph=True)[0]
    d2mu_ds2 = torch.autograd.grad(dmu_ds.sum(), s_grid, create_graph=True)[0]
    loss_concave = torch.mean(torch.relu(d2mu_ds2) ** 2)

    # mu(0) = 0
    loss_zero = (net(torch.tensor([0.0])) ** 2).mean()

    total = loss_data + lam_concave * loss_concave + lam_zero * loss_zero
    return total, {
        "data":    float(loss_data.item()),
        "concave": float(loss_concave.item()),
        "zero":    float(loss_zero.item()),
    }


def pinn_loss_2d(net: MuNet2D, ds: BrakeDataset, lam_concave: float = 1.0,
                 lam_mono_p: float = 0.5) -> tuple[torch.Tensor, dict]:
    p_def = DEFAULTS
    s = torch.tensor(ds.s, dtype=torch.float32)
    pr = torch.tensor(ds.p, dtype=torch.float32)
    v = torch.tensor(ds.v, dtype=torch.float32)
    dv = torch.tensor(ds.dv_dt, dtype=torch.float32)

    mu_eff = net(s, pr)
    dv_pred = -mu_eff * p_def["g"] - (p_def["k"] / p_def["m"]) * v * v
    loss_data = torch.mean((dv - dv_pred) ** 2)

    # Concavity on mu(s) -- single-arch shape, allows the post-peak fall.
    s_grid = torch.linspace(0.0, 0.3, 80, requires_grad=True)
    mu_grid = net.mu(s_grid)
    dmu_ds = torch.autograd.grad(mu_grid.sum(), s_grid, create_graph=True)[0]
    d2mu_ds2 = torch.autograd.grad(dmu_ds.sum(), s_grid, create_graph=True)[0]
    loss_concave = torch.mean(torch.relu(d2mu_ds2) ** 2)

    # Monotonicity of ramp(p): brake force does not decrease as pressure rises.
    p_grid = torch.linspace(0.0, 1.0, 50, requires_grad=True)
    r_grid = net.ramp(p_grid)
    dr_dp = torch.autograd.grad(r_grid.sum(), p_grid, create_graph=True)[0]
    loss_mono = torch.mean(torch.relu(-dr_dp) ** 2)

    # Identifiability: pin ramp(1) ~ 1 so the factorisation isn't a free scale.
    loss_pin = (net.ramp(torch.tensor([1.0])) - 1.0).pow(2).mean()

    total = (loss_data + lam_concave * loss_concave
             + lam_mono_p * loss_mono + 0.5 * loss_pin)
    return total, {
        "data":    float(loss_data.item()),
        "concave": float(loss_concave.item()),
        "mono":    float(loss_mono.item()),
        "pin":     float(loss_pin.item()),
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
                print(f"  epoch {epoch:4d}  total={loss.item():.4f}  "
                      f"data={parts['data']:.4f}  smooth={parts['smooth']:.4f}")
    return net, history


def train_pinn_2d(ds: BrakeDataset, *, epochs: int = 5000, lr: float = 5e-3,
                  seed: int = 0, verbose: bool = False
                  ) -> tuple[MuNet2D, list[float]]:
    torch.manual_seed(seed)
    net = MuNet2D()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history: list[float] = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss, parts = pinn_loss_2d(net, ds)
        loss.backward()
        opt.step()
        if epoch % 200 == 0 or epoch == epochs - 1:
            history.append(float(loss.item()))
            if verbose:
                print(f"  epoch {epoch:4d}  total={loss.item():.4f}  "
                      f"data={parts['data']:.4f}  smooth={parts['smooth']:.4f}  "
                      f"mono={parts['mono']:.4f}  pin={parts['pin']:.4f}")
    return net, history


def evaluate_curve(net: MuNet, n: int = 200, s_max: float = 0.3
                   ) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, s_max, n)
    with torch.no_grad():
        mu = net(torch.tensor(s, dtype=torch.float32)).numpy()
    return s, mu


def evaluate_curve_2d(net: MuNet2D, n: int = 200, s_max: float = 0.3
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns `(s, mu_s, ramp_p)` -- the two factorised heads."""
    s = np.linspace(0.0, s_max, n)
    p = np.linspace(0.0, 1.0, n)
    with torch.no_grad():
        mu_s = net.mu(torch.tensor(s, dtype=torch.float32)).numpy()
        ramp_p = net.ramp(torch.tensor(p, dtype=torch.float32)).numpy()
    return s, mu_s, ramp_p


# ---------------------------------------------------------------------------
# Grey-box Pacejka: parametric ID with 4 learnable scalars.
# ---------------------------------------------------------------------------

class PacejkaNet(nn.Module):
    """Grey-box Pacejka: four learnable scalars (B, C, D, E) wired through
    the analytic magic formula. Trained on the same ODE residual as MuNet
    but with no free-form MLP -- the shape is guaranteed physically valid by
    construction.

    Parameter ranges are kept physical via softplus / sigmoid wrappers so
    Adam can't drive them into nonsense values.
    """

    def __init__(self,
                 B0: float = 8.0, C0: float = 1.6,
                 D0: float = 0.85, E0: float = 0.3):
        super().__init__()
        # Use unconstrained params + smooth maps to physical ranges.
        # softplus keeps B,C strictly positive; sigmoid maps D in (0,1.2),
        # E in (-2, 2).
        import math
        self._B_raw = nn.Parameter(torch.tensor(math.log(math.expm1(B0))))
        self._C_raw = nn.Parameter(torch.tensor(math.log(math.expm1(C0))))
        self._D_raw = nn.Parameter(torch.tensor(
            math.log(D0 / (1.2 - D0))))                       # inv-sigmoid * 1.2
        self._E_raw = nn.Parameter(torch.tensor(
            math.log((E0 + 2.0) / (2.0 - E0))))               # inv-sigmoid * (-2,2)

    @property
    def B(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._B_raw)

    @property
    def C(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._C_raw)

    @property
    def D(self) -> torch.Tensor:
        return 1.2 * torch.sigmoid(self._D_raw)

    @property
    def E(self) -> torch.Tensor:
        return 4.0 * torch.sigmoid(self._E_raw) - 2.0

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s_clip = torch.clamp(s, min=0.0)
        Bs = self.B * s_clip
        inner = Bs - self.E * (Bs - torch.arctan(Bs))
        return self.D * torch.sin(self.C * torch.arctan(inner))

    def params_dict(self) -> dict:
        return {
            "B": float(self.B.item()),
            "C": float(self.C.item()),
            "D": float(self.D.item()),
            "E": float(self.E.item()),
        }


def pacejka_loss(net: PacejkaNet, ds: Dataset) -> tuple[torch.Tensor, dict]:
    p = DEFAULTS
    s = torch.tensor(ds.s, dtype=torch.float32)
    v = torch.tensor(ds.v, dtype=torch.float32)
    dv = torch.tensor(ds.dv_dt, dtype=torch.float32)

    mu = net(s)
    dv_pred = -mu * p["g"] - (p["k"] / p["m"]) * v * v
    loss_data = torch.mean((dv - dv_pred) ** 2)
    return loss_data, {"data": float(loss_data.item())}


def train_pacejka(ds: Dataset, *, epochs: int = 3000, lr: float = 1e-2,
                  seed: int = 0, verbose: bool = False
                  ) -> tuple[PacejkaNet, list[float]]:
    torch.manual_seed(seed)
    net = PacejkaNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history: list[float] = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss, _ = pacejka_loss(net, ds)
        loss.backward()
        opt.step()
        if epoch % 200 == 0 or epoch == epochs - 1:
            history.append(float(loss.item()))
            if verbose:
                p = net.params_dict()
                print(f"  epoch {epoch:4d}  loss={loss.item():.4f}  "
                      f"B={p['B']:.2f} C={p['C']:.2f} D={p['D']:.3f} E={p['E']:.3f}")
    return net, history


def evaluate_pacejka_curve(net: PacejkaNet, n: int = 200, s_max: float = 0.3
                           ) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, s_max, n)
    with torch.no_grad():
        mu = net(torch.tensor(s, dtype=torch.float32)).numpy()
    return s, mu
