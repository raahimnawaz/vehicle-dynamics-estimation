"""Physics-Informed Networks for tire-friction system identification.

Four networks are exposed. The first three illustrate a spectrum from
"function-free" to "grey-box parametric"; the fourth adds a second physical
axis rather than moving along that spectrum:

  * `MuNet`       : s -> mu, free-form MLP, no shape prior by default.
                    Recovers Pacejka shape (rise, peak, fall) without being
                    told the functional form. See `pinn_loss` for why the
                    shape prior this repo used to ship was removed.
  * `MuNet2D`     : (s, p) -> mu, factorised mu(s) * ramp(p).
                    Required when brake-force lag (time-varying mu_eff) is
                    in play -- adds normalised brake pressure as a 2nd input.
  * `PacejkaNet`  : grey-box. Four learnable scalars (B, C, D, E) that are
                    plugged into the analytic Pacejka magic formula. Same
                    ODE-residual loss; no free-form MLP. Industry-standard
                    parameter ID -- the cleanest recovery the data allows.
  * `MuNetCombined`: (s, n) -> mu_x, factorised mu(s) * ellipse(n).
                    Combined slip -- the tire's friction budget is shared
                    between cornering and braking. Needs no shape prior,
                    because ellipse(0) = 1 is an exact identity that pins the
                    factorisation where MuNet2D's scale convention cannot.

The structural-prior story is the point of having both `MuNet` and
`PacejkaNet` in the repo. Three shape priors were tried on MuNet and all
three encoded a false claim about tire physics -- see `pinn_loss` for the
full account and the numbers:

  * Monotonicity (`relu(-d_mu/ds)^2`) forbids the post-peak fall outright.
  * Symmetric smoothness (`mean d2mu^2`) penalises the concavity that forms
    the peak exactly as hard as convexity.
  * One-sided concavity (`relu(d2mu/ds2)^2`) penalises the post-peak
    FLATTENING, because the true curve is convex over the last 22% of the
    slip range as it approaches the sliding-friction asymptote.

No shape prior is applied by default now. Only `mu(0) = 0` remains, which is
a physical constraint rather than a guess about curve shape. Free-form
recovery then lands at mean |d_mu| = 0.013 against the grey-box net's
0.007 -- so the real cost of being function-free is far smaller than it
looked while a misspecified prior was in the way.

Layering `PacejkaNet` alongside `MuNet` still makes that cost explicit:
MuNet rediscovers the shape, PacejkaNet only fits 4 scalars.

Pipeline:
  generate_dataset(...)          : roll forward Pacejka truth + sensor noise
  generate_dataset_braking(...)  : same, with a brake-ramp time constant tau
  generate_dataset_combined(...) : same, with a lateral-utilisation channel
  MuNet / MuNet2D / PacejkaNet /
      MuNetCombined              : the networks
  pinn_loss / pinn_loss_2d /
      pinn_loss_combined /
      pacejka_loss               : residual + structural priors
  train_pinn / train_pinn_2d /
      train_pinn_combined /
      train_pacejka              : Adam loops
  evaluate_curve(_2d/_combined)  : sample on a grid for plotting
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
    """Centred moving average with EDGE padding, not zero padding.

    `np.convolve(mode="same")` pads with zeros, which drags the first/last w//2
    samples of a braking trace toward 0 -- e.g. v_s[0] fell to ~0.55*v0 and its
    finite-difference dv/dt spiked to ~330 m/s^2. Those corrupted boundary
    samples were only kept out of the training set incidentally, by the
    |dv/dt| < 12 mask downstream; a gentler schedule or a looser mask would let
    them through. Edge padding keeps the endpoints on the signal so the guard is
    no longer load-bearing. (Formal-inspection defect D-02.)
    """
    arr = np.asarray(arr, dtype=float)
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w // 2, (w - 1) // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


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


def pinn_loss(net: MuNet, ds: Dataset, lam_concave: float = 0.0,
              lam_zero: float = 2.0) -> tuple[torch.Tensor, dict]:
    """Loss = ODE residual + boundary (+ optional concavity prior).

    `lam_concave` defaults to 0. Read the history before turning it back on --
    this loss went through three shape priors and every one of them encoded a
    claim about tire physics that turned out to be false:

      1. Monotonicity, penalising d_mu/ds < 0. A real tire curve FALLS after
         the peak, so this forbade the correct answer outright. The network
         saturated and it read as a capacity problem.

      2. Symmetric smoothness, penalising (d2_mu/ds2)^2. Same defect in
         disguise: it punishes the concavity that FORMS the peak exactly as
         hard as convexity.

      3. One-sided concavity, penalising relu(d2_mu/ds2)^2 -- the version
         this repo shipped. Subtler, and still wrong. The true Pacejka curve
         is CONVEX on s in [0.233, 0.300], 22.3% of the evaluation range,
         because the post-peak fall flattens out toward the sliding-friction
         asymptote. The prior penalises exactly that flattening; the penalty
         it assigns to the ground-truth curve is 0.34, not 0.

    With the prior on, the recovered curve has no interior peak at all -- its
    maximum sits at the right edge of the grid on every seed tested. Dropping
    it recovers the peak at s = 0.124-0.127 (truth: 0.127) with mean |d_mu| of
    0.013, against 0.061 with the prior, consistently across seeds.

    The data does not need the help: pointwise inversion of the ODE residual
    puts the empirical peak at s = 0.110, mu = 0.897, close to the true
    (0.127, 0.900). The shape was always in the measurements -- each prior was
    an assumption fighting them.

    The boundary term pins mu(0) ~ 0 (no force without slip), which is a
    genuine physical constraint rather than a guess about curve shape, and
    stays on.
    """
    p = DEFAULTS
    s = torch.tensor(ds.s, dtype=torch.float32)
    v = torch.tensor(ds.v, dtype=torch.float32)
    dv = torch.tensor(ds.dv_dt, dtype=torch.float32)

    mu = net(s)
    dv_pred = -mu * p["g"] - (p["k"] / p["m"]) * v * v
    loss_data = torch.mean((dv - dv_pred) ** 2)

    # Optional concavity prior on a dense grid: d2_mu / ds2 <= 0. Off by
    # default -- see the docstring. Skipped entirely when lam_concave == 0 so
    # we do not pay for two autograd passes we are going to multiply by zero.
    if lam_concave > 0.0:
        s_grid = torch.linspace(0.0, 0.3, 80, requires_grad=True)
        mu_grid = net(s_grid)
        dmu_ds = torch.autograd.grad(mu_grid.sum(), s_grid, create_graph=True)[0]
        d2mu_ds2 = torch.autograd.grad(dmu_ds.sum(), s_grid, create_graph=True)[0]
        loss_concave = torch.mean(torch.relu(d2mu_ds2) ** 2)
    else:
        loss_concave = torch.zeros((), dtype=torch.float32)

    # mu(0) = 0
    loss_zero = (net(torch.tensor([0.0])) ** 2).mean()

    total = loss_data + lam_concave * loss_concave + lam_zero * loss_zero
    return total, {
        "data":    float(loss_data.item()),
        "concave": float(loss_concave.item()),
        "zero":    float(loss_zero.item()),
    }


def pinn_loss_2d(net: MuNet2D, ds: BrakeDataset, lam_concave: float = 0.0,
                 lam_mono_p: float = 0.5, lam_pin: float = 50.0
                 ) -> tuple[torch.Tensor, dict]:
    """Loss for the factorised mu(s) * ramp(p) net.

    `mu(s) * ramp(p)` is under-determined -- it admits a family of (shape, scale)
    splits that fit the residual equally well -- so something has to pin the
    split. There are two candidates, and which one you use decides everything:

      * a CONCAVITY PRIOR on mu(s), which is a guess about curve shape, and
      * the ANCHOR ramp(1) = 1, which is an exact identity: at full brake
        pressure the ramp is complete, so mu_eff(s, 1) is the tyre curve itself.

    This net shipped for a long time leaning on the first, with the anchor
    present but weighted 0.5 -- a hundredth of the weight the same anchor
    carries in `pinn_loss_combined`, and four times below the setting that was
    already measured to FAIL there (sec 3c: at weight 2 the optimiser simply
    pays the penalty instead of obeying it). Measured over three seeds:

        lam_pin  concave |  mean|d_mu|  ramp err   recovered peak
            0.5      1.0 |      0.057     0.140    0.207 / 0.170 / 0.258
            5.0      1.0 |      0.133     0.184    all wrong
           50.0      1.0 |      0.042     0.147    all wrong
           50.0      0.0 |      0.010     0.069    0.133 / 0.130 / 0.131  <-
            0.5      0.0 |      0.293     0.139    grid edge

    The anchor alone, at a weight that actually binds, recovers the curve 5.7x
    better than the prior did and finds the true peak (0.127) on every seed --
    which the shipped configuration never did. The prior was not compensating
    for an unidentifiable problem; it was substituting for an under-weighted
    constraint, and it made things worse once the constraint could bind.

    The bottom-left row is the one that caused the original mistake: drop the
    prior while the anchor is still too weak to take over and the net collapses
    to 0.293, which reads as "the prior is load-bearing" when it actually means
    "nothing is pinning the split".

    So the rule from `pinn_loss` holds here after all, with no exception:
    prior off, exact physics on. Every net in this repo now runs shape-prior
    free; only anchors that are true by construction remain.
    """
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

    # Identifiability: pin ramp(1) = 1 so the factorisation isn't a free scale.
    # This is an exact identity, not a preference -- at full brake pressure the
    # ramp is complete by definition -- so it is weighted to bind (see the
    # docstring for what happens at 0.5 and 5).
    loss_pin = (net.ramp(torch.tensor([1.0])) - 1.0).pow(2).mean()

    total = (loss_data + lam_concave * loss_concave
             + lam_mono_p * loss_mono + lam_pin * loss_pin)
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
                      f"data={parts['data']:.4f}  "
                      f"concave={parts['concave']:.4f}  "
                      f"zero={parts['zero']:.4f}")
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
                      f"data={parts['data']:.4f}  "
                      f"concave={parts['concave']:.4f}  "
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


# ---------------------------------------------------------------------------
# Combined slip: mu_x(s, n) = mu(s) * ellipse(n)
# ---------------------------------------------------------------------------
#
# Same factorised shape as MuNet2D, a different second axis: normalised
# lateral utilisation n = a_y / (g D) instead of brake pressure. The physics
# is in `src/physics/wheel.py::friction_ellipse`; the truth is
# ellipse(n) = sqrt(1 - n^2), which the network is not told.
#
# The identifiability story differs from the brake case in one useful way.
# `pinn_loss_2d` pins ramp(1) ~ 1, which is a *convention* -- nothing physical
# says full brake pressure means undiminished friction, it is just where the
# scale is chosen to sit. Here the anchor is physics: at zero lateral demand
# the tire's whole budget is available longitudinally, so ellipse(0) = 1
# exactly. That is a stronger constraint, and it is why this net needs no
# shape prior on mu(s) while the brake-ramp net does.
#
# It is also a constraint the data has to reach, and that turns out to be the
# operationally interesting part. Pin the anchor at n = 0 but give the network
# no samples near n = 0 -- `generate_dataset_combined(n_floor_range=(0.35,
# 0.50))`, a fleet that only ever brakes mid-corner -- and mean |d_mu| goes
# from 0.007 to 0.208 with the peak back at the grid edge, while the pin sits
# in the loss the whole time doing nothing. The practical reading: you cannot
# calibrate combined-slip tire capacity from cornering data alone. The
# straight-line braking events are what fix the scale, and a dataset without
# them fits its own residual just as well while recovering the wrong curve.


@dataclass
class CombinedDataset:
    v: np.ndarray
    s: np.ndarray
    n: np.ndarray            # normalised lateral utilisation in [0, 1)
    dv_dt: np.ndarray
    run_id: np.ndarray


class MuNetCombined(nn.Module):
    """mu_x(s, n) = mu(s) * ellipse(n). Two heads, each a small MLP."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.ellipse_head = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def mu(self, s: torch.Tensor) -> torch.Tensor:
        return 1.1 * torch.sigmoid(self.mu_head(s.view(-1, 1))).squeeze(-1)

    def ellipse(self, n: torch.Tensor) -> torch.Tensor:
        # 1.05 rather than 1.0 for the same reason MuNet uses 1.1: the
        # ellipse(0) = 1 pin sits at the top of the range, and a clean sigmoid
        # reaches 1.0 only asymptotically, so the pin would have to saturate
        # the unit to be satisfied. The headroom keeps that gradient alive.
        return 1.05 * torch.sigmoid(self.ellipse_head(n.view(-1, 1))).squeeze(-1)

    def forward(self, s: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        return self.mu(s) * self.ellipse(n)


def generate_dataset_combined(
    n_runs: int = 16,
    t_final: float = 4.0,
    dt: float = 0.01,
    noise_v: float = 0.15,
    seed: int = 0,
    pacejka: dict | None = None,
    n_peak_range: tuple[float, float] = (0.15, 0.80),
    n_floor_range: tuple[float, float] = (0.0, 0.05),
    ramp_down_frac: float = 0.5,
) -> tuple[CombinedDataset, dict]:
    """Braking runs that are simultaneously cornering, at varying intensity.

    Each run ramps lateral utilisation from a floor to a peak on its own
    schedule, out of phase with the slip sweep. The phase offset is what keeps
    the two inputs from being an exact function of one another, which is the
    one condition under which mu(s) * ellipse(n) genuinely stops being
    separable -- see the note on `ramp_down_frac` below for the measured
    boundary between "correlated" (fine) and "collinear" (not).

    `n_floor_range` is the ablation knob for the identifiability experiment.
    Its default starts runs essentially straight, so the data reaches the
    ellipse(0) = 1 anchor. Raising it to (0.35, 0.50) models a fleet that only
    ever brakes mid-corner: the anchor stays in the loss but leaves the data,
    and mean |d_mu| degrades from 0.007 to 0.208 over three seeds.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_final, dt)
    pj = pacejka or PACEJKA_DRY
    g = DEFAULTS["g"]
    k_over_m = DEFAULTS["k"] / DEFAULTS["m"]

    vs, ss, ns, dvs, rids = [], [], [], [], []
    for run in range(n_runs):
        v0 = float(rng.uniform(22.0, 32.0))
        s_max = float(rng.uniform(0.20, 0.35))
        hold_start = float(rng.uniform(0.0, 0.3))

        n_floor = float(rng.uniform(*n_floor_range))
        n_peak = float(rng.uniform(*n_peak_range))
        n_peak = max(n_peak, n_floor)
        # Lateral ramp gets its own start and duration so it is not collinear
        # with the slip sweep.
        n_start = float(rng.uniform(0.1, 1.2))
        n_ramp = float(rng.uniform(0.6, 2.0))
        # Half the runs release lateral demand as they brake (trail-braking
        # out of a corner) instead of building it, which decorrelates n from
        # the upward slip sweep: corr(s, n) falls from 0.70 to 0.12.
        #
        # Worth being exact about what that buys, because it is less than it
        # looks. Ramp-up-only data (corr 0.70) recovers the split perfectly
        # well -- mean |d_mu| = 0.004 against 0.007 for the mixed set. What
        # breaks identifiability is not partial correlation but *exact*
        # collinearity: tie n rigidly to s (corr = 1.00) and mean |d_mu| goes
        # to 0.124 with the peak pinned at the grid edge, because then no
        # (shape, scale) split is distinguishable from any other. Mixed
        # directions are here for physical variety and margin against that
        # degenerate case, not because the diagonal set fails.
        ramp_down = bool(rng.random() < ramp_down_frac)

        sched = sweep_slip(s_max=s_max, t_total=t_final, hold_start=hold_start)
        s_t = np.array([sched(ti) for ti in t])
        u = np.clip((t - n_start) / max(n_ramp, 1e-9), 0.0, 1.0)
        n_t = (n_peak - (n_peak - n_floor) * u if ramp_down
               else n_floor + (n_peak - n_floor) * u)

        mu_eff = mu_pacejka(s_t, **pj) * np.sqrt(np.clip(1.0 - n_t ** 2, 0.0, None))

        v = np.zeros_like(t)
        v[0] = v0
        for i in range(1, len(t)):
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
        ns.append(n_t[mask])
        dvs.append(dv_dt[mask])
        rids.append(np.full(int(mask.sum()), run, dtype=int))

    ds = CombinedDataset(
        v=np.concatenate(vs),
        s=np.concatenate(ss),
        n=np.concatenate(ns),
        dv_dt=np.concatenate(dvs),
        run_id=np.concatenate(rids),
    )
    meta = {
        "model": "pacejka+friction_ellipse",
        "params": pj,
        "n_samples": int(len(ds.v)),
        "s_range": (float(ds.s.min()), float(ds.s.max())),
        "n_range": (float(ds.n.min()), float(ds.n.max())),
        "n_floor_range": n_floor_range,
        "ramp_down_frac": ramp_down_frac,
    }
    return ds, meta


def pinn_loss_combined(net: MuNetCombined, ds: CombinedDataset,
                       lam_concave: float = 0.0, lam_mono_n: float = 0.5,
                       lam_zero: float = 2.0, lam_anchor: float = 50.0
                       ) -> tuple[torch.Tensor, dict]:
    """Loss for the factorised mu(s) * ellipse(n) net.

    `lam_concave` defaults to 0, matching `pinn_loss` and against
    `pinn_loss_2d`. The brake-ramp net needs a shape prior because its scale
    convention (ramp(1) ~ 1) does not pin the split; here ellipse(0) = 1 is
    physics and does, so the prior is not needed and -- per the C3 story in
    this module's docstring -- a prior that is not needed is a prior that can
    only inject bias. Measured both ways in `reproduce.py::run_pinn_combined`.

    The two physical constraints:
      * mu(0) = 0        -- no slip, no longitudinal force.
      * ellipse(0) = 1   -- no lateral demand, full budget available.
    Plus one structural one, that spending more of the budget laterally cannot
    increase longitudinal grip: d(ellipse)/dn <= 0.

    `lam_anchor` is 50 and that weight is load-bearing, not a hyperparameter
    that happened to work. ellipse(0) = 1 is an exact identity, not a soft
    preference, and it is the only thing pinning the (shape, scale) split. At
    lam_anchor = 2 the optimiser cheaply pays the penalty instead of obeying
    it: measured over three seeds it settles at ellipse(0) ~ 0.83, mu(s)
    saturates against its 1.1 cap with no interior peak, and mean |d_mu|
    degrades from 0.007 to 0.194 -- while the *product* mu * ellipse still
    fits the data to 0.011. The residual cannot see the difference; only the
    anchor can. Training longer does not rescue it (8,000 epochs reaches
    ellipse(0) ~ 0.98 and mean |d_mu| = 0.198): the split is decided early,
    so the anchor has to bind from the start.
    """
    p_def = DEFAULTS
    s = torch.tensor(ds.s, dtype=torch.float32)
    n = torch.tensor(ds.n, dtype=torch.float32)
    v = torch.tensor(ds.v, dtype=torch.float32)
    dv = torch.tensor(ds.dv_dt, dtype=torch.float32)

    mu_eff = net(s, n)
    dv_pred = -mu_eff * p_def["g"] - (p_def["k"] / p_def["m"]) * v * v
    loss_data = torch.mean((dv - dv_pred) ** 2)

    # Physics: no slip, no force.
    loss_zero = net.mu(torch.tensor([0.0])).pow(2).mean()

    # Physics: no lateral demand, full longitudinal budget. This is the anchor
    # that makes the factorisation identifiable -- see the module note.
    loss_anchor = (net.ellipse(torch.tensor([0.0])) - 1.0).pow(2).mean()

    # Structural: more lateral use cannot mean more longitudinal grip.
    n_grid = torch.linspace(0.0, 1.0, 60, requires_grad=True)
    e_grid = net.ellipse(n_grid)
    de_dn = torch.autograd.grad(e_grid.sum(), n_grid, create_graph=True)[0]
    loss_mono = torch.mean(torch.relu(de_dn) ** 2)

    loss_concave = torch.zeros(())
    if lam_concave > 0.0:
        s_grid = torch.linspace(0.0, 0.3, 80, requires_grad=True)
        mu_grid = net.mu(s_grid)
        dmu = torch.autograd.grad(mu_grid.sum(), s_grid, create_graph=True)[0]
        d2mu = torch.autograd.grad(dmu.sum(), s_grid, create_graph=True)[0]
        loss_concave = torch.mean(torch.relu(d2mu) ** 2)

    total = (loss_data + lam_zero * loss_zero + lam_anchor * loss_anchor
             + lam_mono_n * loss_mono + lam_concave * loss_concave)
    return total, {
        "data":    float(loss_data.item()),
        "zero":    float(loss_zero.item()),
        "anchor":  float(loss_anchor.item()),
        "mono":    float(loss_mono.item()),
        "concave": float(loss_concave.item()),
    }


def train_pinn_combined(ds: CombinedDataset, *, epochs: int = 5000,
                        lr: float = 5e-3, seed: int = 0,
                        lam_concave: float = 0.0, verbose: bool = False
                        ) -> tuple[MuNetCombined, list[float]]:
    torch.manual_seed(seed)
    net = MuNetCombined()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history: list[float] = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss, parts = pinn_loss_combined(net, ds, lam_concave=lam_concave)
        loss.backward()
        opt.step()
        if epoch % 200 == 0 or epoch == epochs - 1:
            history.append(float(loss.item()))
            if verbose:
                print(f"  epoch {epoch:4d}  total={loss.item():.4f}  "
                      f"data={parts['data']:.4f}  anchor={parts['anchor']:.4f}  "
                      f"mono={parts['mono']:.4f}")
    return net, history


def evaluate_curve_combined(net: MuNetCombined, n: int = 200,
                            s_max: float = 0.3
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns `(s, mu_s, n_grid, ellipse_n)` -- the two factorised heads."""
    s = np.linspace(0.0, s_max, n)
    n_grid = np.linspace(0.0, 1.0, n)
    with torch.no_grad():
        mu_s = net.mu(torch.tensor(s, dtype=torch.float32)).numpy()
        e_n = net.ellipse(torch.tensor(n_grid, dtype=torch.float32)).numpy()
    return s, mu_s, n_grid, e_n
