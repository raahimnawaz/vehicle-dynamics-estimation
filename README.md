# Vehicle Dynamics & Physics-Informed Parameter Estimation

[![CI](https://github.com/raahimnawaz/vehicle-dynamics-ml-/actions/workflows/ci.yml/badge.svg)](https://github.com/raahimnawaz/vehicle-dynamics-ml-/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A computational framework for modeling vehicle longitudinal braking dynamics from first principles, simulating them with explicit ODE solvers, and recovering the underlying physical parameters from noisy telemetry. Three estimators are compared on the same data: a SciPy batch optimizer, an Extended Kalman Filter for online estimation, and a small MLP trained on synthetic rollouts.

The project sits at the intersection of vehicle dynamics, computational physics, and system identification, and is structured to scale toward real-data and PINN/edge-inference extensions.

---

## Quickstart

```bash
git clone https://github.com/raahimnawaz/vehicle-dynamics-ml-
cd tester
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python reproduce.py                                # regenerates every figure in results/
pytest                                             # runs the test suite
```

`reproduce.py` runs the full pipeline end-to-end (synthetic + real-telemetry-shaped CSV) and writes every figure shown below to `results/`.

---

## Results

### Synthetic benchmark

| Method | μ Estimate | Error | Characteristics |
|--------|------------|-------|-----------------|
| True | 0.7000 | — | Ground truth |
| SciPy Batch | 0.6988 | 0.17% | Offline optimal |
| EKF | 0.6756 | 3.5% | Real-time |
| Neural Net | 0.6854 | 2.1% | Fast inference |

![Vehicle Dynamics Estimation](figures/estimation_results.png)

### Real telemetry (Option A)

A loader in `src/data/telemetry.py` ingests CSV logs of the form `time,speed` (units: seconds, m/s — the format produced by most OBD-II / GPS pipelines and by [comma2k19](https://github.com/commaai/comma2k19) after a one-line conversion). A representative braking clip is shipped in `data/sample_braking.csv` so the demo is fully reproducible offline; `python reproduce.py --real` runs the SciPy and EKF estimators against it and writes `results/real_estimation.png`.

### C++ edge port (Jetson-targeted)

The `cpp-edge-port` branch ports the EKF and PINN to header-only, allocation-free C++17. Weights are baked into the binary at compile time via [`tools/export_weights.py`](tools/export_weights.py) — no file I/O, no PyTorch runtime, no ONNX dependency.

**Latency** (x86_64, MSYS2 UCRT64 g++ 15.1.0, `-O3 -ffast-math`, 50k samples)

| Op | median | p99 | throughput |
|---|---|---|---|
| EKF step (predict + update) | **10 ns** | 13 ns | 100 Mops/s |
| PINN forward (1→32→32→1) | **689 ns** | 1.28 µs | 1.45 Mops/s |

**Numerical parity** vs the Python reference on identical inputs:

| Quantity | max\|Δ\| |
|---|---|
| EKF $v$, $\mu$ | $\sim 10^{-9}$ |
| PINN $\mu_\theta(s)$ | $2.3 \times 10^{-7}$ (limited by float32 → float64) |

**Binary size** (stripped): bench 62 KB, parity 80 KB.

Jetson aarch64 numbers will be added once the hardware is in hand — same `make run-bench` on the device, JSON drops into [`benchmarks/`](benchmarks/). See [`cpp/README.md`](cpp/README.md) for build, cross-compile, and Cortex-M porting notes.

### PINN: discovering μ(s) from data (Option C)

![PINN recovery](results/pinn_recovery.png)

The MLP $\mu_\theta : s \mapsto \mu$ recovers a saturating Pacejka-like tire curve from 8 noisy braking trajectories *without prescribing the functional form*. Training uses the ODE residual of the longitudinal vehicle equation as the data term, plus a monotonicity prior on $d\mu/ds$ enforced on a dense slip grid via autograd.

| | value |
|---|---|
| Architecture | 1 → 32 → 32 → 1 MLP, tanh + sigmoid-scaled output |
| Training data | 1,931 collocation points across 8 runs |
| Slip range covered | $s \in [0, 0.25]$ |
| max $\lvert \hat\mu - \mu_{\text{true}} \rvert$ (in-range) | 0.090 |
| mean $\lvert \hat\mu - \mu_{\text{true}} \rvert$ (in-range) | 0.020 |
| Training loss at convergence | $\approx 2.7$ m²/s⁴ (noise floor of finite-difference $\dot v$) |

The trained weights are exported to `models/pinn_mu.pth` so the C++ edge port can load the same network for inference.

### Model-mismatch study (Option B) — which method when?

Ground-truth trajectories are generated from the *full* slip-aware model and corrupted with one unmodeled effect at a time (road grade, headwind, brake-force ramp). Each estimator is then fed the noisy trace, runs its own (often incorrect) inverse model, and the *predicted* velocity trajectory is scored against the clean ground truth. The point isn't to crown a winner — it's to map the operating envelope of each method, which is the question a robotics PI is actually asking.

![Mismatch heatmap](results/mismatch_heatmap.png)

| Method | model used to predict | best at | worst at |
|---|---|---|---|
| **Batch (SciPy)** | constant-μ with $k$ as a free parameter | everything (≤ 1.6 % RMSE/$v_0$) | brake ramp absorbs into $k$ |
| **EKF** | constant-μ, $k$ hardcoded | none of these (constant-μ is structurally wrong for slip-model truth) | brake ramp (35.8 %) |
| **NN (FrictionNet)** | constant-μ, pretrained on a different distribution | none — fully out-of-distribution | all (~22 – 25 %) |
| **PINN** | slip-aware with learned $\mu_\theta(s)$ | headwind, headwind (0.5 %), grade (6.8 %) | brake ramp (17.1 %) — PINN doesn't model time-varying brake force |

![Per-method degradation](results/mismatch_per_method.png)

![Trajectory overlay](results/mismatch_trajectories.png)

**Read this as an actual map**:

- If you have *enough data offline* and *control over the model class*, the batch fitter is essentially noise-floor accurate even with these mismatches — because it has $k$ as a free parameter to absorb structural error. This is the answer for system identification.
- The EKF is a fine *state* tracker, but using it as a *parameter* estimator under model-structure mismatch is a category error. Its predictive RMSE is dominated by the constant-μ assumption, not by the parameter it estimates. Don't deploy an EKF for parameter ID unless you trust the dynamics structure.
- A pretrained MLP is brittle across distributions. The FrictionNet here was trained on constant-μ data and falls apart on slip-model data, regardless of which extra effect is layered on. The fix is either domain randomization at training time, or — better — embedding the physics structure in the model itself, which is what the PINN does.
- The PINN owns the regime where the slip model is a faithful description of reality. Once an effect violates *its* model (a road grade adding gravity, or a brake ramp making μ time-dependent rather than slip-dependent), it degrades — gracefully on grade (the bias absorbs as effective μ), more sharply on brake ramp (the time dimension isn't in its inputs at all).

### Adversarial EKF scenarios (Option D)

![Adversarial EKF](results/adversarial_ekf.png)

The same EKF, stressed three ways. Each panel pair shows the velocity track (true vs. measurements, with dropouts shown as gaps) and the corresponding $\mu$ estimate with $\pm 2\sigma$ covariance bounds:

| Scenario | Final-window error in $\mu$ | What it shows |
|---|---|---|
| **Mid-run road change** (dry $\mu=0.8 \to$ wet $\mu=0.35$ at $t=2$ s) | 0.04 | Process noise on $\mu$ tuned to chase abrupt transitions. |
| **Sensor dropout bursts** (50% rate, 60-sample bursts) | 0.001 | Covariance grows during blackout, collapses on reacquire. |
| **Biased sensor** (+1.5 m/s constant offset) | 0.06 | Predictable bias-induced bias in $\mu$ — bounds the practical risk of a miscalibrated wheel-speed. |

Tuning knob: `q_mu` in `src/scenarios/runner.py` (process noise on $\mu$). The step-change panel uses `q_mu=1e-2`; the steady-state benchmark uses `1e-4`. This is the kind of trade-off you'd reach for first when porting the filter to a real ECU.

---

## Physical Derivation

### Tire friction force

Normal force: $N = mg$. Friction force:

$$F_f = \mu N = \mu m g$$

### Aerodynamic drag

$$F_d = \tfrac{1}{2}\,\rho\,C_d\,A\,v^2$$

where $\rho$ is air density, $C_d$ the drag coefficient, $A$ the frontal area, and $v$ the vehicle velocity.

### Governing equation of motion

Summing forces on the vehicle:

$$m\frac{dv}{dt} = -F_f - F_d = -\mu m g - \tfrac{1}{2}\rho C_d A v^2$$

which simplifies to

$$\frac{dv}{dt} = -\mu g - \frac{\rho C_d A}{2m}\,v^2.$$

---

## Tire Slip Model (Nonlinear Extension)

Real tires don't have a constant friction coefficient — friction depends on the slip ratio between wheel and ground:

$$s = \frac{R\omega - v}{v}$$

with $R$ the tire radius and $\omega$ the wheel angular velocity. A simple saturating model captures the low-slip / peak / saturation regimes:

$$\mu(s) = \mu_{\max}\,\bigl(1 - e^{-Cs}\bigr).$$

Plugging this back into the force balance yields the full nonlinear ODE that is solved numerically:

$$m\frac{dv}{dt} = -\mu(s)\,m g - \tfrac{1}{2}\,\rho C_d A\,v^2.$$

---

## Numerical Methods

The forward model is solved with:

- Euler integration (baseline)
- Runge-Kutta 4th order (RK4) — default
- SciPy ODE solvers (reference)

The test suite verifies that RK4 converges at 4th order on a known closed-form solution (`tests/test_rk4.py`).

---

## Parameter Estimation (Inverse Problem)

Given an observed velocity trace $v_{\mathrm{obs}}(t)$, we recover

$$\theta = \{\mu,\ C_d,\ \rho,\ \text{slip parameters}\}$$

by minimising the trajectory mismatch

$$\mathcal{L}(\theta) = \sum_t \bigl(v_{\mathrm{obs}}(t) - v_{\mathrm{sim}}(t,\theta)\bigr)^2.$$

Three estimators implement this:

- **SciPy batch** (`src/estimation/optimize.py`) — Nelder-Mead on the full trajectory.
- **EKF** (`src/estimation/kalman.py`) — joint state/parameter filter, $x = [v,\mu]^\top$.
- **FrictionNet** (`src/ml/train.py`) — MLP mapping a 50-sample velocity window to $\mu$.

---

## Pipeline

```text
Physics derivation
        ↓
Forward simulation (RK4 ODE)
        ↓
Synthetic / real telemetry
        ↓
Sensor noise model
        ↓
Estimation (batch / EKF / NN)
        ↓
Validation & visualisation
```

---

## Project Structure

```text
src/
├── physics/        # governing equations
├── solvers/        # Euler, RK4, SciPy wrappers
├── simulation/     # forward vehicle model + sensor model
├── data/           # real-telemetry CSV loader
├── estimation/     # batch optimiser + EKF
├── ml/             # FrictionNet (PyTorch MLP)
├── scenarios/      # adversarial scenarios + runner
└── visualization/  # plotting
tests/              # pytest suite
data/               # sample telemetry CSV
results/            # generated figures (regenerated by reproduce.py)
```

---

## Roadmap

- [x] First-principles forward model + RK4 solver
- [x] Batch / EKF / NN estimators on synthetic data
- [x] Real-telemetry CSV loader + reproducible pipeline (Option A)
- [x] Adversarial / edge-case EKF: mid-run $\mu$ change, dropouts, biased sensor (Option D)
- [x] PINN inverse solver discovering $\mu(s)$ without prescribing the curve (Option C)
- [x] Model-mismatch study mapping where each method breaks (Option B)
- [x] C++ edge port: header-only, allocation-free, weights baked in, parity to $10^{-9}$
- [ ] Jetson aarch64 benchmarks

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
