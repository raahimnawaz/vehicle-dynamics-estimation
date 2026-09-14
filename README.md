# Vehicle Dynamics & Physics-Informed Parameter Estimation

[![CI](https://github.com/raahimnawaz/vehicle-dynamics-estimation/actions/workflows/ci.yml/badge.svg)](https://github.com/raahimnawaz/vehicle-dynamics-estimation/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A reproducible benchmark for vehicle-dynamics system identification: first-principles longitudinal braking dynamics, explicit ODE solvers, five estimators on the same data (batch optimiser, EKF, MLP, two physics-informed networks), an honest model-mismatch study, and an allocation-free C++ edge port that matches the Python reference to $10^{-9}$.

`python reproduce.py --all` regenerates every figure and number in this README from seeded inputs. [Corrections](#corrections) lists the audits that moved several of them.

![Recovering the tire curve from noisy braking data](results/pinn_recovery.png)

*The headline result: a network with no assumed functional form recovers the Pacejka friction curve — rise, peak and post-peak fall — from the scattered red cloud of noisy trajectory data behind it, locating the peak at (0.127, 0.889) against a true (0.127, 0.900).*

---

## Quickstart

```bash
git clone https://github.com/raahimnawaz/vehicle-dynamics-estimation
cd vehicle-dynamics-estimation
python -m venv venv && source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
python reproduce.py --all                            # regenerates every figure below
pytest                                               # 26 fast tests; add -m slow for the other 4
```

## Headline results

| Result | Number | Where |
|---|---|---|
| Batch optimiser, μ-recovery error (synthetic) | **0.1 %** | [Synthetic](#1-synthetic-benchmark) |
| Grey-box Pacejka recovery, mean \|Δμ\| | **0.007** | [PINN](#3-pinn-discovering-the-pacejka-curve-from-data) |
| Function-free PINN, mean \|Δμ\| | **0.013** | [PINN](#3-pinn-discovering-the-pacejka-curve-from-data) |
| Function-free PINN, recovered peak | **(0.127, 0.889)** vs true (0.127, 0.900) | [PINN](#3-pinn-discovering-the-pacejka-curve-from-data) |
| Combined-slip PINN, mean \|Δμ\| / \|Δellipse\| | **0.005** / **0.004** | [Combined slip](#3c-combined-slip-the-friction-ellipse-munetcombined) |
| Combined-slip PINN-C, worst-case cornering RMSE | **1.9 %** of $v_0$ (vs 20.0 % PINN, 19.4 % PINN-B) | [Mismatch](#4-model-mismatch-which-method-when) |
| Brake-aware PINN-B, worst-case brake-ramp RMSE | **3.1 %** of $v_0$ (vs 16.9 % EKF, 17.9 % PINN, 21.3 % MLP) | [Mismatch](#4-model-mismatch-which-method-when) |
| C++ EKF latency | **7 ns** median, 9 ns p99 (Apple M-series) | [C++ port](#6-c-edge-port) |
| C++ binary, deps, allocations | 33 KB deployable, **0** allocator symbols, **0** external deps | [C++ port](#6-c-edge-port) |
| Numerical parity (Py ↔ C++) | $7.4 \times 10^{-9}$ EKF, $3.2 \times 10^{-7}$ PINN | [C++ port](#6-c-edge-port) |

## Scope

What these numbers do and do not show:

- **Longitudinal braking only.** Lateral demand enters as an exogenous input (§3c). There is no bicycle model, yaw dynamics or load transfer.
- **Ground truth is simulated.** Every recovery error is measured against a known Pacejka curve. The one real clip (§2) has no ground truth.
- **Slip is an input, not a state.** Wheel rotational dynamics are not simulated, so the loop is never closed. The estimators are given noisy vehicle velocity; on a real vehicle, slip is derived from wheel speed against an *estimated* vehicle speed, which is circular, since $v$ is what the estimator is producing.
- **The PINNs see more than the baselines.** They receive the slip trajectory noiseless, and PINN-B and PINN-C also get their second channel (brake pressure, lateral utilisation) noiseless and perfectly time-aligned, built from ground truth. Batch, EKF and the MLP see only noisy velocity. Brake pressure and lateral acceleration are real CAN and IMU signals, but not clean ones, so §4 compares partly *more information* against *a different algorithm*. Re-running it on equal information is the first [Roadmap](#roadmap) item.

---

## 1. Synthetic benchmark

Constant-μ forward model, ground-truth μ = 0.7, sensor noise applied; three estimators recover μ from the same noisy trace.

![Synthetic estimation](figures/estimation_results.png)

| Method | μ̂ | error | latency | role |
|---|---:|---:|---|---|
| Ground truth | 0.7000 | — | — | — |
| SciPy batch (Nelder-Mead) | 0.6994 | 0.1 % | offline | offline-optimal |
| Extended Kalman Filter | 0.6902 | 1.4 % | 7 ns / step (C++) | online |
| FrictionNet (MLP, 50-sample window) | 0.6393 | 8.7 % | 262 ns (C++ PINN path) | one-shot inference |

The MLP is the weakest, as expected: it has no physics and must infer a slope from 50 raw velocity samples, across which velocity falls only 3.5 m/s against 0.5 m/s of sensor noise.

**These are one noise draw.** Over 20 seeds, the batch fit's error has median 0.1 % (max 0.7 %), the EKF's 1.5 % (p10–p90 0.9–2.3 %), and the MLP's 2.7 % with p10–p90 of 0.4–8.2 %, so the draw above is an unlucky one for the MLP. Read the ordering, not the third digit.

## 2. Real telemetry

[`src/data/telemetry.py`](src/data/telemetry.py) loads `time,speed` CSVs (seconds, m/s), the format of most OBD-II / GPS pipelines and of [comma2k19](https://github.com/commaai/comma2k19) after a one-line conversion. On the shipped braking clip ([`data/sample_braking.csv`](data/sample_braking.csv), 41 samples at 10 Hz, `python reproduce.py --real`):

| Method | μ̂ |
|---|---:|
| SciPy batch | 0.8333 |
| EKF (final) | 0.7482 |

Both converge without retuning. There is no ground truth for this clip, so the useful signal is agreement between two independent estimators rather than either value on its own.

---

## 3. PINN: discovering the Pacejka curve from data

Real tires do not have a constant friction coefficient. μ depends on the slip ratio $s = (R\omega - v)/v$ and follows the **Pacejka magic formula**:

$$\mu(s) \;=\; D \cdot \sin\!\Big( C \cdot \arctan\!\big( B s - E (B s - \arctan(B s)) \big) \Big).$$

The curve **rises, peaks near $s \approx 0.13$, then falls**. Left of the peak the tire is self-correcting (more slip, more force); right of it $d\mu/ds < 0$ and the loop inverts, so slip grows until the wheel locks. Holding the operating point just left of the peak is what an ABS controller exists to do.

Two networks recover this curve from the same noisy braking data (figure at top), illustrating the cost of prior strength; a third (§3c) adds the lateral axis.

### 3a. Function-free PINN (`MuNet`)

A small MLP $\mu_\theta : s \mapsto \mu$ trained against the ODE residual, with no prescribed functional form and **no shape prior**.

| | value |
|---|---:|
| Architecture | 1 → 32 → 32 → 1 MLP, tanh + scaled-sigmoid output |
| Training | 5,310 collocation points, 16 sweep schedules, 6,000 epochs |
| Constraint | $\mu(0) = 0$ boundary only |
| Recovered peak (s, μ) | $(0.127,\ 0.889)$ — true peak $(0.127,\ 0.900)$ |
| mean $\lvert \hat\mu - \mu_{\text{true}} \rvert$ (in-range) | **0.013** |
| max  $\lvert \hat\mu - \mu_{\text{true}} \rvert$ (in-range) | 0.163 at $s = 0.300$, the edge of the data; 0.029 over $s \leq 0.28$ ([C7](docs/CORRECTIONS.md#c7--two-inspection-fixes-that-moved-published-numbers)) |

**Every shape prior this net once carried was a false claim about tire physics.** Monotonicity forbade the post-peak fall outright. Symmetric smoothness penalised the concavity that forms the peak. One-sided concavity, shipped for months, penalised the true curve too, because Pacejka is convex on $s \in [0.233, 0.300]$ as it flattens toward sliding friction. With that prior on, the recovered peak sat at the grid edge on every seed:

| | peak location (3 seeds) | mean \|Δμ\| |
|---|---|---:|
| `lam_concave = 1.0` | 0.300, 0.300, 0.300 | 0.061 |
| `lam_concave = 0.0` | 0.124, 0.127, 0.125 | **0.013** |

The 1D problem is well posed: $\mu(s)$ is the only unknown, so the ODE residual already determines it and a shape prior can only add bias. Only $\mu(0) = 0$ remains, and that is physics, not a guess about shape. The 2D nets turned out the same way ([C6](docs/CORRECTIONS.md#c6--a-shape-prior-that-was-standing-in-for-an-under-weighted-anchor)): every net here now trains shape-prior free, pinned only by identities true by construction. Full history in [C3](docs/CORRECTIONS.md#c3--a-shape-prior-that-penalised-the-ground-truth).

### 3b. Grey-box Pacejka (`PacejkaNet`)

The industry-standard approach: four learnable scalars $(B, C, D, E)$ through the analytic Pacejka formula, trained against the **same ODE residual**. The physics guarantees a valid curve.

| Parameter | truth | recovered | error |
|---|---:|---:|---:|
| $B$ (stiffness) | 10.00 | 9.35 | 6.5 % |
| $C$ (shape) | 1.90 | 2.05 | 7.9 % |
| $D$ (peak) | 0.900 | 0.889 | 1.2 % |
| $E$ (curvature) | 0.50 | 0.68 | — |
| **mean $\lvert \hat\mu - \mu_{\text{true}} \rvert$** | — | — | **0.007** |
| **recovered peak** | $(0.127, 0.900)$ | $(0.125, 0.889)$ | 1.6 % in s, 1.2 % in μ |

**Why both?** `PacejkaNet` reaches 0.007 because it *knows the family*; `MuNet` reaches 0.013 without that assumption. That gap is the real cost of being function-free, and it is much smaller than it looked while a misspecified prior was in the way (0.053 against 0.007): **most of the apparent price of dropping structural assumptions was the price of adding a wrong one.** If you know your tire family, take the grey-box. `MuNet` earns its place when the truth sits outside the family you would have assumed.

### 3c. Combined slip: the friction ellipse (`MuNetCombined`)

A tire has one friction budget, and force spent turning is not available for stopping. The standard lumped model is the **friction ellipse**: with $n = a_y / (g D)$ the share of the budget spent laterally, the longitudinal coefficient that remains is

$$\mu_x(s, n) \;=\; \mu(s)\,\sqrt{1 - n^2}.$$

`MuNetCombined` is handed $(s, n)$ and has to factorise that product back into a tire curve and a derating factor, without being told the second factor is $\sqrt{1-n^2}$.

![Combined-slip PINN recovery](results/pinn_combined_recovery.png)

| | value |
|---|---:|
| Architecture | two heads, $\mu(s)$ and $\mathrm{ellipse}(n)$, 1 → 32 → 32 → 1 each |
| Training | 4,451 collocation points, 16 runs, 4,000 epochs |
| Constraints | $\mu(0) = 0$ and $\mathrm{ellipse}(0) = 1$ — both physics, no shape prior |
| mean $\lvert\Delta\mu(s)\rvert$ | **0.005** |
| mean $\lvert\Delta\,\mathrm{ellipse}(n)\rvert$ | **0.004** |
| Recovered peak | $(0.128,\ 0.891)$ — true $(0.127,\ 0.900)$ |

0.005 beats both 1D nets, but not like-for-like: this net sees strictly more, and a curve observed at several derating levels is better determined than one observed at a single level. The fair reading is that the lateral channel pays for itself, not that free-form beats grey-box.

**What makes it identifiable.** Many (shape, scale) splits fit the residual equally well, so something has to choose. Three seeds per row:

| ablation | mean $\lvert\Delta\mu\rvert$ | recovered peak |
|---|---:|---|
| **as shipped** | **0.005** | 0.128 ✓ |
| anchor weight 50 → 2 | 0.194 | grid edge |
| data never straight ($n \geq 0.35$) | 0.208 | grid edge |
| $n$ tied rigidly to $s$ (collinear) | 0.124 | grid edge |
| concavity prior on | 0.062 | grid edge |
| ramp-up-only runs (corr 0.70) | 0.004 | 0.125 ✓ |

- **$\mathrm{ellipse}(0) = 1$ has to bind in the loss *and* in the data.** At weight 2 the optimiser pays the penalty, settles at $\mathrm{ellipse}(0) \approx 0.83$ and fits the product to 0.011 with the wrong factors. With no near-straight samples, the anchor constrains a point the data never visits. Training longer rescues neither: 8,000 epochs reaches $\mathrm{ellipse}(0) \approx 0.98$ and still recovers the wrong curve.
- **A prediction that was wrong.** I expected coverage of the $(s,n)$ plane to be the binding constraint. It is not: ramp-up-only data with $\mathrm{corr}(s,n) = 0.70$ recovers the split to 0.004. Only *exact* collinearity breaks it.
- **Operationally,** you cannot calibrate combined-slip tire capacity from cornering data alone. A fleet that only brakes mid-corner yields data its own residual fits perfectly and a tire curve wrong by 0.2 in μ, with no diagnostic that anything is off. Straight-line braking events fix the scale.

This is a lumped derating of the whole curve by one scalar, not the combined-slip Pacejka $(\kappa, \alpha)$ force surface with the $G_{x\alpha}$ / $G_{y\kappa}$ weighting functions of Pacejka (2002) §4.3.2.

---

## 4. Model mismatch: which method, when?

Ground truth comes from the **full** Pacejka slip-aware model, corrupted with one unmodeled effect at a time: road grade, headwind, brake-force ramp (time-varying $\mu_{\text{eff}}$) and sustained cornering (friction-ellipse derating). Each estimator runs its own, often incorrect, inverse model on the noisy trace, and its *predicted* trajectory is scored against clean ground truth. This is not a leaderboard; it is the operating envelope of each method. Read it with the information caveat in [Scope](#scope).

![Mismatch heatmap](results/mismatch_heatmap.png)

### 4a. Worst-case mismatch (RMSE / $v_0$, %)

| Method | grade (0 – 0.12 rad) | headwind (0 – 15 m/s) | brake ramp (τ ≤ 0.8 s) | cornering (n ≤ 0.75) | mean |
|---|---:|---:|---:|---:|---:|
| **Batch (SciPy)** | 0.8 % | 1.0 % | **1.6 %** | 0.5 % | **0.6 %** |
| EKF | 2.8 % | 2.9 % | 16.9 % | 1.8 % | 3.1 % |
| NN (FrictionNet) | 4.8 % | 2.4 % | 21.3 % | 3.0 % | 4.6 % |
| PINN (1D, function-free) | 6.4 % | 1.7 % | 17.9 % | 20.0 % | 4.8 % |
| **PINN-B (brake-aware, 2D)** | 6.9 % | 2.3 % | **3.1 %** | 19.4 % | 3.5 % |
| **PINN-C (cornering-aware, 2D)** | 6.2 % | 1.5 % | 18.2 % | **1.9 %** | 3.1 % |

Each cell is the **worst** RMSE over that effect's five-point sweep; for nine of the 24 it comes from a lighter setting, including both 2-D headlines (PINN-B at τ = 0.15 s, PINN-C at n = 0.40). The heatmap plots the heaviest setting only, so the two differ in places. The mean is over each method's 20 sweep runs.

![Per-method degradation curves](results/mismatch_per_method.png)

### 4b. Brake-ramp degradation — the headline

The brake ramp models lag in building brake force, $F_{\text{brake}}(t) \propto 1 - e^{-t/\tau}$; at $\tau = 0.8$ s the brakes reach only ~50 % force at $t = 0.55$ s. A constant-μ estimator reads the first second as a low-friction surface, and a slip-only PINN cannot represent a force that depends on time independently of slip.

![Brake-ramp degradation](results/mismatch_brake_curve.png)

| τ (s) | Batch | EKF | NN | PINN | **PINN-B** | PINN-C |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 (normal brakes) | 0.7 % | 1.2 % | 3.0 % | 0.4 % | 0.6 % | 0.4 % |
| 0.15 (cold pads) | 0.5 % | 3.4 % | 3.4 % | 0.5 % | **3.1 %** | 0.6 % |
| 0.30 (worn hydraulics) | 1.2 % | 5.5 % | 14.9 % | 5.5 % | **1.5 %** | 5.7 % |
| 0.50 (serious fault) | 1.5 % | 12.9 % | 19.4 % | 11.1 % | **0.5 %** | 11.3 % |
| **0.80 (near-failure)** | **1.6 %** | **16.9 %** | **21.3 %** | **17.9 %** | **0.6 %** | 18.2 % |

**PINN-B** factorises $\mu_{\text{eff}}(s, p) = \mu_\theta(s) \cdot \mathrm{ramp}_\theta(p)$, with $p \in [0,1]$ normalised brake pressure. It is the only online method that does not degrade: it *improves* as the ramp lengthens, ending at 0.6 % where every other online method is between 17 % and 21 %, below even the offline batch fit. Underneath that trajectory score it recovers $\mu(s)$ to 0.013 and the ramp to 0.049.

The 0.6 % is a best case with a noiseless, perfectly aligned pressure trace. The defensible claim is the narrow one: **adding the brake-pressure channel fixes brake-lag mismatch, and brake pressure is a real signal already on the CAN bus.**

![Brake-aware PINN recovery](results/pinn_brake_recovery.png)

### 4c. Cornering degradation — and why an extra channel is not general robustness

| n = a_y / (gD) | Batch | EKF | NN | PINN | PINN-B | **PINN-C** |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 (straight) | 0.1 % | 1.4 % | 0.4 % | 1.8 % | 2.4 % | 1.6 % |
| 0.20 | 0.5 % | 1.5 % | 2.5 % | 0.7 % | 0.2 % | 0.1 % |
| 0.40 | 0.2 % | 1.0 % | 1.4 % | 2.8 % | 2.3 % | 1.9 % |
| 0.60 | 0.5 % | 1.2 % | 2.1 % | 11.4 % | 10.9 % | **0.2 %** |
| **0.75 (hard cornering)** | 0.5 % | 1.8 % | 3.0 % | **20.0 %** | **19.4 %** | **1.2 %** |

**The slip-only PINNs are the worst methods here.** Sustained cornering is a *constant multiplicative* derating, which Batch and the EKF fold into their free μ: biased parameter, accurate trajectory. $\mu_\theta(s)$ is pinned by the slip input and has nowhere to put a force that does not depend on slip. Committing to more physics costs the slack to absorb physics you did not model.

**An extra channel buys robustness to that effect, not robustness in general.** PINN-B is still 19.4 % under cornering; PINN-C is still 18.2 % under the brake ramp. Instrumenting the effect you care about fixes that effect and nothing else, which is an argument about sensor selection rather than estimator sophistication.

### 4d. How to read this map

- **Offline parameter ID → Batch.** Its free $k_{\text{drag}}$ absorbs structural error, so it stays near the noise floor everywhere. That is partly one more degree of freedom than the online methods; handing the EKF the same parameter does not close the gap ([Roadmap](docs/ROADMAP.md)).
- **Online state tracking → EKF.** Folds unmodeled *constant* forces into μ (2.8 % under grade, 2.9 % under headwind). Its weakness is time-varying friction, 16.9 %.
- **MLP → the control condition.** Trained on constant-μ, Euler-integrated data and tested on Pacejka RK4 truth: a distribution-shift demonstration, not a fair competitor.
- **Tire-curve recovery → PINN.** Best at recovering $\mu(s)$ itself (§3a), but worse on road grade (6.4 %) than every constant-μ method and the worst on cornering (20.0 %).
- **Time-varying friction → PINN-B; combined slip → PINN-C.** The right second input takes worst-case brake-ramp error from 16.9 % (EKF) to 3.1 %, and worst-case cornering error from 20.0 % (PINN) to 1.9 %.

---

## 5. Adversarial EKF scenarios

The same EKF, stressed three ways. Each panel pair shows the velocity track (dropouts as gaps) and the μ estimate with $\pm 2\sigma$ bounds.

![Adversarial EKF](results/adversarial_ekf.png)

| Scenario | final-window error in μ | what it shows |
|---|---:|---|
| **Mid-run road change** (dry μ = 0.8 → wet μ = 0.35 at $t = 2$ s) | 0.018 | Process noise on μ tuned to chase abrupt transitions ($q_\mu = 10^{-2}$). |
| **Sensor dropout bursts** (50 % rate, 60-sample bursts) | 0.010 | Covariance grows during blackout, collapses on reacquire. |
| **Biased sensor** (+1.5 m/s constant offset) | 0.005 | A small bias in μ — bounds the practical risk of a miscalibrated wheel-speed sensor. |

Tuning knob: `q_mu` in [`src/scenarios/runner.py`](src/scenarios/runner.py). The step-change panel uses $10^{-2}$, the steady-state benchmark $10^{-4}$; it is the first trade-off you reach for when porting a filter to a real ECU.

---

## 6. C++ edge port

[`cpp/`](cpp/) ports the EKF and PINN to header-only, allocation-free C++17. Weights are baked in at compile time by [`tools/export_weights.py`](tools/export_weights.py): no file I/O, no PyTorch runtime, no ONNX. The 2×2 matrix type is hand-rolled rather than pulled from Eigen so the port cross-compiles to Cortex-M unchanged.

### 6a. Latency

Apple M-series, arm64, median across batched samples. Both columns are archived in [`benchmarks/`](benchmarks/) and regenerable (`python tools/bench_python.py`, `make -C cpp bench-json`).

| Op | Python | C++ | Speedup |
|---|---:|---:|---:|
| EKF step — vs NumPy `VehicleEKF` | 2,520 ns | **7 ns** | 360× |
| EKF step — vs scalar Python, same math | 289 ns | **7 ns** | **41×** |
| PINN forward (1→32→32→1) | 10,580 ns | **262 ns** | 40× |

**41× is the fair number.** The NumPy filter spends most of its time on per-call dispatch for 2×2 arrays, which is where an earlier 3,414× headline came from ([history](docs/CORRECTIONS.md#the-3414-speedup)). Latency was never the constraint anyway: even the slowest Python EKF step measured here, 34 µs, fits a 100 Hz control budget nearly 300 times over.

![Python vs C++ benchmark](results/bench_python_vs_cpp.png)

*The figure plots the older x86_64 / MSYS2 pair (`benchmarks/x86_64-*.json`).*

### 6b. Footprint

| | Python | C++ (stripped, arm64) |
|---|---|---|
| Deployable binary | ~60 MB (CPython + NumPy + PyTorch) | **33 KB**, of which 16 KB is `__TEXT` |
| Runtime allocations per EKF step | 7 (small NumPy temporaries) | **0** |
| Heap touched by PINN inference | grows with autograd graph | **512 bytes** of stack (fp64) |
| Undefined symbols in the shipped code | — | **`exp`, `tanh`, stack guard.** That is the whole list |

The 33 KB is `make -C cpp footprint`: a binary that links only the shipped headers (EKF step, PINN forward pass, baked-in weights), not the benchmark harness. It links no allocator at all, so the shipped code cannot allocate, and CI fails if one appears.

### 6c. Numerical parity

Both implementations are fed the same noisy input stream and diffed point-wise:

| Quantity | max $\lvert \Delta \rvert$ Py − C++ |
|---|---:|
| EKF $v$ | $7.4 \times 10^{-9}$ |
| EKF $\mu$ | $2.0 \times 10^{-9}$ |
| EKF $\sigma_v$, $\sigma_\mu$ | $5 \times 10^{-11}$ — $3.9 \times 10^{-10}$ |
| PINN $\mu_\theta(s)$ | $3.2 \times 10^{-7}$ (float32 weights in the Python net set the floor) |

The EKF gap is FP-reordering noise under `-ffast-math`. `parity_check.py` also asserts that the C++ and Python drag constants agree, and exits non-zero on any tolerance breach.

```bash
make -C cpp run-bench                    # builds + runs C++ bench
python tools/bench_python.py             # writes benchmarks/*-python.json
python tools/plot_bench_comparison.py    # writes results/bench_python_vs_cpp.png
python tools/parity_check.py             # verifies Py <-> C++ agreement
```

**Not yet done:** no SIMD, fp16/int8 or fixed-point path; no MISRA or static-analysis pass; no proven stack bound; no embedded aarch64 numbers. `-ffast-math` should be dropped for anything safety-relevant, since it permits reassociation and assumes no NaN or Inf reaches a covariance. The covariance update uses the simple $(I - KH)P$ form, which computes the two off-diagonals by different expressions and does not force $P$ to stay positive semi-definite; Joseph form is the standard hardening and is cheap at 2×2.

---

## Corrections

Audits in August and September 2026 found seven defects. All are fixed, each has its own commit, and [docs/CORRECTIONS.md](docs/CORRECTIONS.md) writes each one up in full: what was wrong, what it moved, and how it was found.

| | what was wrong | what it moved |
|---|---|---|
| [C1](docs/CORRECTIONS.md#c1--a-75-drag-coefficient-inconsistency-between-the-two-forward-models) | Drag 75× larger in the estimators than in the ground-truth model | EKF mean mismatch error 25.9 % → 3.7 %, MLP 23.3 % → 5.6 % |
| [C2](docs/CORRECTIONS.md#c2--the-c-pinn-applied-the-wrong-activation-scale) | Wrong output activation scale in the C++ PINN | Every C++ friction estimate 9.1 % high; claimed parity was really $2.3\times10^{-1}$ |
| [C3](docs/CORRECTIONS.md#c3--a-shape-prior-that-penalised-the-ground-truth) | Three shape priors, each false about tire physics | `MuNet` mean \|Δμ\| 0.061 → 0.013, interior peak restored |
| [C4](docs/CORRECTIONS.md#c4--a-test-that-pinned-the-bug-in-place) | A test that required the C1 bug to be present | Replaced by a physical invariant |
| [C5](docs/CORRECTIONS.md#c5--two-latent-traps-found-by-audit-closed-before-either-fired) | RK4 at 1st order on time-varying μ; Pacejka defaults for a different tire | No published number; both closed with tests |
| [C6](docs/CORRECTIONS.md#c6--a-shape-prior-that-was-standing-in-for-an-under-weighted-anchor) | A shape prior standing in for an under-weighted anchor | PINN-B mean \|Δμ(s)\| 0.045 → 0.015, worst brake-ramp RMSE 4.4 % → 3.1 % |
| [C7](docs/CORRECTIONS.md#c7--two-inspection-fixes-that-moved-published-numbers) | Inspection fixes (smoother, RNG) shipped without re-running the pipeline | Small shifts across §1, §3 and §4; §4a headers relabelled |

---

## Physical model

### Force balance

Normal force $N = mg$; friction force $F_f = \mu N$; aerodynamic drag $F_d = \tfrac{1}{2} \rho C_d A v^2$. Newton's second law gives

$$m \frac{dv}{dt} \;=\; -F_f - F_d \;=\; -\mu m g \;-\; \tfrac{1}{2} \rho C_d A v^2 \quad\Longrightarrow\quad \frac{dv}{dt} \;=\; -\mu g \;-\; K_{\text{drag}}\, v^2.$$

`K_DRAG` is defined once in [`src/physics/wheel.py`](src/physics/wheel.py) as $\tfrac{1}{2}\rho C_d A / m = 0.4 / 1500 = 2.667\times10^{-4}$ m⁻¹, and every model, estimator, training set, tool and the C++ port import it. That gives 0.21 m/s² of drag deceleration at 28 m/s against 8.8 m/s² of tire friction: aero drag is a ~2 % correction at road speeds.

### Tire slip model

Slip ratio $s = (R\omega - v)/v$. Two friction models are implemented in [`src/physics/wheel.py`](src/physics/wheel.py):

- `mu_exponential(s, μ_max, C)` — saturating ablation, $\mu(s) = \mu_{\max}(1 - e^{-Cs})$. Monotone, no peak.
- `mu_pacejka(s, B, C, D, E)` — **default ground truth**. Pacejka 1989 magic formula, rises → peak → falls.

Wheel rotational dynamics are an order of magnitude stiffer than the vehicle's translational dynamics and would force a sub-millisecond integrator, so they are not simulated: the slip profile an ABS controller would produce is a caller-supplied schedule instead (see [Scope](#scope)).

### Numerical integration

- Euler (baseline), RK4 (default), SciPy adaptive (`solve_ivp`).
- [`tests/test_rk4_order.py`](tests/test_rk4_order.py) verifies 4th-order convergence against a closed-form solution.
- The slip-aware model uses an **inline** RK4 in `wheel.simulate` rather than the generic helper in `solvers/rk4.py`, because a time-varying slip schedule makes the ODE non-autonomous. Each stage must sample $s(t)$ at $t$, $t + dt/2$ and $t + dt$; the generic autonomous helper samples only at the step start and silently degrades to 1st order.

### Inverse problem

Given observed $v_{\text{obs}}(t)$, recover $\theta = \{\mu \text{ or } (B,C,D,E),\ k_{\text{drag}}\}$ by minimising

$$\mathcal{L}(\theta) \;=\; \sum_t \big(v_{\text{obs}}(t) - v_{\text{sim}}(t, \theta)\big)^2$$

via Nelder-Mead (Batch), joint state-parameter EKF, FrictionNet, MuNet, MuNet2D, or PacejkaNet.

For the EKF the state is $x = [v, \mu]^T$ with μ modelled as a random walk. Only velocity is observed, so $H = [1, 0]$, the innovation covariance $S = P_{00} + R$ is a scalar, and no matrix inverse appears anywhere in the filter; the C++ port specialises identically. μ is observable not from the measurement directly but through the off-diagonal $P_{10}$ that the propagation step builds up via the $-g\,dt$ Jacobian entry, coupling μ to the velocity residual.

---

## Project structure

```text
reproduce.py         # the entry point: regenerates every figure and number above
src/
├── physics/        # governing equations, K_DRAG, Pacejka + exponential μ
├── solvers/        # generic RK4 (autonomous problems only — see the docstring)
├── simulation/     # forward vehicle model + sensor model
├── data/           # real-telemetry CSV loader
├── estimation/     # batch optimiser + EKF
├── ml/             # FrictionNet + MuNet/MuNet2D/PacejkaNet/MuNetCombined
├── scenarios/      # adversarial + mismatch sweep
└── visualization/  # plotting
cpp/                # header-only allocation-free C++17 port (+ a footprint target)
tools/              # benchmark + parity + weight-export scripts
tests/              # pytest suite (26 fast, 4 slow — CI runs both)
docs/               # full corrections history and roadmap scoping
data/               # sample telemetry CSV
results/            # generated figures (regenerated by reproduce.py)
figures/            # duplicate of the synthetic-benchmark figure, kept for older links
benchmarks/         # latency JSON dumps, Python and C++, one pair per host
models/             # exported PyTorch weights
```

---

## Roadmap

- [ ] **Equal-information mismatch study.** Noisy derived slip, lagged noisy brake pressure, noisy lateral acceleration: degrade each channel until the PINN advantage disappears. This turns §4a into a fair comparison rather than an informative one.
- [ ] **PINN-D:** $\mu(s)\cdot\mathrm{ramp}(p)\cdot\mathrm{ellipse}(n)$, one net blind on neither axis. The anchor count closes; the data is the hard part.
- [ ] Close the loop: wheel rotational dynamics + slip observer
- [ ] Lateral dynamics: bicycle model + $F_y(\alpha)$ identification
- [ ] Full combined-slip Pacejka with per-axle load transfer
- [ ] Jetson aarch64 benchmarks; NEON / SVE intrinsics; fp16/int8 quantisation
- [ ] Joseph-form covariance update; drop `-ffast-math` for the shipping path
- ~~Augment the EKF state to $[v, \mu, k]$~~ — tried and rejected: $k$ is effectively unobservable at road speeds, and the worst-case gain it buys comes from a slack variable absorbing structural error, not from identified drag.

Scoping, effort estimates, predictions and the rejected approach in full: [docs/ROADMAP.md](docs/ROADMAP.md).

---

## Reproducibility

`python reproduce.py --all` regenerates every figure from seeded inputs and re-bakes the C++ weights header, so `tools/parity_check.py` stays valid. It asserts nothing, so the guarantees live in CI:

| job | what it catches |
|---|---|
| `test` | ordinary unit regressions, on 3.11 and 3.12 |
| `regression-guards` | `pytest -m slow`, including the invariant that keeps [C1](docs/CORRECTIONS.md#c1--a-75-drag-coefficient-inconsistency-between-the-two-forward-models) from recurring. `pytest.ini` deselects slow tests by default, so it needs its own job |
| `cpp-parity` | a stale `pinn_weights.h` against the committed model, any Python ↔ C++ divergence, and any allocator linked into the footprint binary |
| `reproduce` | the end-to-end pipeline still runs |

Latency and binary-size figures are host-dependent and labelled with the machine that produced them; both sides of every benchmark pair are archived under [`benchmarks/`](benchmarks/).

## License

Apache 2.0 — see [LICENSE](LICENSE).
