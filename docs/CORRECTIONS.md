# Corrections

Every defect that moved, or could have moved, a published number in this repo, with what it moved and how it was found. The [README](../README.md) carries a one-line summary of each.

An audit in August 2026 found three defects in the results (C1-C3) and one test that held the first of them in place (C4). A second pass in September 2026 found two latent traps that had not yet fired (C5), one shape prior standing in for a mis-weighted constraint (C6), and that the tooling written to prevent C1 and C2 from recurring was not actually wired into CI — see [Reproducibility](../README.md#reproducibility) for what runs now. A formal inspection later that month closed eight minor defects, two of which moved published numbers without the pipeline being re-run (C7). All are fixed; each has a dedicated commit with the full analysis. They are documented here rather than quietly patched, because two of the first three were invisible in the outputs, the third was actively protected by a passing test, and the last two had no symptom at all.

## C1 — A 75× drag-coefficient inconsistency between the two forward models

`wheel.py` wrote longitudinal drag as $(k/m)v^2$ with $k = 0.4$ kg/m, giving $2.667\times10^{-4}$. `model.py` wrote it as $kv^2$ with $k$ in 1/m, and every caller hardcoded `0.02` — **75× larger**. The mismatch harness generated ground truth with the first convention and fitted it with estimators built on the second, so the EKF and MLP carried 15.7 m/s² of phantom aerodynamic deceleration at 28 m/s, against 8.8 m/s² of *real* braking force.

The symptom was visible in the published results all along: at brake-ramp τ = 0.01, where essentially no unmodeled effect is active, the EKF reported 21.5 % RMSE. A correctly specified estimator is at the noise floor there. **A method that is wrong when nothing is perturbed is misconfigured, not mismatch-sensitive** — that reading is what located the bug.

Effect on the mismatch study (mean RMSE/$v_0$ over all 15 cells):

| Method | before | after |
|---|---:|---:|
| Batch | 1.1 % | 0.7 % |
| EKF | 25.9 % | **3.7 %** |
| NN | 23.3 % | **5.6 %** |
| PINN | 8.3 % | 3.9 % |
| PINN-B | 3.8 % | 3.0 % |

The PINN sections of `reproduce.py` came out byte-identical before and after, because they already used the correct constant — only the two methods that were wrong moved. There is now a single `K_DRAG` in `wheel.py`, and no literal drag value survives anywhere in the repo.

**Two conclusions reversed.** The EKF is not fragile under model mismatch — the earlier "category error" framing was an artifact. And the PINNs are now the *worst* methods on road grade, for the structural reason given in §4d.

It also flattered the MLP in §1. Under the unphysically large drag, velocity fell 9.5 m/s across the MLP's 50-sample window instead of 3.5 m/s, so the network had a 2.7× stronger signal and looked correspondingly more accurate.

## C2 — The C++ PINN applied the wrong activation scale

`cpp/include/vd/pinn.hpp` applied `1.2 * sigmoid` to the network output. `MuNet` applies `1.1 * sigmoid` — the 1.2 belongs to `MuNet2D` and had been copied into the wrong port. **Every friction estimate the C++ produced was 9.1 % high**, which for a value feeding a brake controller is an overestimate of available grip.

This README claimed PINN parity of $2.3\times10^{-7}$. Running the committed code against the committed weights, it was $2.3\times10^{-1}$ — five orders of magnitude worse. The parity harness had been printing a warning for it the whole time.

A second copy of the same problem sat next to it: `reproduce.py --all` retrains the PINN and overwrites `models/pinn_mu.pth` but never re-exported `cpp/include/vd/pinn_weights.h`, so following this README's own instructions silently invalidated the parity claim.

`export_weights.py` now recovers the output scale from the PyTorch module itself and emits it as `kOutScale` in the generated header, which `pinn.hpp` reads; `reproduce.py` re-bakes the header after training, and raises if that re-bake fails rather than warning past it. The activation can no longer diverge from the model.

## C3 — A shape prior that penalised the ground truth

The loss carried three successive shape priors and **every one of them encoded a claim about the tire curve that was false**:

1. **Monotonicity**, penalising $d\mu/ds < 0$. A real tire curve falls after the peak, so this forbade the correct answer outright. The network saturated, and it read as a capacity problem rather than a specification error.

2. **Symmetric smoothness**, penalising $(\partial^2\mu/\partial s^2)^2$. The same defect in disguise — it punishes the concavity that *forms* the peak exactly as hard as the convexity it was meant to suppress.

3. **One-sided concavity**, penalising $\mathrm{ReLU}(\partial^2\mu/\partial s^2)^2$. Subtler, and the version this repo shipped for months. The true Pacejka curve is **convex on $s \in [0.233, 0.300]$** — 22.3 % of the evaluation range — because the post-peak fall flattens toward the sliding-friction asymptote. The prior penalises exactly that flattening; the penalty it assigns to the *ground-truth curve* is 0.34.

With prior (3) active the recovered curve had no interior peak at all — its maximum sat at the right edge of the grid on every seed tested. Removing it:

| | peak location (3 seeds) | mean \|Δμ\| |
|---|---|---:|
| `lam_concave = 1.0` | 0.300, 0.300, 0.300 | 0.061 |
| `lam_concave = 0.0` | 0.124, 0.127, 0.125 | **0.013** |

The data never needed the help. Pointwise inversion of the ODE residual puts the empirical peak at $(0.110,\ 0.897)$ against a true $(0.127,\ 0.900)$, with the post-peak fall clearly resolved above the sample noise. The shape was always in the measurements; each prior was an assumption fighting them.

The 1D problem is well posed — $\mu(s)$ is the only unknown, so the ODE residual already determines it and any shape prior can only inject bias. Only the $\mu(0) = 0$ boundary remains, and that is physics (no force without slip) rather than a guess about curve shape.

## C4 — A test that pinned the bug in place

`test_sweep_qualitative_ranking` asserted `means["NN (FrictionNet)"] > 0.10` — it *required* the MLP's mean error to exceed 10 %, which was only true because of C1. CI was green precisely because the bug was present, and correcting the physics turned the suite red.

A test written by reading off current behaviour pins that behaviour whether or not it is correct. The replacement asserts a physical invariant instead: `test_no_method_is_broken_at_nominal` requires every estimator to be under 5 % RMSE when no effect is active, and would have failed on the original code from the first run.

## C5 — Two latent traps, found by audit, closed before either fired

Neither of these ever produced a wrong published number. Both are recorded because they are the
same defect class as C1 — one quantity with two definitions — and because the reason they were
harmless was luck about which code paths happened to be exercised, not design.

**A 4th-order integrator running at 1st order on any time-varying μ.** `src/solvers/rk4.py` is an
*autonomous* helper: it samples the right-hand side once, at the step start. Its docstring says so
and names the correct alternative. `run_sim.simulate` accepted a callable `mu(t)` and handed it to
that helper anyway, and `scenarios/runner.py` passes exactly such a callable. Sampling a
time-varying input once per step is Euler's approximation *of that input*, so the whole scheme
collapses to 1st order regardless of how many stages it has — measured at 2.00× error reduction
per halving of `dt` where RK4 gives 16×, a factor of 2.2 × 10⁷ at `dt = 0.01`.

It never mattered because every schedule this repo ships is piecewise constant in time — the
dry→wet transition of §5 is a step, and a step is autonomous on each side of the jump. The
measured discrepancy on that scenario was 7.4 × 10⁻³ m/s against 0.25 m/s of sensor noise, and
all three §5 figures are byte-identical before and after the fix. The trap was set for whoever
next wrote a *realistic* μ(t), which is the obvious next thing to write.

**A tire model whose defaults were a different tire.** `mu_pacejka`, `pacejka_peak` and
`mu_combined` carried `E = 0.97` in their signatures while `PACEJKA_DRY` — the set every dataset
and every figure is generated from — specifies `E = 0.5`. Calling any of them without keywords
returned a curve peaking at $s = 0.180$ against the true $0.127$: **42 % wrong in the one quantity
an ABS controller exists to track**, and differing by up to 0.133 in μ where the headline recovery
error is 0.013. Every call site in the repo passes `**PACEJKA_DRY` explicitly, which is the only
reason nothing was affected.

Both now have the treatment C1 got. There is one definition of the Pacejka set and the defaults
are read from it; `run_sim.simulate` samples μ at each RK4 sub-step. And both have a test that
fails if the defect returns — `test_simulate_is_fourth_order_in_a_time_varying_mu` (verified
against the reintroduced bug: observed order 1.0027) and
`test_pacejka_defaults_match_the_ground_truth_set`. A third,
`test_simulate_scalar_mu_is_unchanged_by_substep_sampling`, pins the constant-μ path bit-for-bit
so the integrator fix cannot quietly move §1, §2 or §4.

**The general point, which is why this section exists at all.** C1 and C2 were caught by their
symptoms — a number that was wrong in a way somebody eventually read correctly. These two had no
symptom. They were found by reading the code against its own documentation and asking what would
happen to the *next* caller, and that is the only method that finds this class at all.

## C6 — A shape prior that was standing in for an under-weighted anchor

The third instance of the same lesson, and the one that changed the most numbers.

`MuNet2D` (PINN-B) factorises $\mu_{\text{eff}}(s,p) = \mu_\theta(s)\cdot\mathrm{ramp}_\theta(p)$, which is
genuinely under-determined. It carried two things to pin the split: a **concavity prior** on $\mu(s)$ —
a guess about curve shape — and the **exact identity** $\mathrm{ramp}(1) = 1$, since at full brake
pressure the ramp is complete and $\mu_{\text{eff}}(s,1)$ *is* the tyre curve. §3a argued the prior was
load-bearing here, citing a measured collapse from 0.055 to 0.309 when it was removed.

The measurement was right. The conclusion was wrong. The anchor was weighted **0.5** — a hundredth of
the weight the identical kind of anchor carries in §3c, and four times below the setting already
measured to *fail* there, where at weight 2 the optimiser simply pays the penalty rather than obeying
it. Three seeds per row:

| `lam_pin` | concavity | mean $\lvert\Delta\mu\rvert$ | $\lvert\mathrm{ramp}(p)-p\rvert$ | recovered peak |
|---:|---:|---:|---:|---|
| 0.5 | 1.0 | 0.057 | 0.140 | 0.207 / 0.170 / 0.258 — *as shipped* |
| 5.0 | 1.0 | 0.133 | 0.184 | all wrong |
| 50.0 | 1.0 | 0.042 | 0.147 | all wrong |
| **50.0** | **0.0** | **0.010** | **0.069** | **0.133 / 0.130 / 0.131** ✓ |
| 0.5 | 0.0 | 0.293 | 0.139 | grid edge |

The bottom row is what caused the original error: drop the prior while the anchor is still too weak to
take over, and the net collapses — which reads as *the prior is load-bearing* when it actually means
*nothing is pinning the split*. Weight the anchor to bind and the prior is not merely unnecessary but
**harmful**: it is the difference between 0.042 and 0.010, and between never finding the peak and
finding it on every seed.

**Everything improved at once**, which is the tell that this was a defect rather than a trade:

| | before | after |
|---|---:|---:|
| mean $\lvert\Delta\mu(s)\rvert$ | 0.045 | **0.015** |
| mean $\lvert\mathrm{ramp}(p)-p\rvert$ | 0.136 | **0.049** |
| worst-case brake-ramp RMSE | 4.4 % | **3.1 %** |
| brake-ramp RMSE at τ = 0.8 s | 4.4 % | **0.6 %** |

**What it costs to state honestly.** §4b previously used PINN-B's poor recovery to argue that
trajectory accuracy and parameter identifiability come apart under factorisation. That thesis is still
true — §4c demonstrates it — but this was not an instance of it, and the section has been rewritten to
say so. A number that improves on four axes simultaneously was never measuring a trade-off.

**And the rule now has no exception.** Every network in this repo trains shape-prior free. What
resolves an under-determined factorisation is an identity that is true by construction — $\mu(0)=0$,
$\mathrm{ellipse}(0)=1$, $\mathrm{ramp}(1)=1$ — weighted so the optimiser cannot buy its way out of it.

## C7 — Two inspection fixes that moved published numbers

The inspection commit (`0ed8934`) changed two inputs that `reproduce.py` consumes and then checked the
fast tests, the slow tests it touched, and the 1D net's recovery. Nothing else was retrained. So for one commit this README described models the code no longer
produced. Re-running the full pipeline on that commit *and* on the one before it settled what moved. The
earlier run reproduces every number previously printed here exactly, so each difference below comes from
the two fixes and nothing else.

**Boundary smoothing (D-02).** The training-set smoother padded with zeros, which dragged the first samples
of every trace to ~0.56 $v_0$ and put ~314 m/s² into their dv/dt. The $\lvert dv/dt\rvert < 12$ mask
happened to discard them. Edge padding keeps them: five more samples per run (80 for `MuNet`, 60 for
PINN-B, 80 for PINN-C), all at slip below 0.002. They are sound but not free. Against the clean
trajectory their dv/dt error is 2.05 m/s² RMS with +0.10 bias, against 1.60 m/s² RMS and +0.06 for the
samples that were already in. Every net retrained on the new sets:

| | before | after |
|---|---:|---:|
| `MuNet` mean / max $\lvert\Delta\mu\rvert$ | 0.013 / 0.073 | 0.013 / **0.163** |
| `MuNet` recovered peak | (0.124, 0.890) | (0.127, 0.889) |
| PINN-B mean $\lvert\Delta\mu(s)\rvert$ | 0.015 | 0.013 |
| PINN-C mean $\lvert\Delta\mu\rvert$ / $\lvert\Delta\,\mathrm{ellipse}\rvert$ | 0.005 / 0.004 | 0.005 / 0.004 |
| PINN-B worst-case brake-ramp RMSE | 3.1 % | 3.1 % |
| PINN-C worst-case cornering RMSE | 2.0 % | 1.9 % |
| C++ PINN parity | $1.9\times10^{-7}$ | $3.2\times10^{-7}$ |

**The max error more than doubled, and it is not a regression. That was measured, not assumed.** The
maximum sits at one grid point, $s = 0.300$, the upper edge of the data's slip range, which only the tail
of the steepest sweeps reaches. Four seeds, old smoother against new: mean $\lvert\Delta\mu\rvert$ =
0.0126/0.0128, 0.0115/0.0099, 0.0079/0.0080, 0.0066/0.0065, with no consistent direction. Over
$s \leq 0.28$, the published seed's max error *fell*, 0.033 → 0.029, and no seed exceeds 0.036 under
either smoother. What moved is how an unconstrained net extrapolates at the edge of its data. A max
statistic reports that and a mean does not. The published seed is also the worst of the four on mean
error, so the 0.013 headline is a conservative one. The multi-seed ablation tables in §3a, §3c and C6
were measured under the old smoother and have not been re-run. This check found no directional change
in the quantity they report, but their exact digits are from the earlier data.

**Noise generator (D-06).** `add_noise` moved from numpy's global legacy RNG to an explicit `Generator`.
Same seed, different draw: §1 went from 0.3 / 1.9 / 5.7 % to 0.1 / 1.4 / 8.7 %, with no change to any
estimator. Twenty draws show why neither set deserves much weight on its own. The MLP's p10–p90 spans
0.4–8.2 % around a 2.7 % median. The new draw lands above its p90, and the old one was well above the median. §1 now states the spread.

**Also found while rebuilding §4a from the raw cells.** Its column headers named the heaviest intensity
("grade (0.12 rad)", "brake ramp (τ = 0.8 s)"). Every cell, though, has always been the worst value
over the sweep, and the mean is over each method's 20 sweep runs. Nine of the 24 table cells come from a lighter setting,
including both 2-D headlines: PINN-B's brake-ramp worst is at τ = 0.15 s, and PINN-C's cornering
worst is at n = 0.40. No number was wrong; the labels were, and they now say what
the cells measure.

---

## Benchmark and footprint claims (§6)

Not numbered above because no estimator output changed, but each was a published claim that did not measure what it said.

### The 3,414× speedup

An earlier version of this README headlined a 3,414× EKF speedup, measured on an older Windows/Zen host ([`benchmarks/x86_64-msys2-ucrt64.json`](../benchmarks/)). That measurement is real, but the denominator is misleading: `VehicleEKF` calls into NumPy on 2×2 arrays, where per-call dispatch costs far more than the ~30 floating-point operations the filter performs. `tools/bench_python.py` now also benchmarks `ScalarEkf` — identical math, plain Python, no NumPy — which runs 8.7× faster than the NumPy version. **That is the fair baseline, and against it the C++ port is ~41×, not thousands.** The rest was a library-choice artifact.

### The C++ latency column had no provenance

Until this table was rebuilt, the Python side was archived as JSON and the C++ side was typed in by hand from a run that was never recorded — so the one column nobody could re-derive was the one carrying the claim. Two things came out of fixing that. The benchmark now writes the same JSON schema as its Python counterpart (`make -C cpp bench-json`), and `run-bench` no longer defaults to `batch=1`: a single EKF step is shorter than `steady_clock`'s tick, so an unbatched run times the *clock* and reports ~42 ns instead of 7 ns. The stale 15 ns figure sat between the two. The binary now warns when it is given a batch too small to measure itself.

### The footprint measured the harness, not the shipped code

This row used to report the *benchmark* binary at 35 KB. That was the wrong thing to measure and it proved it: adding JSON output to the benchmark grew it to 52 KB without a single byte changing in `ekf.hpp` or `pinn.hpp`. `bench` and `parity` both link `std::vector`, `<chrono>`, `<fstream>` and a JSON writer that no embedded target would ever flash.

So `cpp/src/footprint.cc` now exists to measure the deployable surface and nothing else — the EKF step, the PINN forward pass, and the baked-in weights — via `make -C cpp footprint`. It comes out at 33 KB stripped, and its **entire** external dependency is `exp`, `tanh` and the stack-protector symbols. There is no allocator symbol in it at all, which is a stronger statement of the zero-allocation claim than the old one: not *"the `operator new` you would find is only the harness's"*, but *"the shipped code cannot allocate, because it never links anything that can."*

Binary size stays toolchain- and platform-dependent; the x86_64 MSYS2 build measures 62 KB / 80 KB for bench / parity.
