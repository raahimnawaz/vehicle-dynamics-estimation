# C++ edge port

Real-time, allocation-free C++ implementations of the [EKF](include/vd/ekf.hpp) and the [PINN forward pass](include/vd/pinn.hpp), targeting Jetson (aarch64 Linux) with code that cross-compiles cleanly to Cortex-M.

## Design choices

| Choice | Why |
|---|---|
| **Header-only library** | Single-translation-unit builds. Trivially vendored into any downstream project. |
| **Hand-rolled 2×2 / 2×1 matrices** instead of Eigen | EKF state is tiny; Eigen's compile-time machinery is overkill and pulls a large header tree. The hand-rolled version is ~70 LOC and is constexpr-friendly. |
| **No heap, no exceptions, no `std::vector` in the hot path** | The same headers drop into a Cortex-M build with no modification. RTOS-ready. |
| **Weights baked into the binary** at compile time | No file I/O at startup, no runtime dependency on PyTorch / ONNX Runtime. Updating the model means re-running `tools/export_weights.py` and recompiling. |
| **Double precision** | Numerical parity with the Python reference is verifiable to ~1e-9. A float32 build is a one-line change if the target platform demands it (Jetson Orin's tensor cores prefer fp16/int8, but pure CPU paths are fastest in fp64 on Cortex-A78AE). |
| **`-O3 -ffast-math -funroll-loops`** | Standard release flags. `-ffast-math` reorders FP ops, which is the only reason parity isn't bit-exact. |

## Build

```bash
make             # builds build/parity and build/bench
make size        # shows stripped binary sizes
make run-bench   # builds + runs the benchmark
```

Cross-compile for Jetson (aarch64-linux-gnu):

```bash
make CXX=aarch64-linux-gnu-g++ TARGET_SUFFIX=-aarch64
```

## Benchmarks (x86_64, MSYS2 UCRT64 g++ 15.1.0)

50,000 timing samples, batch 200, `-O3 -ffast-math -funroll-loops`.

| Op | median | p99 | min | throughput |
|---|---|---|---|---|
| EKF step (predict + update) | **10 ns** | 13 ns | 10 ns | 100 Mops/s |
| PINN forward (1→32→32→1, tanh + sigmoid) | **689 ns** | 1277 ns | 680 ns | 1.45 Mops/s |

| Binary (stripped) | size |
|---|---|
| `bench` | 62 KB |
| `parity` | 80 KB |

Stored as JSON in [`benchmarks/x86_64-msys2-ucrt64.json`](../benchmarks/x86_64-msys2-ucrt64.json). When the Jetson is in hand, run the same `make run-bench` over there and drop the JSON next to it.

## Parity vs Python (max-abs-error on identical inputs)

| Quantity | max\|Δ\| |
|---|---|
| EKF $v$ | 6.7×10⁻⁹ |
| EKF $\mu$ | 2.3×10⁻⁹ |
| EKF $\sigma_v$, $\sigma_\mu$ | 5×10⁻¹¹ — 3×10⁻¹⁰ |
| PINN $\mu_\theta(s)$ | 2.3×10⁻⁷ |

The PINN gap is limited by the Python model running in float32 while the C++ port runs in float64; reductions in the matmul and the weights themselves are quantised, so 2×10⁻⁷ is the algorithmic floor for this comparison.

The EKF gap is the order of magnitude expected from `-ffast-math` FP reordering — no algorithmic divergence.

Reproduce with:

```bash
make parity                      # builds build/parity
python ../tools/parity_check.py  # runs Python reference, diffs against C++
```

## Layout

```
cpp/
├── include/vd/
│   ├── matrix.hpp        # fixed-size 2x2/2x1, no heap
│   ├── ekf.hpp           # joint (v, mu) filter
│   ├── pinn.hpp          # 3-layer MLP forward pass
│   └── pinn_weights.h    # AUTOGEN by tools/export_weights.py
├── src/
│   ├── parity.cc         # runs EKF or PINN on a CSV input
│   └── bench.cc          # latency micro-benchmark
└── Makefile
```

## Porting notes for Cortex-M

The hot paths use only `std::tanh`, `std::exp`, and primitive arithmetic. To run on a Cortex-M4F:

1. Replace `std::tanh` / `std::exp` with the device's math library (e.g. CMSIS-DSP `arm_tanh_f32` / `arm_exp_f32`) — the function signatures match.
2. Build with `-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard`.
3. The PINN stack scratchpad is `2 * H * sizeof(double) = 512 bytes` (H=32). Drops to 256 bytes if you switch to fp32.
4. No code in `vd::Ekf::predict / update` allocates, throws, or calls into the C runtime beyond `sqrt`. Real-time-safe.
