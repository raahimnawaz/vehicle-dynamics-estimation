"""Latency benchmark for the Python EKF + PINN, mirror of cpp/src/bench.cc.

Designed to be apples-to-apples with the C++ benchmark so the README's
Python-vs-C++ table is honest:
  - Per-op latency (median, p99, min), measured with time.perf_counter_ns
  - Per-op throughput (Mops/s)
  - Same iteration counts as the C++ harness by default
  - JSON output in the same schema as benchmarks/x86_64-msys2-ucrt64.json

To avoid the per-call Python overhead drowning out the math itself, we time
*batches* of operations and report per-op timings.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import date

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.estimation.kalman import VehicleEKF  # noqa: E402
from src.ml.pinn import MuNet  # noqa: E402


def _summary(samples: list[int]) -> dict:
    samples = sorted(samples)
    n = len(samples)
    return {
        "median": samples[n // 2],
        "p99":    samples[min(n - 1, int(0.99 * n))],
        "min":    samples[0],
        "max":    samples[-1],
        "throughput_mops": 1e3 / samples[n // 2] if samples[n // 2] > 0 else 0.0,
    }


def bench_ekf(n_iters: int, batch: int) -> dict:
    ekf = VehicleEKF(mu_init=0.5, v_init=30.0)
    ekf.rho_cd_a = 60.0
    ekf.Q[0, 0] = 1e-1
    ekf.Q[1, 1] = 1e-2
    ekf.R[0, 0] = 1.0

    # warm up
    for _ in range(200):
        ekf.predict(0.01)
        ekf.update(29.5)

    samples: list[int] = []
    sink = 0.0
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        for b in range(batch):
            ekf.predict(0.01)
            ekf.update(29.5 + (b % 7) * 0.01)
            sink += float(ekf.x[1])
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) // batch)
    _ = sink
    return _summary(samples)


def bench_pinn(n_iters: int, batch: int, weights: str) -> dict:
    net = MuNet()
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()

    sink = 0.0
    with torch.no_grad():
        for _ in range(200):
            sink += float(net(torch.tensor([0.1])).item())

    samples: list[int] = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter_ns()
            for b in range(batch):
                s = 0.001 + (b % 300) / 1000.0
                sink += float(net(torch.tensor([s])).item())
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0) // batch)
    _ = sink
    return _summary(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters", type=int, default=20000)
    ap.add_argument("--batch",   type=int, default=200)
    ap.add_argument("--weights", default="models/pinn_mu.pth")
    ap.add_argument("--out",     default="benchmarks/x86_64-python.json")
    args = ap.parse_args()

    ekf  = bench_ekf(args.n_iters, args.batch)
    pinn = bench_pinn(args.n_iters, args.batch, args.weights)

    data = {
        "host": {
            "os": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "interpreter": f"CPython {platform.python_version()}",
        "iterations": args.n_iters,
        "batch": args.batch,
        "results": {
            "ekf_step_ns": ekf,
            "pinn_forward_ns": pinn,
        },
        "date": date.today().isoformat(),
        "notes": (
            "Per-op timings; batch=200 amortises perf_counter resolution. "
            "PINN inference here is a single-sample forward pass through a "
            "torch.nn.Sequential; PyTorch's per-call dispatch overhead "
            "dominates compared to the ~1 KB hand-rolled C++ path."
        ),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    print("=== Python benchmark ===")
    print(f"  EKF step    : median={ekf['median']:>6d} ns  p99={ekf['p99']:>6d} ns  "
          f"min={ekf['min']:>6d} ns  ({ekf['throughput_mops']:.2f} Mops/s)")
    print(f"  PINN forward: median={pinn['median']:>6d} ns  p99={pinn['p99']:>6d} ns  "
          f"min={pinn['min']:>6d} ns  ({pinn['throughput_mops']:.2f} Mops/s)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
