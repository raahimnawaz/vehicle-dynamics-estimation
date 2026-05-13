"""Render the Python-vs-C++ latency comparison figure used in the README.

Reads benchmarks/x86_64-python.json and benchmarks/x86_64-msys2-ucrt64.json,
writes results/bench_python_vs_cpp.png.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(p: str) -> dict:
    with open(p) as f:
        return json.load(f)


def main() -> None:
    py  = _load("benchmarks/x86_64-python.json")
    cpp = _load("benchmarks/x86_64-msys2-ucrt64.json")

    ops = ("EKF step", "PINN forward")
    py_median  = [py["results"]["ekf_step_ns"]["median"],
                  py["results"]["pinn_forward_ns"]["median"]]
    cpp_median = [cpp["results"]["ekf_step_ns"]["median"],
                  cpp["results"]["pinn_forward_ns"]["median"]]

    py_p99  = [py["results"]["ekf_step_ns"]["p99"],
               py["results"]["pinn_forward_ns"]["p99"]]
    cpp_p99 = [cpp["results"]["ekf_step_ns"]["p99"],
               cpp["results"]["pinn_forward_ns"]["p99"]]

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0b0b0b")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0b0b0b")
        ax.grid(True, which="both", alpha=0.2)

    # --- Left: log-scale per-op latency (Python vs C++) --------------------
    x = np.arange(len(ops))
    w = 0.35
    bars_py  = ax1.bar(x - w / 2, py_median,  w, color="#FF6B9D", label="Python (median)",
                       yerr=[[0, 0], [v - m for v, m in zip(py_p99, py_median)]],
                       capsize=4, ecolor="#FFAEC9")
    bars_cpp = ax1.bar(x + w / 2, cpp_median, w, color="#00FF85", label="C++ (median)",
                       yerr=[[0, 0], [v - m for v, m in zip(cpp_p99, cpp_median)]],
                       capsize=4, ecolor="#88FFC2")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ops)
    ax1.set_ylabel("per-op latency (ns, log scale)")
    ax1.set_title("Per-op latency: Python vs C++")
    ax1.legend(facecolor="#111", edgecolor="white")

    for bar, v in zip(bars_py, py_median):
        ax1.text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:,}",
                 ha="center", va="bottom", fontsize=9, color="#FFAEC9")
    for bar, v in zip(bars_cpp, cpp_median):
        ax1.text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:,}",
                 ha="center", va="bottom", fontsize=9, color="#88FFC2")

    # --- Right: speedup factor --------------------------------------------
    speedups = [p / c for p, c in zip(py_median, cpp_median)]
    bars = ax2.bar(ops, speedups, color="#00E5FF", width=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("speedup factor (×)")
    ax2.set_title("C++ port speedup over Python reference")
    for bar, v in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2, v * 1.1, f"{v:,.0f}×",
                 ha="center", va="bottom", fontsize=12, color="white")
    ax2.set_ylim(1, max(speedups) * 3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "bench_python_vs_cpp.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#0b0b0b")
    plt.close(fig)

    print("Python median latencies (ns):", py_median)
    print("C++    median latencies (ns):", cpp_median)
    print("Speedups:", [f"{s:.0f}x" for s in speedups])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
