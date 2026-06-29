"""Parity check: run the same inputs through the Python and C++ implementations
and report max-abs-error on EKF state and PINN output.

This is the regression that catches drift between the two ports. Numbers should
agree to ~1e-9 in double precision (no algorithmic difference, only the order
of FP ops which the optimiser may reorder under -ffast-math).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.estimation.kalman import VehicleEKF  # noqa: E402
from src.ml.pinn import MuNet  # noqa: E402


def _run_cpp(binary: str, subcmd: str, input_csv: str, out_csv: str) -> None:
    res = subprocess.run([os.path.abspath(binary), subcmd, input_csv, out_csv],
                         capture_output=True, text=True, check=False)
    if res.returncode != 0:
        print("C++ parity binary failed:", res.stderr)
        sys.exit(res.returncode)


def parity_ekf(binary: str, n_steps: int = 2000, dt: float = 0.01) -> None:
    rng = np.random.default_rng(0)
    true_mu, true_k = 0.7, 0.02
    v = 30.0
    # generate a synthetic noisy trace, store (t, dt, z) per row
    ts = np.arange(n_steps) * dt
    v_true = np.zeros(n_steps)
    v_true[0] = v
    for i in range(1, n_steps):
        a = -true_mu * 9.81 - true_k * v_true[i - 1] ** 2
        v_true[i] = max(v_true[i - 1] + a * dt, 0.0)
    v_obs = v_true + rng.normal(0, 0.25, size=n_steps)

    tmp = tempfile.mkdtemp(prefix="parity_ekf_")
    in_csv  = os.path.join(tmp, "in.csv")
    out_csv = os.path.join(tmp, "out.csv")
    with open(in_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "dt", "z"])
        for t, z in zip(ts, v_obs):
            w.writerow([f"{t:.6f}", f"{dt:.6f}", f"{z:.9e}"])
    _run_cpp(binary, "ekf", in_csv, out_csv)

    # Python EKF parameterises drag as `(rho_cd_a / (2m)) * v^2`; the C++ port
    # uses `k * v^2` directly. Reconcile with rho_cd_a = 2 * m * k_cpp = 60.
    ekf = VehicleEKF(mu_init=0.5, v_init=float(v_obs[0]))
    K_CPP = 0.02
    ekf.rho_cd_a = 2.0 * ekf.m * K_CPP
    ekf.Q[0, 0] = 1e-1
    ekf.Q[1, 1] = 1e-2
    ekf.R[0, 0] = 1.0

    py_v = np.zeros(n_steps)
    py_mu = np.zeros(n_steps)
    py_sv = np.zeros(n_steps)
    py_sm = np.zeros(n_steps)
    for i in range(n_steps):
        ekf.predict(dt)
        ekf.update(v_obs[i])
        py_v[i]  = ekf.x[0]
        py_mu[i] = ekf.x[1]
        py_sv[i] = np.sqrt(ekf.P[0, 0])
        py_sm[i] = np.sqrt(ekf.P[1, 1])

    cpp = np.loadtxt(out_csv, delimiter=",", skiprows=1)
    cpp_v, cpp_mu, cpp_sv, cpp_sm = cpp[:, 1], cpp[:, 2], cpp[:, 3], cpp[:, 4]

    print("EKF parity:")
    print(f"  v     max|diff| = {np.max(np.abs(py_v  - cpp_v )):.3e}")
    print(f"  mu    max|diff| = {np.max(np.abs(py_mu - cpp_mu)):.3e}")
    print(f"  sig_v max|diff| = {np.max(np.abs(py_sv - cpp_sv)):.3e}")
    print(f"  sig_m max|diff| = {np.max(np.abs(py_sm - cpp_sm)):.3e}")
    err_mu = float(np.max(np.abs(py_mu - cpp_mu)))
    if err_mu > 1e-6:
        print(f"  WARNING: mu parity error {err_mu:.3e} exceeds 1e-6")


def parity_pinn(binary: str, weights: str, n: int = 500) -> None:
    s = np.linspace(0.0, 0.3, n)
    tmp = tempfile.mkdtemp(prefix="parity_pinn_")
    in_csv  = os.path.join(tmp, "in.csv")
    out_csv = os.path.join(tmp, "out.csv")
    with open(in_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s"])
        for si in s:
            w.writerow([f"{si:.9e}"])
    _run_cpp(binary, "pinn", in_csv, out_csv)

    net = MuNet()
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()
    with torch.no_grad():
        py_mu = net(torch.tensor(s, dtype=torch.float32)).numpy().astype(np.float64)

    cpp = np.loadtxt(out_csv, delimiter=",", skiprows=1)
    cpp_mu = cpp[:, 1]
    diff = np.abs(py_mu - cpp_mu)
    print("PINN parity:")
    print(f"  mu max|diff| = {np.max(diff):.3e}    mean|diff| = {np.mean(diff):.3e}")
    if np.max(diff) > 1e-5:
        print(f"  WARNING: PINN parity max-diff {np.max(diff):.3e} exceeds 1e-5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="cpp/build/parity.exe")
    ap.add_argument("--weights", default="models/pinn_mu.pth")
    args = ap.parse_args()
    if not os.path.exists(args.binary):
        # try linux name
        alt = args.binary[:-4] if args.binary.endswith(".exe") else args.binary
        if os.path.exists(alt):
            args.binary = alt
        else:
            print(f"parity binary not found at {args.binary}; build with `make -C cpp parity`")
            sys.exit(2)
    parity_ekf(args.binary)
    parity_pinn(args.binary, args.weights)


if __name__ == "__main__":
    main()
