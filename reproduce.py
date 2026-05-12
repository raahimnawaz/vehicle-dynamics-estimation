"""Regenerate every figure shown in the README.

Usage:
    python reproduce.py            # synthetic benchmark only
    python reproduce.py --real     # also run real-telemetry pipeline
    python reproduce.py --all      # both (default for CI-style runs)

Outputs land in results/. Synthetic runs are seeded so numbers match the README.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.telemetry import extract_braking_event, load_csv, resample_uniform
from src.estimation.kalman import VehicleEKF
from src.estimation.optimize import estimate
from src.ml.pinn import evaluate_curve, generate_dataset, train_pinn
from src.ml.train import FrictionNet
from src.physics.wheel import mu_pacejka
from src.scenarios.adversarial import biased_sensor, clean_sensor, dropout_sensor, mu_step
from src.scenarios.runner import run_scenario
from src.simulation.realism import add_noise
from src.simulation.run_sim import simulate
from src.visualization.plot import plot

RESULTS = "results"
FIGURES = "figures"


def run_synthetic(seed: int = 0) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    true_mu, true_k, v0, dt = 0.7, 0.02, 30.0, 0.01
    t = np.arange(0, 10, dt)

    v_true = simulate([true_mu, true_k], v0, t, dt)
    v_obs = add_noise(v_true)

    mu_est, k_est = estimate(v_obs, v0, t, dt)
    v_fit = simulate([mu_est, k_est], v0, t, dt)

    ekf = VehicleEKF(mu_init=0.3, v_init=v0)
    for vi in v_obs:
        if vi > 1.0:
            ekf.predict(dt)
            ekf.update(vi)

    ml_mu = None
    weights = os.path.join("models", "friction_net.pth")
    if os.path.exists(weights):
        window_size = 50
        net = FrictionNet(window_size)
        net.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
        net.eval()
        with torch.no_grad():
            ml_mu = net(torch.tensor(v_obs[10:10 + window_size], dtype=torch.float32)).item()

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    plot(t, v_true, v_obs, v_fit)  # writes figures/estimation_results.png
    # mirror into results/ so the canonical reproducible output lives there
    import shutil
    shutil.copyfile(
        os.path.join(FIGURES, "estimation_results.png"),
        os.path.join(RESULTS, "estimation_results.png"),
    )

    print("=== Synthetic benchmark ===")
    print(f"  true mu        = {true_mu:.4f}")
    print(f"  SciPy batch mu = {mu_est:.4f}")
    print(f"  EKF mu         = {ekf.x[1]:.4f}")
    if ml_mu is not None:
        print(f"  NN mu          = {ml_mu:.4f}")

    return {"mu": mu_est, "k": k_est, "ekf_mu": ekf.x[1], "ml_mu": ml_mu}


def run_real(csv_path: str = "data/sample_braking.csv") -> dict:
    t, v = load_csv(csv_path)
    t, v = extract_braking_event(t, v)

    dt = float(np.median(np.diff(t)))
    t_u, v_u = resample_uniform(t, v, dt)
    v0 = float(v_u[0])

    mu_est, k_est = estimate(v_u, v0, t_u, dt)
    v_fit = simulate([mu_est, k_est], v0, t_u, dt)

    # Telemetry is ~10Hz; bump process noise on mu so the filter adapts within
    # the short braking window (synthetic benchmark runs at 100Hz with 10x more
    # samples for the EKF to settle).
    ekf = VehicleEKF(mu_init=0.5, v_init=v0)
    ekf.Q[1, 1] = 1e-2
    mu_track = []
    for vi in v_u:
        if vi > 1.0:
            ekf.predict(dt)
            ekf.update(vi)
        mu_track.append(ekf.x[1])

    os.makedirs(RESULTS, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0b0b0b")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0b0b0b")
        ax.grid(alpha=0.2)

    ax1.scatter(t_u, v_u, s=12, color="#FF3B3B", alpha=0.6, label="Telemetry (CSV)")
    ax1.plot(t_u, v_fit, color="#00FF85", linewidth=2, linestyle="--", label=f"Fit (μ={mu_est:.3f})")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Real-telemetry braking event")
    ax1.legend(facecolor="#111111", edgecolor="white")

    ax2.plot(t_u, mu_track, color="#00E5FF", linewidth=2, label="EKF μ estimate")
    ax2.axhline(mu_est, color="#00FF85", linestyle="--", label=f"Batch μ={mu_est:.3f}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("μ")
    ax2.legend(facecolor="#111111", edgecolor="white")

    out = os.path.join(RESULTS, "real_estimation.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("=== Real telemetry ===")
    print(f"  source         = {csv_path}")
    print(f"  samples        = {len(t_u)}  dt = {dt:.3f}s")
    print(f"  SciPy batch mu = {mu_est:.4f}")
    print(f"  EKF mu (final) = {ekf.x[1]:.4f}")
    print(f"  wrote          = {out}")
    return {"mu": mu_est, "ekf_mu": ekf.x[1]}


def run_adversarial() -> None:
    """Three EKF stress scenarios: mid-run mu change, sensor dropouts, biased sensor."""
    label_step, sched_step = mu_step(mu_before=0.8, mu_after=0.35, t_change=2.0)
    label_drop, sensor_drop = dropout_sensor(rate=0.5, burst_len=60)
    label_bias, sensor_bias = biased_sensor(bias=1.5)
    _, sensor_clean = clean_sensor()

    results = [
        # higher q_mu lets the filter chase the abrupt road-condition change
        run_scenario("mu step (dry -> wet at 2s)", sched_step, sensor_clean, seed=0, q_mu=1e-2),
        run_scenario("sensor dropout bursts", lambda ti: 0.7, sensor_drop, seed=1),
        run_scenario("biased sensor (+1.5 m/s)", lambda ti: 0.7, sensor_bias, seed=2),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0b0b0b")

    for row, r in enumerate(results):
        ax_v, ax_mu = axes[row]
        for ax in (ax_v, ax_mu):
            ax.set_facecolor("#0b0b0b")
            ax.grid(alpha=0.2)

        ax_v.plot(r.t, r.v_true, color="#00E5FF", linewidth=2, label="True v")
        meas_t = r.t[~np.isnan(r.v_meas)]
        meas_v = r.v_meas[~np.isnan(r.v_meas)]
        ax_v.scatter(meas_t, meas_v, s=6, color="#FF3B3B", alpha=0.5, label="Measurement")
        ax_v.set_ylabel("v (m/s)")
        ax_v.set_title(r.label)
        if row == 0:
            ax_v.legend(facecolor="#111", edgecolor="white", fontsize=8)

        ax_mu.plot(r.t, r.mu_true, color="#FFD166", linewidth=2, linestyle="--", label="True μ")
        ax_mu.plot(r.t, r.mu_est, color="#00FF85", linewidth=2, label="EKF μ")
        ax_mu.fill_between(
            r.t,
            r.mu_est - 2 * r.mu_sigma,
            r.mu_est + 2 * r.mu_sigma,
            color="#00FF85",
            alpha=0.15,
            label="±2σ",
        )
        ax_mu.set_ylabel("μ")
        ax_mu.set_ylim(0, 1.1)
        if row == 0:
            ax_mu.legend(facecolor="#111", edgecolor="white", fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout()

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "adversarial_ekf.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("=== Adversarial EKF scenarios ===")
    for r in results:
        # final-window error after the filter has had a chance to settle
        tail = slice(int(0.8 * len(r.t)), None)
        err = float(np.mean(np.abs(r.mu_est[tail] - r.mu_true[tail])))
        print(f"  {r.label:35s}  |mu_est - mu_true|(tail) = {err:.3f}")
    print(f"  wrote = {out}")


def run_pinn(seed: int = 0, epochs: int = 4000) -> None:
    """Train the PINN to recover mu(s) from noisy braking trajectories."""
    ds, meta = generate_dataset(n_runs=8, t_final=4.0, seed=seed)
    print("=== PINN: discovering mu(s) ===")
    print(f"  ground truth   = mu_max=(1 - e^(-Cs)) with mu_max={meta['mu_max']}, C={meta['C']}")
    print(f"  samples        = {meta['n_samples']}   slip range s in {meta['s_range']}")

    net, history = train_pinn(ds, epochs=epochs, seed=seed)
    s, mu_hat = evaluate_curve(net, s_max=0.3)
    mu_truth = mu_pacejka(s, meta["mu_max"], meta["C"])
    in_range = (s >= meta["s_range"][0]) & (s <= meta["s_range"][1])
    max_err = float(np.max(np.abs(mu_hat[in_range] - mu_truth[in_range])))
    mean_err = float(np.mean(np.abs(mu_hat[in_range] - mu_truth[in_range])))

    os.makedirs(RESULTS, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0b0b0b")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0b0b0b")
        ax.grid(alpha=0.2)

    # left: recovered curve vs truth + data scatter
    ax1.plot(s, mu_truth, color="#FFD166", linewidth=2.5, linestyle="--", label="Ground truth μ(s)")
    ax1.plot(s, mu_hat, color="#00FF85", linewidth=2.5, label="PINN μ_θ(s)")
    # implied (s, mu) pairs from data via inverse of dv equation
    mu_implied = (-ds.dv_dt - (0.4 / 1500.0) * ds.v ** 2) / 9.81
    ax1.scatter(ds.s, mu_implied, s=4, color="#FF3B3B", alpha=0.18, label="Data (implied)")
    ax1.set_xlim(0, 0.3)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("slip ratio s")
    ax1.set_ylabel("μ")
    ax1.set_title("Recovered tire curve")
    ax1.legend(facecolor="#111", edgecolor="white", fontsize=9, loc="lower right")

    # right: training loss
    ax2.plot(np.arange(len(history)) * 200, history, color="#00E5FF", linewidth=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss (ODE residual + monotonicity)")
    ax2.set_title("Training")

    plt.tight_layout()
    out = os.path.join(RESULTS, "pinn_recovery.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  max  |d_mu| (in-range) = {max_err:.3f}")
    print(f"  mean |d_mu| (in-range) = {mean_err:.3f}")
    print(f"  wrote = {out}")

    # Save weights so the C++ port has a fixed reference to load.
    os.makedirs("models", exist_ok=True)
    torch.save(net.state_dict(), os.path.join("models", "pinn_mu.pth"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="run real-telemetry pipeline")
    ap.add_argument("--adversarial", action="store_true", help="run adversarial EKF scenarios")
    ap.add_argument("--pinn", action="store_true", help="train the PINN that recovers mu(s)")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--csv", default="data/sample_braking.csv")
    args = ap.parse_args()

    if args.synthetic_only:
        run_synthetic()
        return

    run_synthetic()
    if args.real or args.all:
        run_real(args.csv)
    if args.adversarial or args.all:
        run_adversarial()
    if args.pinn or args.all:
        run_pinn()


if __name__ == "__main__":
    main()
