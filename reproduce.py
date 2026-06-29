"""Regenerate every figure shown in the README.

Usage:
    python reproduce.py            # synthetic benchmark only
    python reproduce.py --real     # also run real-telemetry pipeline
    python reproduce.py --all      # everything

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
from src.ml.pinn import (
    PacejkaNet,
    evaluate_curve,
    evaluate_curve_2d,
    evaluate_pacejka_curve,
    generate_dataset,
    generate_dataset_braking,
    train_pacejka,
    train_pinn,
    train_pinn_2d,
)
from src.ml.train import FrictionNet
from src.physics.wheel import DEFAULTS, PACEJKA_DRY, mu_pacejka, pacejka_peak
DEFAULTS_K = DEFAULTS["k"]
DEFAULTS_M = DEFAULTS["m"]
from src.scenarios.adversarial import biased_sensor, clean_sensor, dropout_sensor, mu_step
from src.scenarios.mismatch import EFFECTS, run_sweep
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
    plot(t, v_true, v_obs, v_fit)
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
    ax1.plot(t_u, v_fit, color="#00FF85", linewidth=2, linestyle="--", label=f"Fit (mu={mu_est:.3f})")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Real-telemetry braking event")
    ax1.legend(facecolor="#111111", edgecolor="white")

    ax2.plot(t_u, mu_track, color="#00E5FF", linewidth=2, label="EKF mu estimate")
    ax2.axhline(mu_est, color="#00FF85", linestyle="--", label=f"Batch mu={mu_est:.3f}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("mu")
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
    label_step, sched_step = mu_step(mu_before=0.8, mu_after=0.35, t_change=2.0)
    label_drop, sensor_drop = dropout_sensor(rate=0.5, burst_len=60)
    label_bias, sensor_bias = biased_sensor(bias=1.5)
    _, sensor_clean = clean_sensor()

    results = [
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

        ax_mu.plot(r.t, r.mu_true, color="#FFD166", linewidth=2, linestyle="--", label="True mu")
        ax_mu.plot(r.t, r.mu_est, color="#00FF85", linewidth=2, label="EKF mu")
        ax_mu.fill_between(
            r.t,
            r.mu_est - 2 * r.mu_sigma,
            r.mu_est + 2 * r.mu_sigma,
            color="#00FF85",
            alpha=0.15,
            label="+/-2 sigma",
        )
        ax_mu.set_ylabel("mu")
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
        tail = slice(int(0.8 * len(r.t)), None)
        err = float(np.mean(np.abs(r.mu_est[tail] - r.mu_true[tail])))
        print(f"  {r.label:35s}  |mu_est - mu_true|(tail) = {err:.3f}")
    print(f"  wrote = {out}")


def run_pinn(seed: int = 0, epochs: int = 6000) -> None:
    """Train BOTH the free-form PINN (MuNet) and the grey-box PacejkaNet on
    the same Pacejka-truth dataset, then plot them side by side. The figure
    is the cost-of-priors story:

      * MuNet with a concavity prior recovers the shape but not the exact
        peak position -- the cost of being function-free.
      * PacejkaNet is a 4-scalar parametric fit; it nails the curve.
    """
    ds, meta = generate_dataset(n_runs=16, t_final=4.0, seed=seed)
    s_peak, mu_peak = pacejka_peak(**PACEJKA_DRY)
    print("=== PINN: discovering Pacejka mu(s) ===")
    print(f"  ground truth   = Pacejka magic formula, params={meta['params']}")
    print(f"                   true peak: s={s_peak:.3f}, mu={mu_peak:.3f}")
    print(f"  samples        = {meta['n_samples']}   slip range {meta['s_range']}")

    # --- Free-form PINN -------------------------------------------------
    net_mu, hist_mu = train_pinn(ds, epochs=epochs, seed=seed)
    s, mu_hat = evaluate_curve(net_mu, s_max=0.3)
    mu_truth = mu_pacejka(s, **PACEJKA_DRY)
    in_range = (s >= meta["s_range"][0]) & (s <= meta["s_range"][1])
    mu_max_err  = float(np.max(np.abs(mu_hat[in_range] - mu_truth[in_range])))
    mu_mean_err = float(np.mean(np.abs(mu_hat[in_range] - mu_truth[in_range])))
    i_hat = int(np.argmax(mu_hat))
    s_hat_peak, mu_hat_peak = float(s[i_hat]), float(mu_hat[i_hat])

    # --- Grey-box parametric Pacejka ------------------------------------
    net_pj, hist_pj = train_pacejka(ds, epochs=3000, seed=seed)
    _, mu_hat_pj = evaluate_pacejka_curve(net_pj, s_max=0.3)
    pj_max_err  = float(np.max(np.abs(mu_hat_pj[in_range] - mu_truth[in_range])))
    pj_mean_err = float(np.mean(np.abs(mu_hat_pj[in_range] - mu_truth[in_range])))
    pj_params = net_pj.params_dict()
    i_pj = int(np.argmax(mu_hat_pj))
    s_pj_peak, mu_pj_peak = float(s[i_pj]), float(mu_hat_pj[i_pj])

    # --- Figure ---------------------------------------------------------
    os.makedirs(RESULTS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0b0b0b")
    for ax in axes:
        ax.set_facecolor("#0b0b0b")
        ax.grid(alpha=0.2)

    mu_implied = (-ds.dv_dt - (DEFAULTS_K / DEFAULTS_M) * ds.v ** 2) / 9.81

    # Panel 1: function-free PINN
    ax1 = axes[0]
    ax1.scatter(ds.s, mu_implied, s=4, color="#FF3B3B", alpha=0.15,
                label="Data (implied mu)")
    ax1.plot(s, mu_truth, color="#FFD166", linewidth=2.5, linestyle="--",
             label="Ground truth Pacejka")
    ax1.plot(s, mu_hat, color="#00FF85", linewidth=2.5,
             label="MuNet (function-free)")
    ax1.scatter([s_peak], [mu_peak], s=80, color="#FFD166", marker="o",
                edgecolor="white", zorder=5,
                label=f"True peak ({s_peak:.3f}, {mu_peak:.3f})")
    ax1.scatter([s_hat_peak], [mu_hat_peak], s=80, color="#00FF85", marker="s",
                edgecolor="white", zorder=5,
                label=f"MuNet peak ({s_hat_peak:.3f}, {mu_hat_peak:.3f})")
    ax1.set_xlim(0, 0.3)
    ax1.set_ylim(-0.1, 1.15)
    ax1.set_xlabel("slip ratio s")
    ax1.set_ylabel("mu")
    ax1.set_title(f"Function-free PINN  (mean |d_mu| = {mu_mean_err:.3f})")
    ax1.legend(facecolor="#111", edgecolor="white", fontsize=7, loc="lower right")

    # Panel 2: grey-box parametric Pacejka
    ax2 = axes[1]
    ax2.scatter(ds.s, mu_implied, s=4, color="#FF3B3B", alpha=0.15,
                label="Data (implied mu)")
    ax2.plot(s, mu_truth, color="#FFD166", linewidth=2.5, linestyle="--",
             label="Ground truth Pacejka")
    ax2.plot(s, mu_hat_pj, color="#00E5FF", linewidth=2.5,
             label="PacejkaNet (grey-box)")
    ax2.scatter([s_peak], [mu_peak], s=80, color="#FFD166", marker="o",
                edgecolor="white", zorder=5)
    ax2.scatter([s_pj_peak], [mu_pj_peak], s=80, color="#00E5FF", marker="s",
                edgecolor="white", zorder=5,
                label=f"Fit peak ({s_pj_peak:.3f}, {mu_pj_peak:.3f})")
    ax2.set_xlim(0, 0.3)
    ax2.set_ylim(-0.1, 1.15)
    ax2.set_xlabel("slip ratio s")
    ax2.set_title(f"Grey-box (B,C,D,E)  (mean |d_mu| = {pj_mean_err:.3f})")
    pj_text = (f"B={pj_params['B']:.2f}  "
               f"C={pj_params['C']:.2f}\n"
               f"D={pj_params['D']:.3f}  "
               f"E={pj_params['E']:.3f}")
    ax2.text(0.95, 0.05, pj_text, transform=ax2.transAxes,
             ha="right", va="bottom", color="#00E5FF",
             fontsize=9, family="monospace")
    ax2.legend(facecolor="#111", edgecolor="white", fontsize=7, loc="lower right")

    # Panel 3: training curves on shared axes
    ax3 = axes[2]
    ax3.plot(np.arange(len(hist_mu)) * 200, hist_mu, color="#00FF85",
             linewidth=2, label="MuNet")
    ax3.plot(np.arange(len(hist_pj)) * 200, hist_pj, color="#00E5FF",
             linewidth=2, label="PacejkaNet")
    ax3.set_yscale("log")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("loss")
    ax3.set_title("Training")
    ax3.legend(facecolor="#111", edgecolor="white", fontsize=8)

    plt.tight_layout()
    out = os.path.join(RESULTS, "pinn_recovery.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  MuNet      : mean |d_mu| = {mu_mean_err:.3f}  max = {mu_max_err:.3f}  "
          f"peak (s, mu) = ({s_hat_peak:.3f}, {mu_hat_peak:.3f})")
    print(f"  PacejkaNet : mean |d_mu| = {pj_mean_err:.3f}  max = {pj_max_err:.3f}  "
          f"peak (s, mu) = ({s_pj_peak:.3f}, {mu_pj_peak:.3f})")
    print(f"               recovered params  "
          f"B={pj_params['B']:.2f}  C={pj_params['C']:.2f}  "
          f"D={pj_params['D']:.3f}  E={pj_params['E']:.3f}")
    print(f"               truth  params     "
          f"B={PACEJKA_DRY['B']:.2f}  C={PACEJKA_DRY['C']:.2f}  "
          f"D={PACEJKA_DRY['D']:.3f}  E={PACEJKA_DRY['E']:.3f}")
    print(f"  wrote = {out}")

    os.makedirs("models", exist_ok=True)
    torch.save(net_mu.state_dict(), os.path.join("models", "pinn_mu.pth"))
    torch.save(net_pj.state_dict(), os.path.join("models", "pacejka_net.pth"))


def run_pinn_brake(seed: int = 0, epochs: int = 5000) -> None:
    """Train the 2D PINN that factorises mu_eff(s, p) = mu(s) * ramp(p)."""
    ds, meta = generate_dataset_braking(n_runs=12, t_final=4.0, seed=seed)
    print("=== PINN (2D, brake-aware): factorising mu(s)*ramp(p) ===")
    print(f"  ground truth   = Pacejka * (1 - exp(-t/tau)), tau ~ U{meta['tau_range']}")
    print(f"  samples        = {meta['n_samples']}")

    net, history = train_pinn_2d(ds, epochs=epochs, seed=seed)
    s, mu_hat, ramp_hat = evaluate_curve_2d(net, s_max=0.3)
    mu_truth = mu_pacejka(s, **PACEJKA_DRY)
    # ramp(p) = p (identity) is the "ground truth" because the true model is
    # mu_eff = mu(s) * pressure_factor with pressure_factor in [0,1].
    p = np.linspace(0.0, 1.0, len(ramp_hat))
    ramp_err = float(np.mean(np.abs(ramp_hat - p)))

    in_range = (s >= 0.02) & (s <= 0.3)
    mu_err = float(np.mean(np.abs(mu_hat[in_range] - mu_truth[in_range])))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0b0b0b")
    for ax in axes:
        ax.set_facecolor("#0b0b0b")
        ax.grid(alpha=0.2)

    axes[0].plot(s, mu_truth, color="#FFD166", linewidth=2.5, linestyle="--",
                 label="Pacejka truth")
    axes[0].plot(s, mu_hat, color="#00FF85", linewidth=2.5,
                 label="PINN-B mu_theta(s)")
    axes[0].set_xlabel("slip ratio s")
    axes[0].set_ylabel("mu")
    axes[0].set_title("Recovered tire curve (factor 1)")
    axes[0].legend(facecolor="#111", edgecolor="white", fontsize=8)

    axes[1].plot(p, p, color="#FFD166", linewidth=2.5, linestyle="--",
                 label="Ground truth ramp(p) = p")
    axes[1].plot(p, ramp_hat, color="#FF6B9D", linewidth=2.5,
                 label="PINN-B ramp_theta(p)")
    axes[1].set_xlabel("normalised brake pressure p")
    axes[1].set_ylabel("ramp(p)")
    axes[1].set_title("Recovered brake-force factor (factor 2)")
    axes[1].legend(facecolor="#111", edgecolor="white", fontsize=8)

    axes[2].plot(np.arange(len(history)) * 200, history, color="#00E5FF", linewidth=2)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("loss")
    axes[2].set_title("Training")

    plt.tight_layout()
    out = os.path.join(RESULTS, "pinn_brake_recovery.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  mean |d_mu(s)|        = {mu_err:.3f}")
    print(f"  mean |ramp(p) - p|    = {ramp_err:.3f}")
    print(f"  wrote = {out}")

    os.makedirs("models", exist_ok=True)
    torch.save(net.state_dict(), os.path.join("models", "pinn_mu_2d.pth"))


def run_mismatch() -> None:
    """Model-mismatch sweep. Generates three figures:

      * mismatch_per_method.png  -- per-method degradation curves
      * mismatch_heatmap.png     -- worst-case heatmap
      * mismatch_trajectories.png -- nominal vs heavy-mismatch trajectory overlay
      * mismatch_brake_curve.png -- brake-ramp degradation (the headline plot
                                    showing what fixing the PINN buys you)
    """
    cells = run_sweep()
    methods = sorted({c.method for c in cells},
                     key=lambda x: ["Batch", "EKF", "NN", "PINN", "PINN-B"].index(x.split()[0]))
    effect_labels = {e.name: e.label for e in EFFECTS}
    effect_order = [e.name for e in EFFECTS]

    plt.style.use("dark_background")
    colours = {"grade": "#FFD166", "headwind": "#00E5FF", "brake": "#FF6B9D"}
    method_colours = {
        "Batch (SciPy)":         "#00FF85",
        "EKF":                   "#FFD166",
        "NN (FrictionNet)":      "#FF3B3B",
        "PINN":                  "#00E5FF",
        "PINN-B (brake-aware)":  "#FF6B9D",
    }

    # --- Panel 1: per-method degradation curves ----------------------------
    fig1, axes = plt.subplots(1, len(methods), figsize=(3.5 * len(methods), 4.5),
                              sharey=True)
    if len(methods) == 1:
        axes = [axes]
    fig1.patch.set_facecolor("#0b0b0b")
    for ax, method in zip(axes, methods):
        ax.set_facecolor("#0b0b0b")
        ax.grid(alpha=0.2)
        for effect_name in effect_order:
            effect_def = next(e for e in EFFECTS if e.name == effect_name)
            rel = [c for c in cells if c.method == method and c.effect == effect_name]
            rel.sort(key=lambda c: c.intensity)
            i_max = max(effect_def.intensities)
            xs = [(c.intensity / i_max) if i_max > 0 else 0 for c in rel]
            ys = [c.rmse_norm * 100 for c in rel]
            ax.plot(xs, ys, marker="o", linewidth=2,
                    color=colours[effect_name], label=effect_labels[effect_name])
        ax.set_title(method, fontsize=10)
        ax.set_xlabel("normalised severity")
        ax.set_xlim(-0.05, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("trajectory RMSE / v0 (%)")
            ax.legend(facecolor="#111", edgecolor="white", fontsize=7)
    fig1.suptitle("Degradation under each unmodeled effect "
                  "(x: 0 = lightest, 1 = heaviest)", color="white")
    plt.tight_layout()

    os.makedirs(RESULTS, exist_ok=True)
    out1 = os.path.join(RESULTS, "mismatch_per_method.png")
    fig1.savefig(out1, dpi=200, bbox_inches="tight", facecolor="#0b0b0b")
    plt.close(fig1)

    # --- Panel 2: heatmap at maximum mismatch intensity --------------------
    fig2, ax = plt.subplots(figsize=(9, 5))
    fig2.patch.set_facecolor("#0b0b0b")
    ax.set_facecolor("#0b0b0b")

    grid = np.zeros((len(methods), len(effect_order)))
    for i, m in enumerate(methods):
        for j, e in enumerate(effect_order):
            relevant = [c for c in cells if c.method == m and c.effect == e]
            high = max(relevant, key=lambda c: c.intensity)
            grid[i, j] = high.rmse_norm * 100

    im = ax.imshow(grid, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(effect_order)))
    ax.set_xticklabels([effect_labels[e] for e in effect_order])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(effect_order)):
            ax.text(j, i, f"{grid[i, j]:.1f}%", ha="center", va="center",
                    color="white" if grid[i, j] > grid.max() / 2 else "black",
                    fontsize=11)
    ax.set_title("Trajectory RMSE / v0 at maximum mismatch intensity")
    cbar = fig2.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="white")
    out2 = os.path.join(RESULTS, "mismatch_heatmap.png")
    plt.tight_layout()
    fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor="#0b0b0b")
    plt.close(fig2)

    # --- Panel 3: trajectory overlay at nominal vs worst grade -------------
    methods_for_overlay = [m for m in methods if m != "PINN-B (brake-aware)"]
    fig3, axes = plt.subplots(len(methods_for_overlay), 2,
                              figsize=(11, 2 * len(methods_for_overlay)),
                              sharex=True)
    fig3.patch.set_facecolor("#0b0b0b")

    def first(method, effect, target_intensity):
        return min((c for c in cells if c.method == method and c.effect == effect),
                   key=lambda c: abs(c.intensity - target_intensity))

    for row, method in enumerate(methods_for_overlay):
        for col, (eff_name, intensity) in enumerate([("grade", 0.0), ("grade", 0.12)]):
            ax = axes[row, col] if len(methods_for_overlay) > 1 else axes[col]
            ax.set_facecolor("#0b0b0b")
            ax.grid(alpha=0.2)
            c = first(method, eff_name, intensity)
            ax.scatter(c.t, c.v_obs, s=4, color="#FF3B3B", alpha=0.25, label="noisy obs")
            ax.plot(c.t, c.v_truth, color="#00E5FF", linewidth=2, label="truth (Pacejka)")
            ax.plot(c.t, c.v_pred, color="#00FF85", linewidth=2, linestyle="--",
                    label=f"{method} (rmse={c.rmse:.2f})")
            if col == 0:
                ax.set_ylabel("v (m/s)")
            if row == 0:
                ax.set_title("nominal (grade=0)" if intensity == 0.0 else "grade = 0.12 rad")
            if row == len(methods_for_overlay) - 1:
                ax.set_xlabel("time (s)")
            ax.legend(facecolor="#111", edgecolor="white", fontsize=7, loc="upper right")
    out3 = os.path.join(RESULTS, "mismatch_trajectories.png")
    plt.tight_layout()
    fig3.savefig(out3, dpi=200, bbox_inches="tight", facecolor="#0b0b0b")
    plt.close(fig3)

    # --- Panel 4: brake-ramp degradation curve (the headline 'fix') --------
    fig4, ax = plt.subplots(figsize=(9, 5))
    fig4.patch.set_facecolor("#0b0b0b")
    ax.set_facecolor("#0b0b0b")
    ax.grid(alpha=0.2)

    brake_intensities = [i for i in EFFECTS[2].intensities]
    for method in methods:
        rel = [c for c in cells if c.method == method and c.effect == "brake"]
        rel.sort(key=lambda c: c.intensity)
        xs = [c.intensity for c in rel]
        ys = [c.rmse_norm * 100 for c in rel]
        ax.plot(xs, ys, marker="o", linewidth=2,
                color=method_colours.get(method, "#ffffff"), label=method)
    ax.set_xlabel("brake-ramp tau (s)  -- larger = slower hydraulic response")
    ax.set_ylabel("trajectory RMSE / v0  (%)")
    ax.set_title("Brake-ramp degradation: PINN-B holds flat where the others blow up")
    ax.legend(facecolor="#111", edgecolor="white", fontsize=9, loc="upper left")
    out4 = os.path.join(RESULTS, "mismatch_brake_curve.png")
    plt.tight_layout()
    fig4.savefig(out4, dpi=200, bbox_inches="tight", facecolor="#0b0b0b")
    plt.close(fig4)

    print("=== Model-mismatch study ===")
    for m in methods:
        nominal = [c for c in cells if c.method == m and c.intensity in (0.0, 0.01)]
        worst_m = [c for c in cells if c.method == m]
        rmse_lo = float(np.mean([c.rmse for c in nominal]))
        rmse_hi = float(np.max([c.rmse for c in worst_m]))
        print(f"  {m:25s}  rmse(nominal)={rmse_lo:.2f}  rmse(worst)={rmse_hi:.2f}")

    # Per-intensity brake table -- the numbers we cite in the README.
    print("\n  --- Brake-ramp degradation table (RMSE / v0, %) ---")
    header = "  tau (s)  " + "  ".join(f"{m:>22s}" for m in methods)
    print(header)
    for tau in brake_intensities:
        row = f"  {tau:>7.2f}  "
        for m in methods:
            c = next((c for c in cells if c.method == m and c.effect == "brake"
                      and abs(c.intensity - tau) < 1e-6), None)
            row += f"  {c.rmse_norm * 100:>20.2f}%" if c else f"  {'-':>22s}"
        print(row)

    print(f"\n  wrote {out1}")
    print(f"  wrote {out2}")
    print(f"  wrote {out3}")
    print(f"  wrote {out4}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--pinn", action="store_true")
    ap.add_argument("--pinn-brake", action="store_true",
                    help="train the 2D brake-aware PINN")
    ap.add_argument("--mismatch", action="store_true")
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
    if args.pinn_brake or args.all:
        run_pinn_brake()
    if args.mismatch or args.all:
        run_mismatch()


if __name__ == "__main__":
    main()
