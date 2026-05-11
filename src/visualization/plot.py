import matplotlib.pyplot as plt
import numpy as np

def plot(t, v_true, v_obs=None, v_fit=None):

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))

    valid = v_true > 0.05

    t_plot = t[valid]
    v_true_plot = v_true[valid]

    ax.plot(
        t_plot,
        v_true_plot,
        color="#00E5FF",
        linewidth=2,
        label="True Dynamics"
    )

    if v_obs is not None:
        v_obs_plot = v_obs[valid]

        ax.scatter(
            t_plot,
            v_obs_plot,
            color="#FF3B3B",
            s=10,
            alpha=0.5,
            label="Sensor Observations"
        )

    if v_fit is not None:
        v_fit_plot = v_fit[valid]

        ax.plot(
            t_plot,
            v_fit_plot,
            color="#00FF85",
            linewidth=2,
            linestyle="--",
            label="Estimated Dynamics"
        )

    ax.set_title("Vehicle Dynamics Estimation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")

    ax.set_xlim(0, 4)
    ax.grid(alpha=0.2)

    ax.legend(facecolor="#111111", edgecolor="white")

    plt.tight_layout()
    plt.savefig("figures/estimation_results.png", dpi=300, bbox_inches="tight")
    plt.show()