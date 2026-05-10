import matplotlib.pyplot as plt

def plot(t, v_true, v_obs=None, v_fit=None):

    plt.style.use("dark_background")

    plt.figure(figsize=(10,5))

    plt.plot(t, v_true, color="#00E5FF", label="True")

    if v_obs is not None:
        plt.plot(t, v_obs, color="#FF3B3B", alpha=0.5, label="Observed")

    if v_fit is not None:
        plt.plot(t, v_fit, color="#00FF85", label="Estimated")

    plt.legend()
    plt.grid(alpha=0.2)

    plt.xlabel("Time")
    plt.ylabel("Velocity")

    plt.title("Vehicle Dynamics Estimation")

    plt.show()