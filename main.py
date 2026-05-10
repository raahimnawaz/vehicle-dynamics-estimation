import numpy as np

from src.simulation.run_sim import simulate
from src.simulation.realism import add_noise
from src.estimation.optimize import estimate
from src.visualization.plot import plot
from src.estimation.kalman import VehicleEKF

true_mu = 0.7
true_k = 0.02

v0 = 30
dt = 0.01
t = np.arange(0, 10, dt)

v_true = simulate([true_mu, true_k], v0, t, dt)
v_obs = add_noise(v_true)

mu_est, k_est = estimate(v_obs, v0, t, dt)
v_fit = simulate([mu_est, k_est], v0, t, dt)

print("True mu:", true_mu, "Estimated mu:", mu_est)
print("True k:", true_k, "Estimated k:", k_est)

ekf = VehicleEKF(mu_init=0.3)

mu_track = []

for i in range(len(t)):
    ekf.predict(dt)
    ekf.update(v_obs[i])
    mu_track.append(ekf.x[1])

print("Final EKF mu:", ekf.x[1])

plot(t, v_true, v_obs, v_fit)