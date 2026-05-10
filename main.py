import numpy as np
import torch
import os
from src.ml.train import FrictionNet, generate_training_data, train_model
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
    if v_obs[i] > 1.0: 
        ekf.predict(dt)
        ekf.update(v_obs[i])
    
    mu_track.append(ekf.x[1])



# ... (Your EKF loop finishes here) ...

# ---------------------------------------------------------
# ML Neural Network Evaluation (Inference Only!)
# ---------------------------------------------------------
print("\n--- Running ML Prediction ---")

window_size = 50 
ml_model = FrictionNet(window_size)

# We load the brain you already trained and saved using src/ml/train.py
ml_model.load_state_dict(torch.load("models/friction_net.pth", weights_only=True))
ml_model.eval()

# Grab the first 0.5 seconds of your noisy observation data
sensor_window = torch.tensor(v_obs[10:10+window_size], dtype=torch.float32)

# Make the prediction
with torch.no_grad():
    ml_mu = ml_model(sensor_window).item()

# ---------------------------------------------------------
# Final Results & Visualization
# ---------------------------------------------------------
print("\n=== FINAL SYSTEM IDENTIFICATION RESULTS ===")
print(f"True mu:          {true_mu:.4f}")
print(f"SciPy Batch mu:   {mu_est:.4f}")
print(f"EKF Real-Time mu: {ekf.x[1]:.4f}")
print(f"Neural Net mu:    {ml_mu:.4f}")

plot(t, v_true, v_obs, v_fit)

