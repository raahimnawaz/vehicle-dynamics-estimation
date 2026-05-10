import numpy as np
from src.simulation.run_sim import simulate

def loss(theta, v_obs, v0, t, dt):

    v_sim = simulate(theta, v0, t, dt)

    return np.mean((v_sim - v_obs)**2)