import numpy as np
from src.physics.model import dvdt
from src.solvers.rk4 import rk4_step

def simulate(theta, v0, t, dt, m=1500, g=9.81):

    mu, k = theta

    v = v0
    out = []

    for _ in t:
        out.append(v)
        v = rk4_step(dvdt, v, dt, m, mu, g, k)
        v = max(min(v, 100), 0)
    
    return np.array(out)