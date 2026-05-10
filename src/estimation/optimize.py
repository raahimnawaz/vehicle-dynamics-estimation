from scipy.optimize import minimize
from src.estimation.loss import loss

def estimate(v_obs, v0, t, dt):

    x0 = [0.5, 0.02]  # mu, k

    res = minimize(
        loss,
        x0=x0,
        args=(v_obs, v0, t, dt),
        method="Nelder-Mead"
    )

    return res.x