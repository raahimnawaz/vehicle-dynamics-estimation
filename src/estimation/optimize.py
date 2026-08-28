from scipy.optimize import minimize
from src.estimation.loss import loss
from src.physics.wheel import K_DRAG

def estimate(v_obs, v0, t, dt):

    # (mu, k). Seed k at the physical drag coefficient; the previous seed of
    # 0.02 was 75x too large and forced the simplex to cross two orders of
    # magnitude before reaching the basin.
    x0 = [0.5, K_DRAG]

    res = minimize(
        loss,
        x0=x0,
        args=(v_obs, v0, t, dt),
        method="Nelder-Mead"
    )

    return res.x