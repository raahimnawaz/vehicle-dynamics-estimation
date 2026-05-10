from scipy.optimize import minimize
from src.estimation.loss import loss

def recover_parameters(v_obs, v0, t, dt):

    guess = [0.5, 0.01]

    result = minimize(
        loss,
        x0=guess,
        args=(v_obs, v0, t, dt),
        method="Nelder-Mead"
    )

    return result.x