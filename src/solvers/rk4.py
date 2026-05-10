def rk4_step(f, v, dt, *args):
    k1 = f(v, *args)
    k2 = f(v + 0.5 * dt * k1, *args)
    k3 = f(v + 0.5 * dt * k2, *args)
    k4 = f(v + dt * k3, *args)

    return v + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)