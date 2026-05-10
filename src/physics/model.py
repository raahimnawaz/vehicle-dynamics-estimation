def dvdt(v, m, mu, g, k):

    v = max(v, 0.0)

    return -mu * g - k * v**2