import numpy as np

def add_noise(v, std=0.25):
    return v + np.random.normal(0, std, size=len(v))