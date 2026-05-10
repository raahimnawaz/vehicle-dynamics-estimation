import numpy as np

class VehicleEKF:

    def __init__(self, mu_init=0.5):

        self.x = np.array([30.0, mu_init])  # [v, mu]
        self.P = np.eye(2) * 0.1

        self.g = 9.81
        self.k = 0.02

    def predict(self, dt):

        v, mu = self.x

        dv = -mu * self.g - self.k * v**2

        self.x[0] = v + dv * dt

        self.P += np.eye(2) * 0.01

        return self.x

    def update(self, z):

        H = np.array([[1, 0]])
        R = np.array([[0.5]])

        y = np.array([z]) - H @ self.x.reshape(-1,1)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y).flatten()

        self.P = (np.eye(2) - K @ H) @ self.P

        return self.x