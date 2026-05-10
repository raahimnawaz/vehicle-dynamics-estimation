import numpy as np

class VehicleEKF:
    def __init__(self, mu_init=0.3, v_init=30.0):
        # State vector: [velocity, friction_mu]
        self.x = np.array([v_init, mu_init], dtype=float)
        
        # Covariance Matrix (P)
        # We put a high variance on mu (e.g., 1.0) because our initial guess is uncertain
        self.P = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
        
        # Process Noise (Q)
        # Small noise on mu allows it to "drift" and update over time
        self.Q = np.array([[0.1, 0.0],
                           [0.0, 1e-4]])
        
        # Measurement Noise (R) - Adjust based on your sensor noise variance
        self.R = np.array([[1.0]]) 
        
        # Measurement Matrix (H) - We only observe velocity
        self.H = np.array([[1.0, 0.0]])
        
        # Constants
        self.g = 9.81
        self.m = 1500.0  # Mass (adjust to match your simulation)
        self.rho_cd_a = 0.02 * 2 * self.m # Derived from your true_k = 0.02

    def predict(self, dt):
        v = self.x[0]
        mu = self.x[1]
        
        # 1. Predict State
        # a = -mu * g - k * v^2
        a = -mu * self.g - (self.rho_cd_a / (2 * self.m)) * (v**2)
        v_next = v + a * dt
        
        self.x = np.array([v_next, mu])
        
        # 2. Compute Jacobian (F)
        df_dv = 1.0 - dt * (self.rho_cd_a / self.m) * v
        df_dmu = -self.g * dt
        
        F = np.array([[df_dv, df_dmu],
                      [0.0,   1.0]])
        
        # 3. Predict Covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, v_obs):
        # 1. Compute Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 2. Compute Error (Innovation)
        y = v_obs - self.x[0] 
        
        # 3. Update State
        self.x = self.x + (K @ np.array([y])).flatten()
        
        # 4. Update Covariance
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P