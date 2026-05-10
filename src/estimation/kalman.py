import numpy as np

class VehicleEKF:
    def __init__(self, mu_init=0.3, v_init=30.0):
        self.x = np.array([v_init, mu_init], dtype=float)
        
        
        self.P = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
        
        
        self.Q = np.array([[0.1, 0.0],
                           [0.0, 1e-4]])
        
        
        self.R = np.array([[1.0]]) 
        
        self.H = np.array([[1.0, 0.0]])
        
        # 
        self.g = 9.81
        self.m = 1500.0  
        self.rho_cd_a = 0.02 * 2 * self.m 
    def predict(self, dt):
        v = self.x[0]
        mu = self.x[1]
        
        a = -mu * self.g - (self.rho_cd_a / (2 * self.m)) * (v**2)
        v_next = v + a * dt
        
        self.x = np.array([v_next, mu])
        
        df_dv = 1.0 - dt * (self.rho_cd_a / self.m) * v
        df_dmu = -self.g * dt
        
        F = np.array([[df_dv, df_dmu],
                      [0.0,   1.0]])
        
        self.P = F @ self.P @ F.T + self.Q

    def update(self, v_obs):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        y = v_obs - self.x[0] 
        
        self.x = self.x + (K @ np.array([y])).flatten()
        
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
