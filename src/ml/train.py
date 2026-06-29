import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



class FrictionNet(nn.Module):
    def __init__(self, window_size):
        super(FrictionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  )

    def forward(self, x):
        return self.net(x)


def generate_training_data(num_samples, window_size, dt=0.01):
    X = []
    y = []
    
    g = 9.81
    k = 0.02 
    
    for _ in range(num_samples):
        true_mu = np.random.uniform(0.3, 1.0) 
        v0 = np.random.uniform(20.0, 40.0)    
        
        v = v0
        history = []
        
        for _ in range(window_size):
            noise = np.random.normal(0, 0.5)
            history.append(v + noise)
            
            a = -true_mu * g - k * (v**2)
            v = v + a * dt
            
            if v < 0: v = 0
            
        X.append(history)
        y.append(true_mu)
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)


def train_model():
    window_size = 50 
    epochs = 2500
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Accelerating via NVIDIA CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Accelerating via Apple MPS")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Defaulting to CPU.")

    print("Generating synthetic telemetry data...")
    X_train, y_train = generate_training_data(10000, window_size)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    model = FrictionNet(window_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training FrictionNet...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    return model, device




    
    