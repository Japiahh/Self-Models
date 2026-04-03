import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import Modular
from config import OrganismConfig
from src.organism import Organism

# --- Generator Data ---
def get_lorenz_data(seq_len=1000):
    state = np.array([1.0, 1.0, 1.0])
    dt = 0.01
    data = []
    for _ in range(seq_len):
        x, y, z = state
        dx = 10.0 * (y - x)
        dy = x * (28.0 - z) - y
        dz = x * y - (8.0/3.0) * z
        state = state + np.array([dx, dy, dz]) * dt
        data.append(state)
    data = np.array(data)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data, dtype=torch.float32)

def main():
    # 1. Config V2 (Passive Mode)
    config = OrganismConfig(
        input_dim=3, output_dim=3, hidden_dim=64, latent_dim=16,
        active_mode=False # False = old version (Standard)
    )
    
    model = Organism(config)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    mse = nn.MSELoss()

    # 2. Data
    full_data = get_lorenz_data(1500)
    train_data = full_data[:1000].unsqueeze(0) # [1, 1000, 3]
    test_data = full_data[1000:]    
    
    # 3. Training Loop
    print("--- [V2] Passive Models ---")
    loss_hist = []
    
    model.train()
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        inputs = train_data[:, :-1, :]
        targets = train_data[:, 1:, :]
        
        outputs, self_states, self_predictions = model(inputs)
        
        loss_task = mse(outputs, targets)
        loss_dream = mse(self_predictions[:, :-1], self_states[:, 1:])
        total_loss = loss_task + (0.8 * loss_dream)
        
        total_loss.backward()
        optimizer.step()
        loss_hist.append(total_loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.5f}")

    print("\n Closed Loop Simulation ")
    dream_path = []
    model.eval() 
    
    with torch.no_grad():
        x = test_data[0].unsqueeze(0) # [1, 3]
        h = torch.zeros(1, config.hidden_dim)
        c = torch.zeros(1, config.hidden_dim)
        s = torch.zeros(1, config.latent_dim)
        
        for i in range(len(test_data)):
            y_pred, h, c, s_curr = model.step(x, h, c, s)
            dream_path.append(y_pred.numpy()[0])
            x = y_pred 
            s = s_curr

    dream_path = np.array(dream_path)
    real_path = test_data.numpy()

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(real_path[:,0], real_path[:,1], real_path[:,2], lw=1, color='black', label='Real Physics (Chaos)', alpha=0.6)
    ax1.plot(dream_path[:,0], dream_path[:,1], dream_path[:,2], lw=1.5, color='red', label='Model Hallucination', linestyle='--')
    ax1.set_title("[V2 Passive] Lorenz Attractor: Reality vs Dream")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(loss_hist)
    ax2.set_title("Free Energy (Learning Curve)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Total Loss")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()