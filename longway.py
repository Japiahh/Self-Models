import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_lorenz_data(seq_len=5000, dt=0.01): 
    sigma=10.0; rho=28.0; beta=8.0/3.0
    state = np.array([1.0, 1.0, 1.0])
    data = []
    
    for _ in range(seq_len):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        state = state + np.array([dx, dy, dz]) * dt
        data.append(state)

    data = np.array(data)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data, dtype=torch.float32)

class ChaosOrganism(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim),
            nn.Tanh() 
        )
        self.core = nn.LSTMCell(input_dim + latent_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x, h, c, s_prev):
        combined_input = torch.cat([x, s_prev], dim=1)
        h_new, c_new = self.core(combined_input, (h, c))
        s_curr = self.encoder(h_new)
        s_pred_next = s_curr + self.predictor(s_curr)
        y_out = self.decoder(s_curr)
        return y_out, s_curr, s_pred_next, h_new, c_new

def train_and_dream_long():
    TOTAL_LEN = 5000
    TRAIN_LEN = 1500 
    TEST_LEN = TOTAL_LEN - TRAIN_LEN 
    
    print(f"Generating Data: {TOTAL_LEN} steps.")
    print(f"Training: {TRAIN_LEN} step. Dreaming Test: {TEST_LEN} step.")
    
    full_data = get_lorenz_data(TOTAL_LEN)
    train_data = full_data[:TRAIN_LEN]
    test_data = full_data[TRAIN_LEN:] 
    
    model = ChaosOrganism()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    mse = nn.MSELoss()
    
    print("\n--- PHASE 1: TRAINING (finding 'Aha' Moment) ---")
    loss_history = []
    epochs = 100 
    
    for epoch in range(epochs):
        h = torch.zeros(1, 64)
        c = torch.zeros(1, 64)
        s = torch.zeros(1, 16)
        
        optimizer.zero_grad()
        
        predictions = []
        self_states = []
        self_preds = []
        
        for t in range(len(train_data) - 1):
            x_t = train_data[t].unsqueeze(0)
            y_pred, s_curr, s_next_pred, h, c = model(x_t, h, c, s)
            predictions.append(y_pred)
            self_states.append(s_curr)
            self_preds.append(s_next_pred)
            s = s_curr
            
        preds_tensor = torch.cat(predictions)
        targets_tensor = train_data[1:]
        states_tensor = torch.cat(self_states)
        dream_tensor = torch.cat(self_preds)
        
        loss_task = mse(preds_tensor, targets_tensor)
        loss_dream = mse(dream_tensor[:-1], states_tensor[1:])
        total_loss = loss_task + (0.8 * loss_dream)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.5f}")

    print("\n--- FASE 2: LONG DREAMING (stability test) ---")
    dream_path = []
    
    with torch.no_grad():
        x = train_data[-1].unsqueeze(0)
        
        for i in range(TEST_LEN):
            y_pred, s_curr, _, h, c = model(x, h, c, s)
            dream_path.append(y_pred.numpy()[0])
            x = y_pred 
            s = s_curr
            
    return np.array(dream_path), test_data.numpy(), loss_history

dream_path, real_path, loss_hist = train_and_dream_long()

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(real_path[:,0], real_path[:,1], real_path[:,2], lw=0.5, color='black', alpha=0.3, label='Real Physics')
ax1.plot(dream_path[:,0], dream_path[:,1], dream_path[:,2], lw=1.0, color='red', alpha=0.9, label='Long Dream (3500 steps)')
ax1.set_title(f"Stability Test: {len(dream_path)} Steps Dreaming")
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(loss_hist)
ax2.set_title("Training Convergence")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Free Energy")
ax2.grid(True)

plt.tight_layout()
plt.savefig('long_lorenz_dream.png')