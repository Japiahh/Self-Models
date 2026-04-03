import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_lorenz_data(seq_len=1000, dt=0.01):
    # Parameter Chaos standar
    sigma=10.0; rho=28.0; beta=8.0/3.0
    
    # Inisialisasi (x, y, z)
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
        
        # A. The Observer (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim),
            nn.Tanh() 
        )
        
        # B. The Core (LSTM Memory)
        self.core = nn.LSTMCell(input_dim + latent_dim, hidden_dim)
        
        # C. The Oracle (Self-Predictor)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )
        
        # D. The Actor (Decoder ke Dunia Nyata)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c, s_prev):
        # 1. Update Memory (Core)
        combined_input = torch.cat([x, s_prev], dim=1)
        h_new, c_new = self.core(combined_input, (h, c))
        
        # 2. Perceive Self (Observer)
        s_curr = self.encoder(h_new)
        
        # 3. Predict Future Self (Oracle)
        # Residual connection: s_next = s_curr + delta
        s_pred_next = s_curr + self.predictor(s_curr)
        
        # 4. Act (Decoder)
        y_out = self.decoder(s_curr)
        
        return y_out, s_curr, s_pred_next, h_new, c_new

def train_chaos():
    print("Membangkitkan Lorenz Attractor (Chaos Data)...")
    seq_len = 1000
    full_data = get_lorenz_data(seq_len)
    
    train_data = full_data[:800]
    test_data = full_data[800:]
    
    model = ChaosOrganism()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Dual Loss
    mse = nn.MSELoss()
    
    print("\nMulai Latihan Organisme...")
    loss_history = []
    
    for epoch in range(150):
        h = torch.zeros(1, 64)
        c = torch.zeros(1, 64)
        s = torch.zeros(1, 16)
        
        optimizer.zero_grad()
        
        loss_task_total = 0
        loss_dream_total = 0
        
        # --- PHASE 1 ---
        predictions = []
        self_states = []
        self_preds = []
        
        for t in range(len(train_data) - 1):
            x_t = train_data[t].unsqueeze(0)     
            target = train_data[t+1].unsqueeze(0) 
            
            y_pred, s_curr, s_next_pred, h, c = model(x_t, h, c, s)
            
            predictions.append(y_pred)
            self_states.append(s_curr)
            self_preds.append(s_next_pred)

            s = s_curr
            
        # --- PHASE 2 ---
        preds_tensor = torch.cat(predictions)
        targets_tensor = train_data[1:]
        
        states_tensor = torch.cat(self_states)
        dream_tensor = torch.cat(self_preds)  
        
        # A. Task Loss
        loss_task = mse(preds_tensor, targets_tensor)
        
        # B. Dream Loss
        loss_dream = mse(dream_tensor[:-1], states_tensor[1:])
        
        # Total Loss (Free Energy)
        total_loss = loss_task + (0.8 * loss_dream) 
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 30 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss.item():.5f} (Dream Error: {loss_dream.item():.5f})")

    return model, test_data, loss_history

model, test_data, loss_hist = train_chaos()

# --- CLOSED LOOP DREAMING ---
print("\nMenjalankan Simulasi 'Closed Loop' (Mimpi)...")
dream_path = []
with torch.no_grad():

    x = test_data[0].unsqueeze(0)
    h = torch.zeros(1, 64)
    c = torch.zeros(1, 64)
    s = torch.zeros(1, 16)
    
    for i in range(len(test_data)):
        y_pred, s_curr, _, h, c = model(x, h, c, s)
        dream_path.append(y_pred.numpy()[0])
        x = y_pred 
        s = s_curr

dream_path = np.array(dream_path)
real_path = test_data.numpy()

# PLOTTING
fig = plt.figure(figsize=(12, 5))

# Plot 1
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(real_path[:,0], real_path[:,1], real_path[:,2], lw=1, color='black', label='Real Physics (Chaos)', alpha=0.6)
ax1.plot(dream_path[:,0], dream_path[:,1], dream_path[:,2], lw=1.5, color='red', label='Model Hallucination', linestyle='--')
ax1.set_title("Lorenz Attractor: Reality vs Dream")
ax1.legend()

# Plot 2
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(loss_hist)
ax2.set_title("Free Energy (Learning Curve)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Total Loss")
ax2.grid(True)

plt.tight_layout()
plt.show()