# main_test.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from config import OrganismConfig
from src.organism import Organism

def generate_dummy_data(batch_size, seq_len, input_dim):
    data = torch.zeros(batch_size, seq_len, input_dim)
    for i in range(batch_size):
        phase = np.random.rand() * 2 * np.pi
        x = np.linspace(0, 4 * np.pi, seq_len)
        wave = np.sin(x + phase)
        data[i, :, :] = torch.tensor(wave).unsqueeze(1).repeat(1, input_dim)
    return data

def train():
    # 1. Setup
    config = OrganismConfig(input_dim=1, output_dim=1, hidden_dim=32, latent_dim=8)
    model = Organism(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Loss Functions
    task_criterion = nn.MSELoss() 
    dream_criterion = nn.MSELoss() 

    print("Memulai Inisialisasi Organisme...")
    
    # Data Dummy
    data = generate_dummy_data(32, 50, 1) # [Batch, Seq, Dim]
    inputs = data[:, :-1, :]  # t=0 to t=48
    targets = data[:, 1:, :]  # t=1 to t=49

    # 2. Training Loop
    epochs = 100
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward Pass
        outputs, states, predictions = model(inputs)
        
        # === CALCULATE DUAL LOSS ===
        
        # A. Task Loss 
        loss_task = task_criterion(outputs, targets)
        
        # B. Self-Modeling Loss (Free Energy Principle)
        #  (t) ~ (t+1)
        # predictions[:, :-1] for t+1
        # states[:, 1:] fact in t+1
        loss_self = dream_criterion(predictions[:, :-1], states[:, 1:])
        
        # Total Loss (Free Energy Total)
        total_loss = loss_task + (config.prediction_loss_weight * loss_self)
        
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.5f} (Self-Loss: {loss_self.item():.5f})")

    print("done.")
    
    plt.plot(loss_history)
    plt.title("Free Energy (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    torch.save(model.state_dict(), "base_organism.pth")
    print("save as'base_organism.pth'")

if __name__ == "__main__":
    train()