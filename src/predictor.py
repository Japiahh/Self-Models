import torch
import torch.nn as nn

class SelfPredictor(nn.Module):
    """
    [The Oracle]
    s_hat(t+1) = s(t) + Delta(s(t))
    """
    def __init__(self, latent_dim: int, hidden_layers: int = 2):
        super(SelfPredictor, self).__init__()
        
        layers = []
        # Input Layer
        layers.append(nn.Linear(latent_dim, latent_dim * 2))
        layers.append(nn.GELU()) 
        
        # Hidden Deep Layers 
        for _ in range(hidden_layers):
            layers.append(nn.Linear(latent_dim * 2, latent_dim * 2))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            
        # Output Layer 
        layers.append(nn.Linear(latent_dim * 2, latent_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, s_current: torch.Tensor) -> torch.Tensor:
        delta = self.net(s_current)

        return s_current + delta