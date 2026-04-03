import torch
import torch.nn as nn

class SelfEncoder(nn.Module):
    """
    [The Observer]
    short-teerm memory (h) to identity (s).
    
    s(t) = Tanh(Linear(h(t)))
    """
    def __init__(self, hidden_dim: int, latent_dim: int, dropout: float = 0.1):
        super(SelfEncoder, self).__init__()
        
        self.compressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh()  
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Input: [Batch, Hidden_Dim]
        # Output: [Batch, Latent_Dim]
        return self.compressor(h)