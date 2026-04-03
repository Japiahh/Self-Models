import torch
import torch.nn as nn

class ActionDecoder(nn.Module):
    """
    y(t) = Linear(s(t))
    """
    def __init__(self, latent_dim: int, output_dim: int):
        super(ActionDecoder, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(latent_dim * 2, output_dim)
            )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Input: [Batch, Latent_Dim]
        # Output: [Batch, Output_Dim]
        return self.head(s)