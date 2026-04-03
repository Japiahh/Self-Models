from dataclasses import dataclass

@dataclass
class OrganismConfig:
    input_dim: int = 3      
    hidden_dim: int = 64     
    latent_dim: int = 16     
    output_dim: int = 3      
    active_mode: bool = False 
    learning_rate: float = 0.002
    prediction_loss_weight: float = 0.8