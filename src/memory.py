import torch
import torch.nn as nn

class RecursiveCore(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        input_size = config.input_dim + config.latent_dim
        
        if config.active_mode:
            input_size += config.latent_dim 
            
        self.cell = nn.LSTMCell(input_size, config.hidden_dim)

    def forward(self, x_t, s_prev, h_prev, c_prev, surprise_signal=None):
        features = [x_t, s_prev]
        
        if surprise_signal is not None:
            features.append(surprise_signal)
            
        combined_input = torch.cat(features, dim=1)
        
        h_curr, c_curr = self.cell(combined_input, (h_prev, c_prev))
        return h_curr, c_curr