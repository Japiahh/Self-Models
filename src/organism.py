import torch
import torch.nn as nn
from .encoder import SelfEncoder
from .predictor import SelfPredictor
from .decoder import ActionDecoder
from .memory import RecursiveCore

class Organism(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.core = RecursiveCore(config)
        
        self.observer = SelfEncoder(config.hidden_dim, config.latent_dim)
        self.oracle = SelfPredictor(config.latent_dim)
        self.actor = ActionDecoder(config.latent_dim, config.output_dim)

    def init_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.config.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.config.hidden_dim).to(device)
        s = torch.zeros(batch_size, self.config.latent_dim).to(device)
        return h, c, s

    # --- MODE V2: STANDARD SEQUENCE TRAINING ---
    def forward(self, x_sequence, h=None, c=None, s=None):
        batch_size, seq_len, _ = x_sequence.size()
        device = x_sequence.device
        
        if h is None:
            h, c, s = self.init_state(batch_size, device)

        outputs = []
        self_states = []      
        self_predictions = [] 

        for t in range(seq_len):
            x_t = x_sequence[:, t, :] 
            
            h, c = self.core(x_t, s, h, c, surprise_signal=None)
            
            s_curr = self.observer(h)
            s_pred_next = self.oracle(s_curr)
            y_t = self.actor(s_curr)
            
            outputs.append(y_t)
            self_states.append(s_curr)
            self_predictions.append(s_pred_next)
            s = s_curr

        outputs = torch.stack(outputs, dim=1)
        self_states = torch.stack(self_states, dim=1)
        self_predictions = torch.stack(self_predictions, dim=1) 

        return outputs, self_states, self_predictions

    # --- MODE V2: STANDARD STEP ---
    def step(self, x_t, h, c, s):
        if x_t.dim() == 1: x_t = x_t.unsqueeze(0)
        h, c = self.core(x_t, s, h, c, surprise_signal=None)
        s_curr = self.observer(h)
        y_t = self.actor(s_curr)
        return y_t, h, c, s_curr

    # --- MODE V3: AUTONOMOUS ACTIVE INFERENCE ---
    def forward_autonomous(self, x_sensory, h, c, s_prev, s_pred_prev):
        # 1. calculate suprises
        if self.config.active_mode and s_pred_prev is not None:
            raw_surprise = s_prev - s_pred_prev
            surprise_signal = torch.clamp(raw_surprise, -1.0, 1.0) 
        else:
            surprise_signal = torch.zeros_like(s_prev)

        # 2. update
        h_new, c_new = self.core(x_sensory, s_prev, h, c, surprise_signal)
        
        # 3. identity observasion
        s_curr = self.observer(h_new)
        
        # 4. predict
        s_pred_next = self.oracle(s_curr) 
        
        # 5. act
        x_action = self.actor(s_curr)
        
        return x_action, h_new, c_new, s_curr, s_pred_next