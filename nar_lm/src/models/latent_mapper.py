import torch
import torch.nn as nn

class LatentMapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down_proj = nn.Linear(config.hidden_size, config.latent_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.latent_size)
        self.up_proj = nn.Linear(config.latent_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, encoder_output):
        latent = self.down_proj(encoder_output)
        latent = self.act(latent)
        latent = self.layer_norm(latent)
        latent = self.dropout(latent)
        output = self.up_proj(latent)
        return output, latent