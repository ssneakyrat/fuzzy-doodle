import torch
import torch.nn as nn
import math

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_pred_length = config.max_position_embeddings
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Add length prediction head
        self.length_predictor = nn.Sequential(
            nn.Linear(config.latent_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, self.max_pred_length)
        )
        
    def get_sinusoidal_embedding(self, seq_length, device):
        """Generate sinusoidal position embeddings for arbitrary sequence lengths"""
        positions = torch.arange(seq_length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2, device=device) * 
                            -(math.log(10000.0) / self.hidden_size))
        
        pos_embedding = torch.zeros(seq_length, self.hidden_size, device=device)
        pos_embedding[:, 0::2] = torch.sin(positions * div_term)
        pos_embedding[:, 1::2] = torch.cos(positions * div_term[:pos_embedding.size(1)//2])
        
        return pos_embedding
        
    def forward(self, latent_output, encoder_output=None, latent=None, target_seq_length=None):
        batch_size, seq_length = latent_output.size(0), latent_output.size(1)
        device = latent_output.device
        
        # Get position embeddings for the sequence
        position_embeds = self.get_sinusoidal_embedding(seq_length, device)
        # Expand to batch size
        position_embeds = position_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add positional embeddings to latent output
        latent_output = latent_output + position_embeds
        
        # For the initial generation, we use self-attention only
        if encoder_output is None:
            # For non-autoregressive generation, we don't use causal masking
            decoder_output = self.decoder(latent_output, latent_output)
        else:
            # For refinement, we use cross-attention with the encoder output
            decoder_output = self.decoder(latent_output, encoder_output)
        
        # Get token predictions
        logits = self.output_proj(decoder_output)
        
        # Predict sequence length if latent is provided
        length_logits = None
        if latent is not None:
            # Pool latent across sequence dimension
            pooled_latent = torch.mean(latent, dim=1)
            length_logits = self.length_predictor(pooled_latent)
            
        return logits, length_logits