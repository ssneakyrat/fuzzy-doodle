import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.hidden_size = config.hidden_size
        
        # Remove fixed position embedding, we'll use sinusoidal embeddings
        # that can handle arbitrary sequence lengths
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        
    def get_sinusoidal_embedding(self, seq_length, device):
        """Generate sinusoidal position embeddings for arbitrary sequence lengths"""
        positions = torch.arange(seq_length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2, device=device) * 
                            -(math.log(10000.0) / self.hidden_size))
        
        pos_embedding = torch.zeros(seq_length, self.hidden_size, device=device)
        pos_embedding[:, 0::2] = torch.sin(positions * div_term)
        pos_embedding[:, 1::2] = torch.cos(positions * div_term[:pos_embedding.size(1)//2])
        
        return pos_embedding
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Get token embeddings
        inputs_embeds = self.embedding(input_ids)
        
        # Get sinusoidal position embeddings for the current sequence length
        position_embeds = self.get_sinusoidal_embedding(seq_length, device)
        # Expand to batch size
        position_embeds = position_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add position embeddings to token embeddings
        hidden_states = inputs_embeds + position_embeds
        
        # Apply transformer encoder with attention mask if provided
        if attention_mask is not None:
            # Convert boolean mask to attention mask (1.0 for tokens, 0.0 for padding)
            encoder_attention_mask = attention_mask.float()
            # Invert mask for PyTorch transformer (1.0 for padding)
            encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0
            encoder_output = self.encoder(hidden_states, src_key_padding_mask=~attention_mask.bool())
        else:
            encoder_output = self.encoder(hidden_states)
        
        return encoder_output