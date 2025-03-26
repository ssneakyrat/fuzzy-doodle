import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        inputs_embeds = self.embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = inputs_embeds + position_embeds
        encoder_output = self.encoder(hidden_states)
        
        return encoder_output