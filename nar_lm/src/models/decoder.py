import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
    def forward(self, latent_output, encoder_output=None):
        seq_length = latent_output.size(1)
        position_ids = torch.arange(seq_length, device=latent_output.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        # Add positional embeddings to latent output
        latent_output = latent_output + position_embeds
        
        # For the initial generation, we use self-attention only
        if encoder_output is None:
            # For non-autoregressive generation, we don't use causal masking
            decoder_output = self.decoder(latent_output, latent_output)
        else:
            # For refinement, we use cross-attention with the encoder output
            decoder_output = self.decoder(latent_output, encoder_output)
        
        logits = self.output_proj(decoder_output)
        return logits