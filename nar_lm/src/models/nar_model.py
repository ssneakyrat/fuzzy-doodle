# src/models/nar_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer
# Set non-GUI backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.latent_mapper import LatentMapper

class LatentNARModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize components
        self.encoder = Encoder(config)
        self.latent_mapper = LatentMapper(config)
        self.decoder = Decoder(config)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # Sample prompts for generation logging
        self.sample_prompts = [
            "This is a test",
            "The model can",
            "Non-autoregressive generation is"
        ]
    
    def forward(self, input_ids, target_ids=None):
        encoder_output = self.encoder(input_ids)
        latent_output, latent = self.latent_mapper(encoder_output)
        
        logits = self.decoder(latent_output)
        
        loss = None
        if target_ids is not None:
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), target_ids.view(-1))
            
        return loss, logits, latent
    
    def generate(self, input_ids, max_length=None):
        if max_length is None:
            max_length = self.config.max_position_embeddings
            
        # Initial encoding
        encoder_output = self.encoder(input_ids)
        latent_output, _ = self.latent_mapper(encoder_output)
        
        # Initial prediction
        logits = self.decoder(latent_output)
        preds = torch.argmax(logits, dim=-1)
        
        # Iterative refinement
        for _ in range(self.config.num_refinement_steps):
            # Re-encode with the predicted tokens
            encoder_output = self.encoder(preds)
            latent_output, _ = self.latent_mapper(encoder_output)
            
            # Get new predictions
            logits = self.decoder(latent_output, encoder_output)
            new_preds = torch.argmax(logits, dim=-1)
            
            # Update high-confidence predictions
            probs = F.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
            mask = confidence > self.config.confidence_threshold
            preds = torch.where(mask, new_preds, preds)
        
        return preds
    
    def _shared_step(self, batch, batch_idx, step_type):
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        loss, logits, _ = self(input_ids, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1)
        mask = (labels != self.tokenizer.pad_token_id)
        correct = ((preds == labels) & mask).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0
        
        # Log metrics
        self.log(f"{step_type}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{step_type}_accuracy", accuracy, prog_bar=True, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")
    
    def _log_text_generations(self):
        """Generate text samples and log them to TensorBoard"""
        if not self.sample_prompts:
            return
        
        try:
            device = self.device
            generated_texts = []
            
            for prompt in self.sample_prompts:
                # Tokenize prompt
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                
                # Generate text
                with torch.no_grad():
                    output_ids = self.generate(input_ids)
                    
                # Decode output
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated_texts.append(f"Prompt: {prompt}\nGenerated: {output_text}\n{'-'*40}")
            
            # Combine all text samples
            text_output = "\n".join(generated_texts)
            
            # Log to TensorBoard
            self.logger.experiment.add_text("text_generations", text_output, self.global_step)
        except Exception as e:
            print(f"Warning: Text generation logging failed with error: {e}")
    
    def on_validation_epoch_end(self):
        """Log text generations at the end of each validation epoch"""
        self._log_text_generations()
    
    def configure_optimizers(self):
        lr = float(self.config.learning_rate)
        weight_decay = float(self.config.weight_decay)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        if self.config.use_lr_scheduler:
            # Ensure numerical types for scheduler parameters
            max_epochs = int(self.config.max_epochs)
            steps_per_epoch = int(self.config.steps_per_epoch)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=max_epochs * steps_per_epoch
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        
        return optimizer