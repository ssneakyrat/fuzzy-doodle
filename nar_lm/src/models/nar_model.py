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
        
        # Loss function for length prediction
        self.length_loss_fct = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, target_ids=None, target_seq_length=None):
        # Encode input
        encoder_output = self.encoder(input_ids, attention_mask)
        latent_output, latent = self.latent_mapper(encoder_output)
        
        # Get predictions from decoder
        logits, length_logits = self.decoder(latent_output, None, latent, target_seq_length)
        
        loss = None
        length_loss = None
        return_loss = None
        
        if target_ids is not None:
            # Calculate token prediction loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), target_ids.view(-1))
            return_loss = loss  # Start with base loss
            
            # Calculate length prediction loss if target_seq_length is provided
            if target_seq_length is not None and length_logits is not None:
                length_loss = self.length_loss_fct(length_logits, target_seq_length)
                # Combine losses with a weight factor
                return_loss = loss + 0.2 * length_loss
        
        # Always return the same structure
        return return_loss, logits, latent, length_logits
    
    def _predict_length(self, input_ids, latent=None):
        """Predict output sequence length based on input or latent"""
        if latent is None:
            # Get latent from input
            encoder_output = self.encoder(input_ids)
            _, latent = self.latent_mapper(encoder_output)
        
        # Pool latent across sequence dimension
        pooled_latent = torch.mean(latent, dim=1)
        length_logits = self.decoder.length_predictor(pooled_latent)
        pred_lengths = torch.argmax(length_logits, dim=-1)
        
        # Ensure minimum length (e.g., at least 4 tokens)
        min_length = 4
        pred_lengths = torch.maximum(pred_lengths, torch.tensor(min_length, device=pred_lengths.device))
        
        return pred_lengths
    
    def generate(self, input_ids, attention_mask=None, target_length=None):
        """Generate text with dynamic sequence length"""
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Encode input
        encoder_output = self.encoder(input_ids, attention_mask)
        latent_output, latent = self.latent_mapper(encoder_output)
        
        # Predict sequence length if not provided
        if target_length is None:
            pred_lengths = self._predict_length(None, latent)
            # Add small margin to predicted length
            max_length = int(torch.max(pred_lengths).item() * 1.2)
            # Limit to config's max position embeddings
            max_length = min(max_length, self.config.max_position_embeddings)
        else:
            # Use provided target length
            max_length = target_length
            pred_lengths = torch.tensor([max_length] * batch_size, device=device)
        
        # Initial prediction
        logits, _ = self.decoder(latent_output)
        preds = torch.argmax(logits, dim=-1)
        
        # Create an initial mask based on predicted lengths
        # This ensures we only update tokens within each sequence's predicted length
        length_mask = torch.arange(preds.size(1), device=device).unsqueeze(0) < pred_lengths.unsqueeze(1)
        
        # Iterative refinement
        for _ in range(self.config.num_refinement_steps):
            # Re-encode with the predicted tokens
            encoder_output = self.encoder(preds)
            latent_output, _ = self.latent_mapper(encoder_output)
            
            # Get new predictions
            logits, _ = self.decoder(latent_output, encoder_output)
            new_preds = torch.argmax(logits, dim=-1)
            
            # Update high-confidence predictions
            probs = F.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
            
            # Only update tokens that are:
            # 1. Within the predicted sequence length (length_mask)
            # 2. Have confidence above threshold
            update_mask = (confidence > self.config.confidence_threshold) & length_mask
            preds = torch.where(update_mask, new_preds, preds)
        
        # Apply length mask to final predictions (replace tokens beyond predicted length with pad token)
        preds = torch.where(length_mask, preds, torch.tensor(self.tokenizer.pad_token_id, device=device))
        
        return preds
    
    def _shared_step(self, batch, batch_idx, step_type):
        try:
            # Debug print to check batch structure
            print(f"[DEBUG] {step_type}_step batch keys: {batch.keys()}")
            print(f"[DEBUG] {step_type}_step input_ids shape: {batch['input_ids'].shape}")
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            seq_length = batch.get('seq_length')
            
            # Debug print for sequence length
            if seq_length is not None:
                print(f"[DEBUG] {step_type}_step seq_length: shape={seq_length.shape}, dtype={seq_length.dtype}, values={seq_length[:5]}")
            
            loss, logits, _, length_logits = self(input_ids, attention_mask, labels, seq_length)
            
            # Calculate accuracy with error handling
            accuracy = 0
            try:
                preds = torch.argmax(logits, dim=-1)
                # Debug predictions
                print(f"[DEBUG] {step_type}_step pred shape: {preds.shape}, label shape: {labels.shape}")
                
                # Verify token IDs
                print(f"[DEBUG] {step_type}_step pad_token_id: {self.tokenizer.pad_token_id}")
                
                mask = (labels != self.tokenizer.pad_token_id)
                correct = ((preds == labels) & mask).sum().item()
                total = mask.sum().item()
                accuracy = correct / total if total > 0 else 0
                
                # Debug accuracy calculation
                print(f"[DEBUG] {step_type}_step accuracy calculation: correct={correct}, total={total}, accuracy={accuracy}")
                
                # Log accuracy only if valid
                if not np.isnan(accuracy) and not np.isinf(accuracy):
                    self.log(f"{step_type}_accuracy", accuracy, prog_bar=True, sync_dist=True)
            except Exception as e:
                print(f"[ERROR] Error calculating {step_type}_accuracy: {str(e)}")
            
            # Calculate length prediction accuracy if applicable
            length_accuracy = 0
            try:
                if seq_length is not None and length_logits is not None:
                    pred_lengths = torch.argmax(length_logits, dim=-1)
                    
                    # Debug length predictions
                    print(f"[DEBUG] {step_type}_step length_logits shape: {length_logits.shape}")
                    print(f"[DEBUG] {step_type}_step pred_lengths: {pred_lengths[:5]}")
                    
                    # Ensure consistent types
                    if pred_lengths.dtype != seq_length.dtype:
                        print(f"[DEBUG] {step_type}_step converting pred_lengths from {pred_lengths.dtype} to {seq_length.dtype}")
                        pred_lengths = pred_lengths.to(seq_length.dtype)
                    
                    # Consider length correct if within ±2 tokens of actual length
                    diff = torch.abs(pred_lengths - seq_length)
                    print(f"[DEBUG] {step_type}_step length diff: {diff[:5]}")
                    
                    length_correct = (diff <= 2).sum().item()
                    length_accuracy = length_correct / len(seq_length)
                    
                    # Debug length accuracy
                    print(f"[DEBUG] {step_type}_step length accuracy: correct={length_correct}, total={len(seq_length)}, accuracy={length_accuracy}")
                    
                    # Log length accuracy only if valid
                    if not np.isnan(length_accuracy) and not np.isinf(length_accuracy):
                        self.log(f"{step_type}_length_accuracy", length_accuracy, prog_bar=True, sync_dist=True)
            except Exception as e:
                print(f"[ERROR] Error calculating {step_type}_length_accuracy: {str(e)}")
            
            # Log loss only if valid
            if loss is not None:
                loss_value = loss.item() if not torch.isnan(loss) and not torch.isinf(loss) else None
                print(f"[DEBUG] {step_type}_step loss: {loss_value}")
                if loss_value is not None:
                    self.log(f"{step_type}_loss", loss, prog_bar=True, sync_dist=True)
            
            # Print comprehensive debug summary
            print(f"[DEBUG] {step_type}_step SUMMARY - Loss: {loss.item() if loss is not None else 'None'}, "
                  f"Accuracy: {accuracy}, Length Accuracy: {length_accuracy}")
            
            return loss
        except Exception as e:
            print(f"[ERROR] Critical error in {step_type}_step: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return zero loss to avoid breaking the training loop
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        print(f"[DEBUG] Running test_step with batch_idx={batch_idx}, batch shape={batch['input_ids'].shape}")
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
                input_data = self.tokenizer(prompt, return_tensors="pt")
                input_ids = input_data.input_ids.to(device)
                attention_mask = input_data.attention_mask.to(device)
                
                # Generate text
                with torch.no_grad():
                    output_ids = self.generate(input_ids, attention_mask)
                    
                # Decode output
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated_texts.append(f"Prompt: {prompt}\nGenerated: {output_text}\n{'-'*40}")
            
            # Combine all text samples
            text_output = "\n".join(generated_texts)
            
            # Log to TensorBoard
            self.logger.experiment.add_text("text_generations", text_output, self.global_step)
        except Exception as e:
            print(f"[ERROR] Text generation logging failed: {e}")
            import traceback
            traceback.print_exc()
    
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