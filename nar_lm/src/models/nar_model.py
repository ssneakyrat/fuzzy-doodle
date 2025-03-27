# src/models/enhanced_nar_model.py
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
from src.utils.logger import setup_logger

# Set up module logger
logger = setup_logger(__name__)

class EnhancedNARModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize tokenizer first to get accurate vocab size
        logger.info(f"Loading tokenizer: {config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # Update vocab_size to match tokenizer's vocabulary
        self.config.vocab_size = len(self.tokenizer.vocab)
        logger.info(f"Updated vocab_size to {self.config.vocab_size} to match tokenizer")

        # Initialize components
        logger.info(f"Initializing enhanced model with hidden_size={config.hidden_size}, latent_size={config.latent_size}")
        self.encoder = Encoder(config)
        self.latent_mapper = LatentMapper(config)
        self.decoder = Decoder(config)  # Use the ImprovedDecoder
        
        # Sample prompts for generation logging
        self.sample_prompts = [
            "I'm not sure if this is a compliment",
            "but I find listening to your videos whilst I'm falling asleep extremely soothing",
            "I have very severe anxiety disorder and this helps me a lot, Thanks for  your content"
        ]
        
        # Loss function for length prediction
        self.length_loss_fct = nn.CrossEntropyLoss()
        
        # Progressive confidence threshold schedule
        self.base_confidence_threshold = getattr(config, 'confidence_threshold', 0.9)
        # Lower thresholds in early refinement steps
        self.confidence_schedule = [
            self.base_confidence_threshold - 0.2,  # Step 1: Lower threshold
            self.base_confidence_threshold - 0.1,  # Step 2: Slightly higher
            self.base_confidence_threshold,        # Step 3: Original threshold
            self.base_confidence_threshold + 0.05  # Step 4+: Higher threshold for later steps
        ]
    
    # In the forward method, add validation for target indices
    def forward(self, input_ids, attention_mask=None, target_ids=None, target_seq_length=None):
        # Encode input
        encoder_output = self.encoder(input_ids, attention_mask)
        latent_output, latent = self.latent_mapper(encoder_output)
        
        # Get predictions from decoder
        logits, length_logits, _ = self.decoder(latent_output, None, latent, target_seq_length)
        
        loss = None
        length_loss = None
        return_loss = None
        
        if target_ids is not None:
            # Log statistics about target IDs to debug issues
            with torch.no_grad():
                max_id = target_ids.max().item()
                min_id = target_ids.min().item()
                if max_id >= self.config.vocab_size:
                    logger.warning(f"Found target ID {max_id} >= vocab_size {self.config.vocab_size}, clipping will occur")
            
            # Calculate token prediction loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), target_ids.view(-1))
            return_loss = loss  # Start with base loss
            
            # Calculate length prediction loss if target_seq_length is provided
            if target_seq_length is not None and length_logits is not None:
                # Ensure target sequence lengths are within valid range
                max_pred_length = length_logits.size(-1)
                # Add logging to debug length prediction issues
                with torch.no_grad():
                    max_length = target_seq_length.max().item()
                    min_length = target_seq_length.min().item()
                    if max_length >= max_pred_length:
                        logger.warning(f"Found target length {max_length} >= max_pred_length {max_pred_length}, clipping will occur")
                
                # Clip target sequence lengths to valid range
                target_seq_length_clipped = torch.clamp(target_seq_length, 0, max_pred_length - 1)
                
                # Use clipped values for loss calculation
                length_loss = self.length_loss_fct(length_logits, target_seq_length_clipped)
                
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
    
    def _get_confidence_threshold(self, step_idx):
        """Get confidence threshold based on refinement step"""
        if step_idx < len(self.confidence_schedule):
            return self.confidence_schedule[step_idx]
        return self.confidence_schedule[-1]  # Use the last value for all later steps
    
    def generate(self, input_ids, attention_mask=None, target_length=None):
        """Generate text with progressive refinement"""
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
        
        # Initial prediction (step 0)
        logits, _, _ = self.decoder(latent_output)
        preds = torch.argmax(logits, dim=-1)
        
        # Create an initial mask based on predicted lengths
        # This ensures we only update tokens within each sequence's predicted length
        length_mask = torch.arange(preds.size(1), device=device).unsqueeze(0) < pred_lengths.unsqueeze(1)
        
        # Apply length masking immediately to initial predictions
        preds = torch.where(length_mask, preds, torch.tensor(self.tokenizer.pad_token_id, device=device))
        
        # Progressive refinement
        for step in range(self.config.num_refinement_steps):
            # Re-encode with the predicted tokens
            encoder_output = self.encoder(preds)
            latent_output, _ = self.latent_mapper(encoder_output)
            
            # Get new predictions with confidence scores
            logits, _, confidence_scores = self.decoder(latent_output, encoder_output, prev_token_ids=preds)
            new_preds = torch.argmax(logits, dim=-1)
            
            # If confidence scores not available, calculate from logits
            if confidence_scores is None:
                probs = F.softmax(logits, dim=-1)
                confidence_scores = torch.max(probs, dim=-1)[0]
            
            # Get dynamic confidence threshold for this step
            threshold = self._get_confidence_threshold(step)
            
            # Only update tokens that are:
            # 1. Within the predicted length (length_mask)
            # 2. Have confidence above threshold
            update_mask = (confidence_scores > threshold) & length_mask
            
            # Log refinement progress for debugging
            update_ratio = torch.sum(update_mask).item() / torch.sum(length_mask).item()
            logger.debug(f"Refinement step {step+1}: Updating {update_ratio:.2%} of tokens with threshold {threshold:.2f}")
            
            # Progressive update strategy
            preds = torch.where(update_mask, new_preds, preds)
            
            # Always ensure length masking is applied
            preds = torch.where(length_mask, preds, torch.tensor(self.tokenizer.pad_token_id, device=device))
        
        return preds
    
    def _shared_step(self, batch, batch_idx, step_type):
        try:
            # Log batch structure
            logger.debug(f"{step_type}_step batch keys: {batch.keys()}")
            logger.debug(f"{step_type}_step input_ids shape: {batch['input_ids'].shape}")
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            seq_length = batch.get('seq_length')
            
            # Forward pass
            loss, logits, _, length_logits = self(input_ids, attention_mask, labels, seq_length)
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            
            # Make sure preds and labels have same size for accurate comparison
            if preds.size() != labels.size():
                logger.warning(f"Size mismatch: preds={preds.shape}, labels={labels.shape}. Truncating to smaller size.")
                # Truncate to the smaller size
                min_len = min(preds.size(1), labels.size(1))
                preds = preds[:, :min_len]
                labels = labels[:, :min_len]
            
            mask = (labels != self.tokenizer.pad_token_id)
            correct = ((preds == labels) & mask).sum().item()
            total = mask.sum().item()
            accuracy = correct / total if total > 0 else 0
            
            # Log accuracy
            self.log(f"{step_type}_accuracy", accuracy, prog_bar=True, sync_dist=True)
            
            # Calculate length prediction accuracy if applicable
            if seq_length is not None and length_logits is not None:
                pred_lengths = torch.argmax(length_logits, dim=-1)
                
                # Ensure consistent types
                if pred_lengths.dtype != seq_length.dtype:
                    pred_lengths = pred_lengths.to(seq_length.dtype)
                
                # Consider length correct if within ±2 tokens of actual length
                diff = torch.abs(pred_lengths - seq_length)
                
                length_correct = (diff <= 2).sum().item()
                length_accuracy = length_correct / len(seq_length)
                
                # Log length accuracy
                self.log(f"{step_type}_length_accuracy", length_accuracy, prog_bar=True, sync_dist=True)
            
            # Log loss
            self.log(f"{step_type}_loss", loss, prog_bar=True, sync_dist=True)
            
            return loss
        except Exception as e:
            logger.error(f"Critical error in {step_type}_step: {str(e)}", exc_info=True)
            # Return zero loss to avoid breaking the training loop
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
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
            logger.error(f"Text generation logging failed: {e}", exc_info=True)
    
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