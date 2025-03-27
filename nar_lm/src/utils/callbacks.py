# src/utils/callbacks.py
import pytorch_lightning as pl
import torch
# Set non-GUI backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import torchvision

class LatentVisualizationCallback(pl.Callback):
    """Visualize the latent space during training"""
    
    def __init__(self, num_samples=10):
        super().__init__()
        self.num_samples = num_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            # Get a batch of validation data
            val_dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))
            
            # Debug print to help troubleshoot
            print(f"[DEBUG] LatentVisualizationCallback: batch keys: {batch.keys()}")
            
            # Move to the same device as the model
            input_ids = batch['input_ids'].to(pl_module.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(pl_module.device)
            
            print(f"[DEBUG] LatentVisualizationCallback: input_ids shape: {input_ids.shape}")
            
            # Forward pass to get latent representations with error handling
            with torch.no_grad():
                encoder_output = pl_module.encoder(input_ids, attention_mask)
                print(f"[DEBUG] LatentVisualizationCallback: encoder_output shape: {encoder_output.shape}")
                _, latent = pl_module.latent_mapper(encoder_output)
                print(f"[DEBUG] LatentVisualizationCallback: latent shape: {latent.shape}")
            
            # Take a subset and ensure we have enough data
            max_samples = min(self.num_samples, latent.size(0))
            if max_samples == 0:
                print("[WARNING] LatentVisualizationCallback: No samples available for visualization")
                return
                
            latent_subset = latent[:max_samples].cpu().numpy()
            print(f"[DEBUG] LatentVisualizationCallback: latent_subset shape: {latent_subset.shape}")
            
            # Visualize latent space (first 2 dimensions)
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Use a more robust approach for visualization
            for i, sample_latent in enumerate(latent_subset):
                # Average across sequence length (handling variable lengths)
                avg_latent = np.mean(sample_latent, axis=0)
                
                # Ensure we have at least 2 dimensions
                if len(avg_latent) < 2:
                    print(f"[WARNING] Latent dimension too small: {len(avg_latent)}")
                    continue
                    
                # Take first 2 dimensions
                ax.scatter(avg_latent[0], avg_latent[1], label=f"Sample {i+1}")
            
            ax.set_title("Latent Space Visualization (First 2 Dimensions)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.legend()
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to image
            image = np.array(Image.open(buf))
            plt.close(fig)
            
            # Log to TensorBoard
            trainer.logger.experiment.add_image(
                "latent_space", 
                image.transpose(2, 0, 1),  # Convert to channel-first format
                global_step=trainer.global_step
            )
        except Exception as e:
            print(f"Warning: Latent visualization failed with error: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for better debugging


class AttentionVisualizationCallback(pl.Callback):
    """Visualize attention patterns during training"""
    
    def __init__(self, prompts=["This is a test visualization"]):
        super().__init__()
        self.prompts = prompts
    
    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            # Instead of trying to extract real attention weights (which requires model modification),
            # create a placeholder visualization that explains the situation
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get a sample sentence for visualization
            device = pl_module.device
            tokenizer = pl_module.tokenizer
            prompt = self.prompts[0]
            
            # Tokenize
            tokens = tokenizer.tokenize(prompt)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            seq_len = input_ids.size(1)
            
            # Create a pattern that resembles attention but is just for visualization
            # This creates a pattern where tokens attend more to nearby tokens
            # (Just a placeholder until real attention weights can be extracted)
            att_map = np.zeros((seq_len, seq_len))
            for i in range(seq_len):
                for j in range(seq_len):
                    # Distance-based pattern (closer tokens attend more to each other)
                    att_map[i, j] = 1.0 / (1.0 + abs(i-j))
                    
            # Normalize rows
            for i in range(seq_len):
                att_map[i] = att_map[i] / att_map[i].sum()
            
            # Create heatmap visualization
            im = ax.imshow(att_map, cmap='viridis')
            fig.colorbar(im, ax=ax)
            
            # Add labels
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens)
            
            ax.set_title("Simulated Attention Pattern\n(Note: Uses distance-based pattern, not actual model attention)")
            
            # Convert to image and log
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            image = np.array(Image.open(buf))
            plt.close(fig)
            
            trainer.logger.experiment.add_image(
                "attention_map", 
                image.transpose(2, 0, 1),
                global_step=trainer.global_step
            )
            
            # Also log a text note explaining the situation
            trainer.logger.experiment.add_text(
                "attention_map_note",
                "Note: To visualize actual attention weights, the model needs to be modified to expose them during the forward pass.",
                global_step=trainer.global_step
            )
        except Exception as e:
            print(f"Warning: Attention visualization failed with error: {e}")
            import traceback
            traceback.print_exc()


class GenerationProgressCallback(pl.Callback):
    """Visualize the refinement process for generation"""
    
    def __init__(self, prompts=["The model generates"]):
        super().__init__()
        self.prompts = prompts
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 != 0:  # Only run every 5 epochs
            return
        
        try:    
            device = pl_module.device
            tokenizer = pl_module.tokenizer
            
            # Process the first prompt
            prompt = self.prompts[0]
            
            # Tokenize
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            # Generate with intermediate outputs - matching the model's generate method
            encoder_output = pl_module.encoder(input_ids)
            latent_output, latent = pl_module.latent_mapper(encoder_output)
            
            # Predict sequence length
            pred_lengths = pl_module._predict_length(None, latent)
            max_length = int(torch.max(pred_lengths).item() * 1.2)
            max_length = min(max_length, pl_module.config.max_position_embeddings)
            
            # Initial prediction
            logits, _ = pl_module.decoder(latent_output)
            preds = torch.argmax(logits, dim=-1)
            
            # Create length mask
            length_mask = torch.arange(preds.size(1), device=device).unsqueeze(0) < pred_lengths.unsqueeze(1)
            
            # Save all intermediate outputs
            all_outputs = [tokenizer.decode(preds[0], skip_special_tokens=True)]
            
            # Iterative refinement
            for step in range(pl_module.config.num_refinement_steps):
                # Re-encode with the predicted tokens
                encoder_output = pl_module.encoder(preds)
                latent_output, _ = pl_module.latent_mapper(encoder_output)
                
                # Get new predictions
                logits, _ = pl_module.decoder(latent_output, encoder_output)
                new_preds = torch.argmax(logits, dim=-1)
                
                # Update high-confidence predictions with length masking
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                update_mask = (confidence > pl_module.config.confidence_threshold) & length_mask
                preds = torch.where(update_mask, new_preds, preds)
                
                # Apply length mask to ensure tokens beyond predicted length are pad tokens
                preds = torch.where(length_mask, preds, 
                                    torch.tensor(tokenizer.pad_token_id, device=device))
                
                # Save the current output
                all_outputs.append(tokenizer.decode(preds[0], skip_special_tokens=True))
            
            # Log to TensorBoard
            log_text = f"Prompt: {prompt}\n\n"
            log_text += f"Predicted length: {pred_lengths[0].item()}\n\n"
            for i, output in enumerate(all_outputs):
                log_text += f"Step {i}: {output}\n\n"
            
            trainer.logger.experiment.add_text(
                "generation_refinement", 
                log_text,
                global_step=trainer.global_step
            )
        except Exception as e:
            print(f"Warning: Generation progress visualization failed with error: {e}")
            # Print more detailed error information
            import traceback
            traceback.print_exc()