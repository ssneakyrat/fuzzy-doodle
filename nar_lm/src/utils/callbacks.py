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
            
            # Move to the same device as the model
            input_ids = batch['input_ids'].to(pl_module.device)
            
            # Forward pass to get latent representations
            with torch.no_grad():
                encoder_output = pl_module.encoder(input_ids)
                _, latent = pl_module.latent_mapper(encoder_output)
            
            # Visualize latent space (first 2 dimensions)
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Take a subset of samples for visualization
            latent_subset = latent[:self.num_samples].cpu().numpy()
            
            # PCA-like 2D visualization
            for i, sample_latent in enumerate(latent_subset):
                # Average across sequence length
                avg_latent = np.mean(sample_latent, axis=0)
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


class AttentionVisualizationCallback(pl.Callback):
    """Visualize attention patterns during training"""
    
    def __init__(self, prompts=["This is a test visualization"]):
        super().__init__()
        self.prompts = prompts
    
    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            device = pl_module.device
            tokenizer = pl_module.tokenizer
            
            # Process the first prompt
            prompt = self.prompts[0]
            
            # Tokenize
            tokens = tokenizer.tokenize(prompt)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            # Extract attention weights (we need to modify the model to save attention weights)
            # This is a placeholder for the actual implementation
            attention_weights = torch.rand(
                (1, pl_module.config.encoder_layers, pl_module.config.num_attention_heads, 
                 input_ids.size(1), input_ids.size(1))
            )
            
            # Create and log attention heatmap
            layer_idx = 0  # Visualize first layer
            head_idx = 0   # Visualize first attention head
            
            att_map = attention_weights[0, layer_idx, head_idx].cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(att_map, cmap='viridis')
            fig.colorbar(im, ax=ax)
            
            # Add labels
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens)
            
            ax.set_title(f"Attention Pattern (Layer {layer_idx+1}, Head {head_idx+1})")
            
            # Convert to image and log
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            image = np.array(Image.open(buf))
            plt.close(fig)
            
            trainer.logger.experiment.add_image(
                "attention_map", 
                image.transpose(2, 0, 1),  # Convert to channel-first format
                global_step=trainer.global_step
            )
        except Exception as e:
            print(f"Warning: Attention visualization failed with error: {e}")


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
            
            # Generate with intermediate outputs
            encoder_output = pl_module.encoder(input_ids)
            latent_output, _ = pl_module.latent_mapper(encoder_output)
            
            # Initial prediction
            logits = pl_module.decoder(latent_output)
            preds = torch.argmax(logits, dim=-1)
            
            # Save all intermediate outputs
            all_outputs = [tokenizer.decode(preds[0], skip_special_tokens=True)]
            
            # Iterative refinement
            for step in range(pl_module.config.num_refinement_steps):
                # Re-encode with the predicted tokens
                encoder_output = pl_module.encoder(preds)
                latent_output, _ = pl_module.latent_mapper(encoder_output)
                
                # Get new predictions
                logits = pl_module.decoder(latent_output, encoder_output)
                new_preds = torch.argmax(logits, dim=-1)
                
                # Update high-confidence predictions
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                mask = confidence > pl_module.config.confidence_threshold
                preds = torch.where(mask, new_preds, preds)
                
                # Save the current output
                all_outputs.append(tokenizer.decode(preds[0], skip_special_tokens=True))
            
            # Log to TensorBoard
            log_text = f"Prompt: {prompt}\n\n"
            for i, output in enumerate(all_outputs):
                log_text += f"Step {i}: {output}\n\n"
            
            trainer.logger.experiment.add_text(
                "generation_refinement", 
                log_text,
                global_step=trainer.global_step
            )
        except Exception as e:
            print(f"Warning: Generation progress visualization failed with error: {e}")