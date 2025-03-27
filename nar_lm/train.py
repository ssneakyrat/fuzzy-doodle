# scripts/train.py
import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# Set non-GUI backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from src.models.nar_model import LatentNARModel
from src.datasets.data_module import NARDataModule
from src.utils.callbacks import (
    LatentVisualizationCallback, 
    AttentionVisualizationCallback,
    GenerationProgressCallback
)

def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

class DotDict(dict):
    """Dictionary that also supports dot notation access"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
# Add this function to train.py and inference.py instead of the SimpleNamespace version
def merge_configs_dict(model_config, train_config):
    """Merge model and training configurations"""
    config = {**model_config, **train_config}
    # Convert to a dictionary that also supports dot notation
    return DotDict(config)

def main(args):

    # Set to medium precision
    torch.set_float32_matmul_precision('medium')

    """Main training function"""
    # Load configurations
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)
    
    # Merge configs
    config = merge_configs_dict(model_config, train_config)
    
    # Create directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Initialize logger
    logger = TensorBoardLogger(
        save_dir=config['log_dir'],
        name='nar_lm'
    )
    
    # Initialize model
    model = LatentNARModel(config)
    
    # Initialize data module
    data_module = NARDataModule(config)
    
    # Debug: print model and data module configurations
    print(f"[DEBUG] Model config: latent_size={config['latent_size']}, hidden_size={config['hidden_size']}")
    print(f"[DEBUG] Training config: batch_size={config['batch_size']}, num_train_samples={config['num_train_samples']}")
    print(f"[DEBUG] Validation config: num_val_samples={config['num_val_samples']}")
    
    # Define callbacks
    callbacks = [
        # Checkpoint callback
        ModelCheckpoint(
            dirpath=os.path.join(config['output_dir'], 'checkpoints'),
            filename='{epoch:02d}-{val_loss:.4f}',
            save_top_k=config['save_top_k'],
            monitor='val_loss',
            mode='min'
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval='step'),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )
    ]
    
    # Conditionally add visualization callbacks with try-except blocks
    try:
        callbacks.append(LatentVisualizationCallback())
        callbacks.append(AttentionVisualizationCallback())
        callbacks.append(GenerationProgressCallback())
    except Exception as e:
        print(f"[ERROR] Could not initialize visualization callbacks: {e}")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',  # Automatically detect GPU
        devices=1,
        precision=config['precision'],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config['gradient_clip_val'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        log_every_n_steps=config['log_every_n_steps']
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Debug information before testing
    print(f"[DEBUG] Starting model testing")
    # Setup test data explicitly to ensure it's initialized
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()
    print(f"[DEBUG] Testing model with {len(test_loader.dataset)} test samples")
    print(f"[DEBUG] Test batch size: {config['batch_size']}")
    
    # Test model
    test_result = trainer.test(model, datamodule=data_module)
    print(f"[DEBUG] Test results: {test_result}")
    
    # Save final model
    final_path = os.path.join(config['output_dir'], 'final_model.ckpt')
    trainer.save_checkpoint(final_path)
    print(f"[DEBUG] Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NAR Language Model")
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="./configs/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--train_config", 
        type=str, 
        default="./configs/training_config.yaml",
        help="Path to training configuration file"
    )
    args = parser.parse_args()
    
    main(args)