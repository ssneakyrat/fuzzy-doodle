# scripts/test_dataset_pipeline.py
import os
import argparse
import yaml
import torch
import time
from transformers import AutoTokenizer
import logging
import sys

# Add the parent directory to the Python path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.data_module import NARDataModule
from src.utils.logger import setup_logger

def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

class DotDict(dict):
    """Dictionary that also supports dot notation access"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def merge_configs_dict(model_config, train_config):
    """Merge model and training configurations"""
    config = {**model_config, **train_config}
    # Convert to a dictionary that also supports dot notation
    return DotDict(config)

def main(args):
    """Test the dataset pipeline"""
    # Set up logger
    logger = setup_logger('test_dataset', level=logging.INFO, disable_info=False)
    
    # Load configurations
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)
    
    # Add dataset_name and config if provided
    if args.dataset_name:
        parts = args.dataset_name.split('/')
        if len(parts) > 1 and '/' in args.dataset_name:
            train_config['dataset_name'] = parts[0]
            train_config['dataset_config'] = parts[1]
            logger.info(f"Using dataset: {parts[0]} with config: {parts[1]}")
        else:
            train_config['dataset_name'] = args.dataset_name
            
            # Special handling for wikitext which requires a config
            if args.dataset_name == "wikitext":
                train_config['dataset_config'] = "wikitext-103-v1"  # Default config
                logger.info(f"Using dataset: {args.dataset_name} with default config: {train_config['dataset_config']}")
            else:
                logger.info(f"Using dataset: {args.dataset_name}")
    
    # Set the number of samples
    train_config['num_train_samples'] = args.num_samples
    train_config['num_val_samples'] = max(1, args.num_samples // 10)
    
    # Merge configs
    config = merge_configs_dict(model_config, train_config)
    
    logger.info(f"Testing dataset pipeline with dataset: {config.get('dataset_name', 'None')}")
    logger.info(f"Number of samples: {config['num_train_samples']}")
    
    # Initialize data module
    logger.info("Initializing data module...")
    data_module = NARDataModule(config)
    
    # Prepare data
    logger.info("Preparing data...")
    start_time = time.time()
    data_module.prepare_data()
    logger.info(f"Data preparation took {time.time() - start_time:.2f} seconds")
    
    # Setup data
    logger.info("Setting up data...")
    start_time = time.time()
    data_module.setup(stage='fit')
    logger.info(f"Data setup took {time.time() - start_time:.2f} seconds")
    
    # Create train and validation dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    
    # Test train dataloader
    logger.info("Testing train dataloader...")
    logger.info(f"Train dataset size: {len(data_module.train_dataset)}")
    
    batch = next(iter(train_dataloader))
    logger.info(f"Sample batch keys: {batch.keys()}")
    logger.info(f"Input shape: {batch['input_ids'].shape}")
    logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")
    logger.info(f"Label shape: {batch['labels'].shape}")
    logger.info(f"Sequence length shape: {batch['seq_length'].shape}")
    
    # Test sequence lengths
    seq_lengths = batch['seq_length'].tolist()
    logger.info(f"Sequence lengths (sample): {seq_lengths[:5]}...")
    
    # Decode a sample input
    input_ids = batch['input_ids'][0]
    tokenizer = data_module.tokenizer
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    logger.info(f"Sample text: {decoded_text[:100]}...")
    
    # Summarize statistics
    total_train_batches = len(train_dataloader)
    total_val_batches = len(val_dataloader)
    
    logger.info(f"Train batches: {total_train_batches}, samples: {len(data_module.train_dataset)}")
    logger.info(f"Val batches: {total_val_batches}, samples: {len(data_module.val_dataset)}")
    
    # Calculate memory usage
    batch_size_bytes = 0
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_size_bytes += v.element_size() * v.nelement()
    
    logger.info(f"Memory usage per batch: {batch_size_bytes / (1024 * 1024):.2f} MB")
    
    # Traverse the entire train dataloader to check for errors
    logger.info("Traversing train dataloader to check for errors...")
    start_time = time.time()
    num_samples = 0
    
    try:
        for i, batch in enumerate(train_dataloader):
            num_samples += batch['input_ids'].size(0)
            if i % 10 == 0:
                logger.info(f"Processed {i}/{total_train_batches} batches, {num_samples} samples")
            
            # Check for valid sequence lengths
            if torch.any(batch['seq_length'] <= 0):
                logger.warning(f"Found invalid sequence length in batch {i}: {batch['seq_length']}")
                
    except Exception as e:
        logger.error(f"Error during dataloader traversal: {e}")
    
    logger.info(f"Full dataloader traversal took {time.time() - start_time:.2f} seconds")
    logger.info(f"Successfully processed {num_samples} samples")
    
    logger.info("Dataset pipeline test complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the dataset pipeline")
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
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default=None,
        help="HuggingFace dataset name (e.g., 'wikitext' or 'wikitext/wikitext-103-v1')"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of samples to test with"
    )
    args = parser.parse_args()
    
    main(args)