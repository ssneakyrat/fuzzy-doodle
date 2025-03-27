# scripts/inference.py
import os
import argparse
import yaml
import torch
import time
from transformers import AutoTokenizer
import logging

# Set non-GUI backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from src.models.nar_model import LatentNARModel
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
    """Main inference function"""
    # Set up logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('nar_inference', log_file=args.log_file, level=log_level)
    
    # Load configurations
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)
    
    # Merge configs
    config = merge_configs_dict(model_config, train_config)
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {config['tokenizer_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = LatentNARModel.load_from_checkpoint(args.checkpoint_path, config=config)
    model.to(device)
    model.eval()
    
    # Process input prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    
    if args.input_file:
        logger.info(f"Reading prompts from file: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    
    if not prompts:
        prompts = [
            "This is a test",
            "The model generates",
            "Non-autoregressive generation is",
            "PyTorch Lightning helps"
        ]
    
    # Generate text for each prompt
    results = []
    
    for prompt in prompts:
        logger.info(f"Processing prompt: {prompt}")
        
        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Measure generation time
        start_time = time.time()
        
        # Generate text
        with torch.no_grad():
            output_ids = model.generate(input_ids)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        logger.info(f"Generated: {output_text}")
        logger.info(f"Generation time: {generation_time:.4f} seconds")
        
        # Save result
        results.append({
            "prompt": prompt,
            "generated_text": output_text,
            "generation_time": generation_time
        })
    
    # Save results to file if requested
    if args.output_file:
        logger.info(f"Saving results to {args.output_file}")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Generated: {result['generated_text']}\n")
                f.write(f"Generation time: {result['generation_time']:.4f} seconds\n")
                f.write("-" * 50 + "\n")
                
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with NAR Language Model")
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
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        help="File to save generation results"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="./inference.log",
        help="Path to log file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    args = parser.parse_args()
    
    main(args)