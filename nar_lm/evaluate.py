# scripts/evaluate_model.py
import os
import argparse
import yaml
import torch
import time
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import your model
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
    return DotDict(config)

def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score between reference and hypothesis"""
    if not hypothesis:
        return 0.0
    
    # Tokenize sentences into words
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    # Apply smoothing for short sequences
    smoothie = SmoothingFunction().method1
    
    # Calculate BLEU scores at different n-gram levels
    try:
        bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        return {
            'bleu1': bleu1,
            'bleu2': bleu2,
            'bleu4': bleu4
        }
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return {
            'bleu1': 0,
            'bleu2': 0,
            'bleu4': 0
        }

def calculate_rouge(reference, hypothesis):
    """Calculate ROUGE scores between reference and hypothesis"""
    if not hypothesis:
        return {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        return {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}

def evaluate_nar_model(model, tokenizer, test_samples, device, refinement_steps=None):
    """Evaluate NAR model with different number of refinement steps"""
    results = []
    
    # Original refinement steps
    original_steps = model.config.num_refinement_steps
    
    # Set custom refinement steps if provided
    if refinement_steps is not None:
        model.config.num_refinement_steps = refinement_steps
    
    for sample in tqdm(test_samples, desc=f"Evaluating with {model.config.num_refinement_steps} refinement steps"):
        prompt = sample['prompt']
        reference = sample.get('reference', prompt)  # Use prompt as reference if not provided
        
        # Tokenize prompt
        input_data = tokenizer(prompt, return_tensors="pt")
        input_ids = input_data.input_ids.to(device)
        attention_mask = input_data.attention_mask.to(device)
        
        # Measure generation time
        start_time = time.time()
        
        # Generate text
        with torch.no_grad():
            output_ids = model.generate(input_ids, attention_mask)
            
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Calculate metrics
        bleu_scores = calculate_bleu(reference, output_text)
        rouge_scores = calculate_rouge(reference, output_text)
        
        # Store result
        result = {
            'prompt': prompt,
            'reference': reference,
            'output': output_text,
            'time': generation_time,
            'refinement_steps': model.config.num_refinement_steps,
            'bleu1': bleu_scores['bleu1'],
            'bleu2': bleu_scores['bleu2'],
            'bleu4': bleu_scores['bleu4'],
            'rouge1': rouge_scores['rouge-1']['f'],
            'rouge2': rouge_scores['rouge-2']['f'],
            'rougeL': rouge_scores['rouge-l']['f']
        }
        
        results.append(result)
    
    # Restore original refinement steps
    model.config.num_refinement_steps = original_steps
    
    return results

def evaluate_ar_model(model, tokenizer, test_samples, device):
    """Evaluate autoregressive model for comparison"""
    results = []
    
    for sample in tqdm(test_samples, desc="Evaluating AR model"):
        prompt = sample['prompt']
        reference = sample.get('reference', prompt)  # Use prompt as reference if not provided
        
        # Tokenize prompt
        input_data = tokenizer(prompt, return_tensors="pt")
        input_ids = input_data.input_ids.to(device)
        
        # Measure generation time
        start_time = time.time()
        
        # Generate text
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_length=256,
                num_return_sequences=1,
                do_sample=False  # Use greedy decoding for fair comparison
            )
            
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Calculate metrics
        bleu_scores = calculate_bleu(reference, output_text)
        rouge_scores = calculate_rouge(reference, output_text)
        
        # Store result
        result = {
            'prompt': prompt,
            'reference': reference,
            'output': output_text,
            'time': generation_time,
            'bleu1': bleu_scores['bleu1'],
            'bleu2': bleu_scores['bleu2'],
            'bleu4': bleu_scores['bleu4'],
            'rouge1': rouge_scores['rouge-1']['f'],
            'rouge2': rouge_scores['rouge-2']['f'],
            'rougeL': rouge_scores['rouge-l']['f']
        }
        
        results.append(result)
    
    return results

def plot_evaluation_results(nar_results_list, ar_results=None, output_dir='./evaluation_results'):
    """Plot evaluation results and save visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group NAR results by refinement steps
    nar_results_by_steps = {}
    for results in nar_results_list:
        if results:
            steps = results[0]['refinement_steps']
            nar_results_by_steps[steps] = results
    
    # Prepare data for plotting
    refinement_steps = sorted(nar_results_by_steps.keys())
    
    # Average metrics by refinement steps
    avg_metrics = {
        'time': [],
        'bleu1': [],
        'bleu2': [],
        'bleu4': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for steps in refinement_steps:
        results = nar_results_by_steps[steps]
        avg_metrics['time'].append(np.mean([r['time'] for r in results]))
        avg_metrics['bleu1'].append(np.mean([r['bleu1'] for r in results]))
        avg_metrics['bleu2'].append(np.mean([r['bleu2'] for r in results]))
        avg_metrics['bleu4'].append(np.mean([r['bleu4'] for r in results]))
        avg_metrics['rouge1'].append(np.mean([r['rouge1'] for r in results]))
        avg_metrics['rouge2'].append(np.mean([r['rouge2'] for r in results]))
        avg_metrics['rougeL'].append(np.mean([r['rougeL'] for r in results]))
    
    # Add AR model metrics if available
    ar_avg_metrics = None
    if ar_results:
        ar_avg_metrics = {
            'time': np.mean([r['time'] for r in ar_results]),
            'bleu1': np.mean([r['bleu1'] for r in ar_results]),
            'bleu2': np.mean([r['bleu2'] for r in ar_results]),
            'bleu4': np.mean([r['bleu4'] for r in ar_results]),
            'rouge1': np.mean([r['rouge1'] for r in ar_results]),
            'rouge2': np.mean([r['rouge2'] for r in ar_results]),
            'rougeL': np.mean([r['rougeL'] for r in ar_results])
        }
    
    # Plot 1: Generation Time vs. Refinement Steps
    plt.figure(figsize=(10, 6))
    plt.plot(refinement_steps, avg_metrics['time'], 'o-', label='NAR Model')
    if ar_avg_metrics:
        plt.axhline(y=ar_avg_metrics['time'], color='r', linestyle='--', label='AR Model')
    plt.xlabel('Number of Refinement Steps')
    plt.ylabel('Average Generation Time (seconds)')
    plt.title('Generation Time vs. Refinement Steps')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'generation_time.png'))
    
    # Plot 2: BLEU Scores vs. Refinement Steps
    plt.figure(figsize=(10, 6))
    plt.plot(refinement_steps, avg_metrics['bleu1'], 'o-', label='BLEU-1')
    plt.plot(refinement_steps, avg_metrics['bleu2'], 's-', label='BLEU-2')
    plt.plot(refinement_steps, avg_metrics['bleu4'], '^-', label='BLEU-4')
    if ar_avg_metrics:
        plt.axhline(y=ar_avg_metrics['bleu1'], color='r', linestyle='--', label='AR BLEU-1')
        plt.axhline(y=ar_avg_metrics['bleu4'], color='darkred', linestyle='--', label='AR BLEU-4')
    plt.xlabel('Number of Refinement Steps')
    plt.ylabel('Average BLEU Score')
    plt.title('BLEU Scores vs. Refinement Steps')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'bleu_scores.png'))
    
    # Plot 3: ROUGE Scores vs. Refinement Steps
    plt.figure(figsize=(10, 6))
    plt.plot(refinement_steps, avg_metrics['rouge1'], 'o-', label='ROUGE-1')
    plt.plot(refinement_steps, avg_metrics['rouge2'], 's-', label='ROUGE-2')
    plt.plot(refinement_steps, avg_metrics['rougeL'], '^-', label='ROUGE-L')
    if ar_avg_metrics:
        plt.axhline(y=ar_avg_metrics['rouge1'], color='r', linestyle='--', label='AR ROUGE-1')
        plt.axhline(y=ar_avg_metrics['rougeL'], color='darkred', linestyle='--', label='AR ROUGE-L')
    plt.xlabel('Number of Refinement Steps')
    plt.ylabel('Average ROUGE Score')
    plt.title('ROUGE Scores vs. Refinement Steps')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'rouge_scores.png'))
    
    # Plot 4: Quality vs. Speed Trade-off
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_metrics['time'], avg_metrics['rougeL'], s=100, c=refinement_steps, cmap='viridis')
    for i, steps in enumerate(refinement_steps):
        plt.annotate(f"{steps}", (avg_metrics['time'][i], avg_metrics['rougeL'][i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
    if ar_avg_metrics:
        plt.scatter(ar_avg_metrics['time'], ar_avg_metrics['rougeL'], s=200, c='red', marker='*', label='AR Model')
    plt.colorbar(label='Refinement Steps')
    plt.xlabel('Generation Time (seconds)')
    plt.ylabel('ROUGE-L Score')
    plt.title('Quality vs. Speed Trade-off')
    plt.grid(True, linestyle='--', alpha=0.5)
    if ar_avg_metrics:
        plt.legend()
    plt.savefig(os.path.join(output_dir, 'quality_speed_tradeoff.png'))
    
    # Create summary table and save as CSV
    summary_data = {
        'Refinement Steps': refinement_steps,
        'Avg Time (s)': avg_metrics['time'],
        'BLEU-1': avg_metrics['bleu1'],
        'BLEU-4': avg_metrics['bleu4'],
        'ROUGE-L': avg_metrics['rougeL']
    }
    
    if ar_avg_metrics:
        summary_data['AR Avg Time (s)'] = [ar_avg_metrics['time']] * len(refinement_steps)
        summary_data['AR BLEU-1'] = [ar_avg_metrics['bleu1']] * len(refinement_steps)
        summary_data['AR BLEU-4'] = [ar_avg_metrics['bleu4']] * len(refinement_steps)
        summary_data['AR ROUGE-L'] = [ar_avg_metrics['rougeL']] * len(refinement_steps)
        
        # Calculate speedup and relative quality
        summary_data['Speedup'] = ar_avg_metrics['time'] / np.array(avg_metrics['time'])
        summary_data['Quality Ratio'] = np.array(avg_metrics['rougeL']) / ar_avg_metrics['rougeL']
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    
    # Generate a detailed report for each test sample
    for steps in refinement_steps:
        results = nar_results_by_steps[steps]
        
        with open(os.path.join(output_dir, f'detailed_results_{steps}_steps.txt'), 'w') as f:
            for i, result in enumerate(results):
                f.write(f"Sample {i+1}:\n")
                f.write(f"Prompt: {result['prompt'][:100]}...\n")
                f.write(f"Reference: {result['reference'][:100]}...\n")
                f.write(f"Generated: {result['output'][:100]}...\n")
                f.write(f"Time: {result['time']:.4f} seconds\n")
                f.write(f"BLEU-1: {result['bleu1']:.4f}, BLEU-4: {result['bleu4']:.4f}\n")
                f.write(f"ROUGE-L: {result['rougeL']:.4f}\n")
                f.write("-" * 80 + "\n\n")
    
    print(f"Evaluation results saved to {output_dir}")

def load_test_samples(filename):
    """Load test samples from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    samples = []
    current_sample = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith('Prompt:'):
            if current_sample:
                samples.append(current_sample)
                current_sample = {}
            current_sample['prompt'] = line[len('Prompt:'):].strip()
        elif line.startswith('Reference:'):
            current_sample['reference'] = line[len('Reference:'):].strip()
    
    if current_sample:
        samples.append(current_sample)
    
    return samples

def main(args):
    """Main evaluation function"""
    # Set up logger
    logger = setup_logger('model_evaluation', level=logging.INFO)
    
    # Load configurations
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)
    
    # Merge configs
    config = merge_configs_dict(model_config, train_config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load NAR model
    logger.info(f"Loading NAR model from checkpoint: {args.nar_checkpoint}")
    nar_model = LatentNARModel.load_from_checkpoint(args.nar_checkpoint, config=config)
    nar_model.to(device)
    nar_model.eval()
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    
    # Load test samples
    if args.test_file:
        logger.info(f"Loading test samples from {args.test_file}")
        test_samples = load_test_samples(args.test_file)
    else:
        # Default test samples
        logger.info("Using default test samples")
        test_samples = [
            {
                'prompt': "This is a sample text to test the model's generation capabilities."
            },
            {
                'prompt': "Natural language processing has advanced significantly in recent years."
            },
            {
                'prompt': "The non-autoregressive model generates all tokens simultaneously, which is faster but sometimes less coherent than autoregressive models."
            },
            {
                'prompt': "When I was a child, I used to spend hours reading books about science and history. The knowledge I gained from those early experiences shaped my worldview."
            }
        ]
    
    logger.info(f"Loaded {len(test_samples)} test samples")
    
    # Evaluate NAR model with different refinement steps
    refinement_steps_to_test = args.refinement_steps if args.refinement_steps else [1, 3, 5, 7]
    logger.info(f"Testing with refinement steps: {refinement_steps_to_test}")
    
    nar_results_list = []
    for steps in refinement_steps_to_test:
        logger.info(f"Evaluating NAR model with {steps} refinement steps")
        results = evaluate_nar_model(nar_model, tokenizer, test_samples, device, steps)
        nar_results_list.append(results)
    
    # Evaluate AR model if provided
    ar_results = None
    if args.ar_model_name:
        logger.info(f"Loading AR model: {args.ar_model_name}")
        try:
            ar_model = AutoModelForCausalLM.from_pretrained(args.ar_model_name)
            ar_model.to(device)
            ar_model.eval()
            
            logger.info("Evaluating AR model")
            ar_results = evaluate_ar_model(ar_model, tokenizer, test_samples, device)
        except Exception as e:
            logger.error(f"Error loading or evaluating AR model: {e}")
    
    # Plot and save results
    output_dir = args.output_dir if args.output_dir else './evaluation_results'
    logger.info(f"Plotting and saving results to {output_dir}")
    plot_evaluation_results(nar_results_list, ar_results, output_dir)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NAR Language Model")
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
        "--nar_checkpoint", 
        type=str, 
        required=True,
        help="Path to NAR model checkpoint"
    )
    parser.add_argument(
        "--ar_model_name", 
        type=str, 
        default=None,
        help="Name of HuggingFace AR model for comparison"
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default=None,
        help="Path to test samples file"
    )
    parser.add_argument(
        "--refinement_steps", 
        type=int, 
        nargs="+",
        default=None,
        help="List of refinement steps to test"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    args = parser.parse_args()
    
    main(args)