# scripts/analyze_dataset.py
import argparse
from datasets import load_dataset
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import logging
from collections import Counter
import re

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset(dataset_name, num_samples=1000, output_dir="./dataset_analysis", tokenizer_name="bert-base-uncased"):
    """Analyze a dataset and save visualizations"""
    logger.info(f"Analyzing dataset: {dataset_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    try:
        # Split dataset name and config if provided
        parts = dataset_name.split('/')
        dataset_config = None
        
        if len(parts) > 1 and '/' in dataset_name:
            dataset_name, dataset_config = parts[0], parts[1]
        
        # Special handling for datasets that require configs
        if dataset_name == "wikitext" and not dataset_config:
            dataset_config = "wikitext-103-v1"  # Default config
            logger.info(f"Using default config '{dataset_config}' for wikitext dataset")
        
        # Load dataset with appropriate config
        if dataset_config:
            logger.info(f"Loading dataset with name={dataset_name}, config={dataset_config}")
            dataset = load_dataset(dataset_name, dataset_config)
        else:
            logger.info(f"Loading dataset with name={dataset_name}")
            dataset = load_dataset(dataset_name)
        
        logger.info(f"Dataset loaded: {dataset}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Get splits
    splits = list(dataset.keys())
    logger.info(f"Available splits: {splits}")
    
    # Use 'train' split or the first available split
    split_name = 'train' if 'train' in splits else splits[0]
    logger.info(f"Using split: {split_name}")
    
    # Sample data
    try:
        data = dataset[split_name].shuffle(seed=42).select(range(min(num_samples, len(dataset[split_name]))))
        logger.info(f"Sampled {len(data)} examples")
    except Exception as e:
        logger.error(f"Error sampling data: {e}")
        return
    
    # Determine text column (common text columns in datasets)
    text_columns = ['text', 'content', 'document', 'sentence', 'article']
    text_column = None
    
    # Find the first available text column
    sample = data[0]
    for col in text_columns:
        if col in sample:
            text_column = col
            break
    
    if text_column is None:
        # If no common text column found, use the first string column
        for key, value in sample.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                text_column = key
                break
    
    if text_column is None:
        logger.error(f"Could not find text column in dataset {dataset_name}")
        return
    
    logger.info(f"Using '{text_column}' as text column")
    
    # Extract texts
    texts = [item[text_column] for item in data if len(item[text_column].strip()) > 0]
    logger.info(f"Extracted {len(texts)} non-empty texts")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Loaded tokenizer: {tokenizer_name}")
    
    # Analyze text lengths
    char_lengths = [len(text) for text in texts]
    
    # Analyze token lengths
    token_lengths = []
    for text in tqdm(texts, desc="Tokenizing texts"):
        tokens = tokenizer.encode(text)
        token_lengths.append(len(tokens))
    
    # Calculate statistics
    avg_char_len = np.mean(char_lengths)
    med_char_len = np.median(char_lengths)
    max_char_len = np.max(char_lengths)
    
    avg_token_len = np.mean(token_lengths)
    med_token_len = np.median(token_lengths)
    max_token_len = np.max(token_lengths)
    
    logger.info(f"Character length stats: avg={avg_char_len:.1f}, median={med_char_len}, max={max_char_len}")
    logger.info(f"Token length stats: avg={avg_token_len:.1f}, median={med_token_len}, max={max_token_len}")
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(char_lengths, bins=50)
    plt.axvline(x=avg_char_len, color='r', linestyle='--', label=f'Mean: {avg_char_len:.1f}')
    plt.axvline(x=med_char_len, color='g', linestyle='--', label=f'Median: {med_char_len}')
    plt.title(f'Character Length Distribution (max={max_char_len})')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(token_lengths, bins=50)
    plt.axvline(x=avg_token_len, color='r', linestyle='--', label=f'Mean: {avg_token_len:.1f}')
    plt.axvline(x=med_token_len, color='g', linestyle='--', label=f'Median: {med_token_len}')
    plt.title(f'Token Length Distribution (max={max_token_len})')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_distribution.png'))
    logger.info(f"Saved length distribution plot to {os.path.join(output_dir, 'length_distribution.png')}")
    
    # Analyze most common tokens
    all_tokens = []
    for text in tqdm(texts[:min(100, len(texts))], desc="Analyzing common tokens"):
        # Convert to lowercase and split by whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        all_tokens.extend(words)
    
    token_counter = Counter(all_tokens)
    most_common = token_counter.most_common(30)
    
    # Save token frequency
    plt.figure(figsize=(12, 8))
    words, counts = zip(*most_common)
    plt.barh(range(len(words)), counts, align='center')
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frequency')
    plt.title('Most Common Tokens')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_frequency.png'))
    logger.info(f"Saved token frequency plot to {os.path.join(output_dir, 'token_frequency.png')}")
    
    # Sample a few data points for inspection
    with open(os.path.join(output_dir, 'sample_texts.txt'), 'w', encoding='utf-8') as f:
        f.write(f"DATASET: {dataset_name}\n")
        f.write(f"TEXT COLUMN: {text_column}\n\n")
        f.write("=" * 80 + "\n\n")
        
        for i, text in enumerate(texts[:5]):
            f.write(f"SAMPLE {i+1}:\n")
            f.write("-" * 40 + "\n")
            f.write(text[:2000] + ("..." if len(text) > 2000 else ""))
            f.write("\n\n")
            f.write("=" * 80 + "\n\n")
    
    logger.info(f"Saved sample texts to {os.path.join(output_dir, 'sample_texts.txt')}")
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Split: {split_name}\n")
        f.write(f"Number of samples analyzed: {len(texts)}\n\n")
        
        f.write("Character Length Statistics:\n")
        f.write(f"  Mean: {avg_char_len:.1f}\n")
        f.write(f"  Median: {med_char_len}\n")
        f.write(f"  Max: {max_char_len}\n\n")
        
        f.write("Token Length Statistics:\n")
        f.write(f"  Mean: {avg_token_len:.1f}\n")
        f.write(f"  Median: {med_token_len}\n")
        f.write(f"  Max: {max_token_len}\n\n")
        
        f.write("Most Common Tokens:\n")
        for word, count in most_common:
            f.write(f"  {word}: {count}\n")
    
    logger.info(f"Saved statistics to {os.path.join(output_dir, 'statistics.txt')}")
    
    # Recommend max sequence length
    recommended_len = int(np.percentile(token_lengths, 95))  # 95th percentile
    logger.info(f"Recommended max_position_embeddings: {recommended_len} (covers 95% of samples)")
    
    # Generate recommendations
    with open(os.path.join(output_dir, 'recommendations.txt'), 'w') as f:
        f.write("Model Configuration Recommendations:\n\n")
        
        f.write(f"1. max_position_embeddings: {recommended_len}\n")
        f.write(f"   This covers 95% of sample lengths.\n\n")
        
        f.write(f"2. max_char_length: {int(np.percentile(char_lengths, 90))}\n")
        f.write(f"   This keeps 90% of samples in their entirety.\n\n")
        
        f.write("3. Consider these settings for training_config.yaml:\n")
        f.write(f"   num_train_samples: {min(10000, len(dataset[split_name]))}\n")
        f.write(f"   batch_size: {8 if recommended_len > 256 else 16}\n")
        f.write(f"   steps_per_epoch: {min(10000, len(dataset[split_name])) // (8 if recommended_len > 256 else 16)}\n")
    
    logger.info(f"Saved recommendations to {os.path.join(output_dir, 'recommendations.txt')}")
    logger.info("Analysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dataset for NAR Language Model")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="wikitext/wikitext-103-v1",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=1000,
        help="Number of samples to analyze"
    )
    parser.add_argument(
        "--tokenizer_name", 
        type=str, 
        default="bert-base-uncased",
        help="Tokenizer to use for analysis"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./dataset_analysis",
        help="Directory to save analysis results"
    )
    args = parser.parse_args()
    
    analyze_dataset(
        args.dataset_name, 
        args.num_samples, 
        args.output_dir,
        args.tokenizer_name
    )