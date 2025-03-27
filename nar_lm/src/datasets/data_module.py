import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import torch
from transformers import AutoTokenizer
from functools import partial
import logging
from datasets import load_dataset, Dataset
import random

from src.utils.logger import setup_logger
from src.datasets.text_dataset import SimpleTextDataset

# Set up module logger
logger = setup_logger(__name__)

class NARDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.tokenizer = None
        
    def prepare_data(self):
        """Download data or prepare data if needed"""
        # Download tokenizer
        logger.info(f"Downloading tokenizer {self.config.tokenizer_name}")
        AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        
        # If dataset_name is provided, download it
        if hasattr(self.config, 'dataset_name') and self.config.dataset_name:
            dataset_name = self.config.dataset_name
            dataset_config = getattr(self.config, 'dataset_config', None)
            
            # Special handling for datasets that require configs
            if dataset_name == "wikitext" and not dataset_config:
                dataset_config = "wikitext-103-v1"  # Default config
                logger.info(f"Using default config '{dataset_config}' for wikitext dataset")
            
            logger.info(f"Downloading dataset {dataset_name}" + 
                       (f" with config {dataset_config}" if dataset_config else ""))
            
            try:
                if dataset_config:
                    # Just to trigger download
                    load_dataset(dataset_name, dataset_config, split="train[:1%]")
                else:
                    # Just to trigger download
                    load_dataset(dataset_name, split="train[:1%]")
                    
                logger.info(f"Dataset {dataset_name} downloaded successfully")
            except Exception as e:
                logger.error(f"Error downloading dataset {dataset_name}: {e}")
        
    def setup(self, stage=None):
        """Setup datasets for training, validation, and testing"""
        logger.info(f"Setting up for stage={stage}")
        
        if self.tokenizer is None:
            logger.info(f"Initializing tokenizer {self.config.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token={self.tokenizer.pad_token}")
        
        # Create datasets based on stage
        if stage == 'fit' or stage is None:
            logger.info("Setting up train and validation datasets")
            
            # Try to load from files first (keeping backward compatibility)
            train_texts = self._load_texts(self.config.train_file) if hasattr(self.config, 'train_file') else None
            val_texts = self._load_texts(self.config.val_file) if hasattr(self.config, 'val_file') else None
            
            # If no data files, try to load from dataset_name
            if train_texts is None and hasattr(self.config, 'dataset_name') and self.config.dataset_name:
                dataset_name = self.config.dataset_name
                dataset_config = getattr(self.config, 'dataset_config', None)
                
                # Format the dataset name with config if provided
                dataset_id = dataset_name
                if dataset_config:
                    logger.info(f"Loading real-world dataset: {dataset_name}/{dataset_config}")
                    dataset_id = f"{dataset_name}/{dataset_config}"
                else:
                    logger.info(f"Loading real-world dataset: {dataset_name}")
                
                train_texts, val_texts = self._load_real_dataset(
                    dataset_id,
                    num_train=getattr(self.config, 'num_train_samples', 10000),
                    num_val=getattr(self.config, 'num_val_samples', 1000)
                )
                logger.info(f"Loaded {len(train_texts)} train samples and {len(val_texts)} validation samples")
            
            # If still no data, fall back to synthetic data (for testing)
            if train_texts is None:
                logger.info(f"Creating synthetic data - train={self.config.num_train_samples}, val={self.config.num_val_samples}")
                train_texts = self._create_synthetic_data(self.config.num_train_samples)
                val_texts = self._create_synthetic_data(self.config.num_val_samples)
            
            # Create dataset instances
            max_len = self.config.max_position_embeddings
            logger.debug(f"Using max_length={max_len}")
            
            logger.info(f"Creating train dataset with {len(train_texts)} texts")
            self.train_dataset = SimpleTextDataset(
                train_texts, 
                self.tokenizer, 
                max_len
            )
            
            logger.info(f"Creating validation dataset with {len(val_texts)} texts")
            self.val_dataset = SimpleTextDataset(
                val_texts, 
                self.tokenizer, 
                max_len
            )
            
        if stage == 'test' or stage is None:
            logger.info("Setting up test dataset")
            test_texts = self._load_texts(self.config.test_file) if hasattr(self.config, 'test_file') else None
            
            # Try to load from dataset_name if available
            if test_texts is None and hasattr(self.config, 'dataset_name') and self.config.dataset_name:
                dataset_name = self.config.dataset_name
                dataset_config = getattr(self.config, 'dataset_config', None)
                
                # Format the dataset name with config if provided
                dataset_id = dataset_name
                if dataset_config:
                    logger.info(f"Loading real-world dataset for testing: {dataset_name}/{dataset_config}")
                    dataset_id = f"{dataset_name}/{dataset_config}"
                else:
                    logger.info(f"Loading real-world dataset for testing: {dataset_name}")
                
                _, _, test_texts = self._load_real_dataset(
                    dataset_id,
                    num_train=0,
                    num_val=0,
                    num_test=getattr(self.config, 'num_test_samples', 1000),
                    include_test=True
                )
                logger.info(f"Loaded {len(test_texts)} test samples from dataset")
            
            # Fall back to synthetic if needed
            if test_texts is None:
                logger.info(f"Creating synthetic test data - {self.config.num_test_samples} samples")
                test_texts = self._create_synthetic_data(self.config.num_test_samples)
                
            logger.info(f"Creating test dataset with {len(test_texts)} texts")
            self.test_dataset = SimpleTextDataset(
                test_texts, 
                self.tokenizer, 
                self.config.max_position_embeddings
            )
    
    def _load_real_dataset(self, dataset_name, num_train=10000, num_val=1000, num_test=1000, include_test=False):
        """Load a real dataset from HuggingFace datasets library"""
        try:
            logger.info(f"Loading dataset {dataset_name}")
            
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
            
            # Check if the dataset has train/validation/test splits
            has_train = 'train' in dataset
            has_val = 'validation' in dataset
            has_test = 'test' in dataset

            # Determine text column (common text columns in datasets)
            text_columns = ['text', 'content', 'document', 'sentence', 'article']
            text_column = None
            
            # Find the first available text column
            sample = dataset[list(dataset.keys())[0]][0]  # Get first item from first split
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
                return None, None, None if include_test else None, None
            
            logger.info(f"Using '{text_column}' as text column")
            
            # Extract texts from dataset
            train_texts = []
            val_texts = []
            test_texts = []
            
            # Load train split if available, otherwise use default split
            if has_train and num_train > 0:
                # Get a subset of the training data (up to num_train samples)
                train_split = dataset['train'].shuffle(seed=42).select(range(min(num_train, len(dataset['train']))))
                train_texts = [item[text_column] for item in train_split if len(item[text_column].strip()) > 10]
                logger.info(f"Loaded {len(train_texts)} training samples from dataset")
            
            # Load validation split if available, otherwise use a part of train
            if has_val and num_val > 0:
                val_split = dataset['validation'].shuffle(seed=42).select(range(min(num_val, len(dataset['validation']))))
                val_texts = [item[text_column] for item in val_split if len(item[text_column].strip()) > 10]
                logger.info(f"Loaded {len(val_texts)} validation samples from dataset")
            elif has_train and num_val > 0:
                # If no validation split, use a part of the training data (that wasn't used for training)
                offset = min(num_train, len(dataset['train']))
                max_val = min(num_val, max(0, len(dataset['train']) - offset))
                if max_val > 0:
                    val_split = dataset['train'].shuffle(seed=43).select(range(offset, offset + max_val))
                    val_texts = [item[text_column] for item in val_split if len(item[text_column].strip()) > 10]
                    logger.info(f"Used {len(val_texts)} training samples as validation")
            
            # Load test split if requested and available
            if include_test and has_test and num_test > 0:
                test_split = dataset['test'].shuffle(seed=44).select(range(min(num_test, len(dataset['test']))))
                test_texts = [item[text_column] for item in test_split if len(item[text_column].strip()) > 10]
                logger.info(f"Loaded {len(test_texts)} test samples from dataset")
            
            # Process texts to ensure quality
            train_texts = self._filter_and_preprocess_texts(train_texts)
            val_texts = self._filter_and_preprocess_texts(val_texts)
            test_texts = self._filter_and_preprocess_texts(test_texts) if include_test else None
            
            if include_test:
                return train_texts, val_texts, test_texts
            else:
                return train_texts, val_texts
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}", exc_info=True)
            return None, None, None if include_test else None, None
    
    def _filter_and_preprocess_texts(self, texts):
        """Filter and preprocess texts for quality"""
        if not texts:
            return []
            
        # Remove texts that are too short
        min_length = 20  # Minimum character length
        filtered_texts = [text for text in texts if len(text.strip()) >= min_length]
        
        # Remove duplicates
        unique_texts = list(set(filtered_texts))
        
        # Trim down to a more manageable length if needed
        max_char_length = getattr(self.config, 'max_char_length', 2000)
        trimmed_texts = [text[:max_char_length] for text in unique_texts]
        
        logger.info(f"Text preprocessing: {len(texts)} → {len(filtered_texts)} → {len(unique_texts)} → {len(trimmed_texts)}")
        
        return trimmed_texts
    
    def _load_texts(self, file_path):
        """Load texts from file (original method preserved)"""
        if file_path and os.path.exists(file_path):
            logger.info(f"Loading texts from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Filter out empty lines
            texts = [line.strip() for line in lines if len(line.strip()) > 10]
            logger.info(f"Loaded {len(texts)} texts")
            return texts
        return None
    
    def _create_synthetic_data(self, num_samples):
        """Create synthetic data for testing (original method preserved)"""
        logger.info(f"Creating {num_samples} synthetic samples")
        synthetic_texts = [
            "This is a synthetic text for training.",
            "We are using PyTorch Lightning to train our model efficiently.",
            "This model has encoder and decoder components with a latent space.",
            "Non-autoregressive generation produces all tokens in parallel rather than sequentially like traditional autoregressive language models.",
            "TensorBoard logging helps us track the training progress and text generation quality across multiple experiments and iterations."
        ]
        
        # Create more variants with different lengths
        more_variants = []
        for text in synthetic_texts:
            # Short version
            more_variants.append(text.split('.')[0] + ".")
            # Extended version
            more_variants.append(text + " This extension adds more context and increases sequence length variability for better training.")
        
        # Combine all variants
        all_texts = synthetic_texts + more_variants
        
        # Repeat to reach desired count
        texts = all_texts * (num_samples // len(all_texts) + 1)
        return texts[:num_samples]
    
    @staticmethod
    def collate_fn(batch, tokenizer, max_length=None):
        """Collate function (original method preserved)"""
        try:
            # Extract raw texts from the batch
            texts = [item['text'] for item in batch]
            
            # Debug info with logger
            if len(batch) > 0:
                logger.debug(f"Processing batch of {len(texts)} texts")
            
            # Perform tokenization with padding in a single step
            encodings = tokenizer(
                texts,
                padding=True,
                truncation=True if max_length else False,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Use input_ids as labels for reconstruction task
            labels = encodings['input_ids'].clone()
            
            # Calculate sequence lengths from attention mask
            seq_lengths = encodings['attention_mask'].sum(dim=1).long()
            
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': labels,
                'seq_length': seq_lengths
            }
        except Exception as e:
            logger.error(f"Error in collate_fn: {e}", exc_info=True)
            # Return empty tensors as a fallback to avoid crashing the training
            return {
                'input_ids': torch.zeros((1, 1), dtype=torch.long),
                'attention_mask': torch.zeros((1, 1), dtype=torch.long),
                'labels': torch.zeros((1, 1), dtype=torch.long),
                'seq_length': torch.ones((1,), dtype=torch.long)
            }
    
    def train_dataloader(self):
        logger.info(f"Creating train dataloader with batch_size={self.config.batch_size}")
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=(self.config.num_workers > 0),
            pin_memory=True,
            collate_fn=partial(self.collate_fn, 
                             tokenizer=self.tokenizer, 
                             max_length=self.config.max_position_embeddings)
        )
    
    def val_dataloader(self):
        logger.info(f"Creating validation dataloader with batch_size={self.config.batch_size}")
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            persistent_workers=(self.config.num_workers > 0),
            pin_memory=True,
            collate_fn=partial(self.collate_fn, 
                             tokenizer=self.tokenizer, 
                             max_length=self.config.max_position_embeddings)
        )
    
    def test_dataloader(self):
        logger.info(f"Creating test dataloader with batch_size={self.config.batch_size}")
        if self.test_dataset is None:
            logger.warning("test_dataset is None, setting up explicitly")
            self.setup(stage='test')
            
        if self.test_dataset is None:
            raise ValueError("test_dataset is still None after explicit setup!")
            
        return DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            persistent_workers=(self.config.num_workers > 0),
            pin_memory=True,
            collate_fn=partial(self.collate_fn, 
                             tokenizer=self.tokenizer, 
                             max_length=self.config.max_position_embeddings)
        )