import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import torch
from transformers import AutoTokenizer
from functools import partial

from src.datasets.text_dataset import SimpleTextDataset

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
        AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        
    def setup(self, stage=None):
        """Setup datasets for training, validation, and testing"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create datasets based on stage
        if stage == 'fit' or stage is None:
            train_texts = self._load_texts(self.config.train_file)
            val_texts = self._load_texts(self.config.val_file)
            
            # If no validation file, split from training
            if val_texts is None and train_texts is not None:
                split_idx = int(len(train_texts) * (1 - self.config.val_split))
                val_texts = train_texts[split_idx:]
                train_texts = train_texts[:split_idx]
            
            # If no data files, create synthetic data
            if train_texts is None:
                train_texts = self._create_synthetic_data(self.config.num_train_samples)
                val_texts = self._create_synthetic_data(self.config.num_val_samples)
            
            # Create datasets with dynamic sequence handling
            # Pass max_length as None for fully dynamic handling, or keep config's max_position_embeddings for cap
            max_len = self.config.max_position_embeddings
            
            self.train_dataset = SimpleTextDataset(
                train_texts, 
                self.tokenizer, 
                max_len  # Can set to None for full dynamic handling
            )
            
            self.val_dataset = SimpleTextDataset(
                val_texts, 
                self.tokenizer, 
                max_len
            )
            
        if stage == 'test' or stage is None:
            test_texts = self._load_texts(self.config.test_file)
            if test_texts is None:
                test_texts = self._create_synthetic_data(self.config.num_test_samples)
                
            self.test_dataset = SimpleTextDataset(
                test_texts, 
                self.tokenizer, 
                self.config.max_position_embeddings
            )
    
    def _load_texts(self, file_path):
        """Load texts from file"""
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Filter out empty lines
            texts = [line.strip() for line in lines if len(line.strip()) > 10]
            return texts
        return None
    
    def _create_synthetic_data(self, num_samples):
        """Create synthetic data for testing with varied lengths"""
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
    def collate_fn(batch, tokenizer):
        """Static collate function for dynamic batching to avoid pickling issues"""
        # Extract individual elements
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        seq_lengths = [item['seq_length'] for item in batch]
        
        # Use tokenizer's pad function for dynamic padding within batch
        batch_encoding = tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            padding=True,
            return_tensors='pt'
        )
        
        # Pad labels to match input_ids
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        
        # Convert seq_lengths to tensor
        seq_lengths = torch.tensor(seq_lengths)
        
        return {
            'input_ids': batch_encoding['input_ids'],
            'attention_mask': batch_encoding['attention_mask'],
            'labels': padded_labels,
            'seq_length': seq_lengths
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=(self.config.num_workers > 0),
            pin_memory=True,
            collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer)
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            persistent_workers=(self.config.num_workers > 0),
            pin_memory=True,
            collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer)
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            persistent_workers=(self.config.num_workers > 0),
            pin_memory=True,
            collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer)
        )