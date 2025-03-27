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
        print(f"[DEBUG] NARDataModule.prepare_data: Downloading tokenizer {self.config.tokenizer_name}")
        AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        
    def setup(self, stage=None):
        """Setup datasets for training, validation, and testing"""
        print(f"[DEBUG] NARDataModule.setup: Setting up for stage={stage}")
        
        if self.tokenizer is None:
            print(f"[DEBUG] NARDataModule.setup: Initializing tokenizer {self.config.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"[DEBUG] NARDataModule.setup: Set pad_token to eos_token={self.tokenizer.pad_token}")
        
        # Create datasets based on stage
        if stage == 'fit' or stage is None:
            print(f"[DEBUG] NARDataModule.setup: Setting up train and validation datasets")
            train_texts = self._load_texts(self.config.train_file)
            val_texts = self._load_texts(self.config.val_file)
            
            # If no validation file, split from training
            if val_texts is None and train_texts is not None:
                split_idx = int(len(train_texts) * (1 - self.config.val_split))
                val_texts = train_texts[split_idx:]
                train_texts = train_texts[:split_idx]
                print(f"[DEBUG] NARDataModule.setup: Split training data - train={len(train_texts)}, val={len(val_texts)}")
            
            # If no data files, create synthetic data
            if train_texts is None:
                print(f"[DEBUG] NARDataModule.setup: Creating synthetic data - train={self.config.num_train_samples}, val={self.config.num_val_samples}")
                train_texts = self._create_synthetic_data(self.config.num_train_samples)
                val_texts = self._create_synthetic_data(self.config.num_val_samples)
            
            # Create datasets with dynamic sequence handling
            # Pass max_length as None for fully dynamic handling, or keep config's max_position_embeddings for cap
            max_len = self.config.max_position_embeddings
            print(f"[DEBUG] NARDataModule.setup: Using max_length={max_len}")
            
            print(f"[DEBUG] NARDataModule.setup: Creating train dataset with {len(train_texts)} texts")
            self.train_dataset = SimpleTextDataset(
                train_texts, 
                self.tokenizer, 
                max_len  # Can set to None for full dynamic handling
            )
            
            print(f"[DEBUG] NARDataModule.setup: Creating validation dataset with {len(val_texts)} texts")
            self.val_dataset = SimpleTextDataset(
                val_texts, 
                self.tokenizer, 
                max_len
            )
            
        if stage == 'test' or stage is None:
            print(f"[DEBUG] NARDataModule.setup: Setting up test dataset")
            test_texts = self._load_texts(self.config.test_file)
            if test_texts is None:
                print(f"[DEBUG] NARDataModule.setup: Creating synthetic test data - {self.config.num_test_samples} samples")
                test_texts = self._create_synthetic_data(self.config.num_test_samples)
                
            print(f"[DEBUG] NARDataModule.setup: Creating test dataset with {len(test_texts)} texts")
            self.test_dataset = SimpleTextDataset(
                test_texts, 
                self.tokenizer, 
                self.config.max_position_embeddings
            )
    
    def _load_texts(self, file_path):
        """Load texts from file"""
        if file_path and os.path.exists(file_path):
            print(f"[DEBUG] NARDataModule._load_texts: Loading texts from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Filter out empty lines
            texts = [line.strip() for line in lines if len(line.strip()) > 10]
            print(f"[DEBUG] NARDataModule._load_texts: Loaded {len(texts)} texts")
            return texts
        return None
    
    def _create_synthetic_data(self, num_samples):
        """Create synthetic data for testing with varied lengths"""
        print(f"[DEBUG] NARDataModule._create_synthetic_data: Creating {num_samples} synthetic samples")
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
        
        # Debug batch consistency
        if len(batch) > 0:
            print(f"[DEBUG] collate_fn: Batch size={len(batch)}, seq_length dtype={type(seq_lengths[0])}")
        
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
        
        # Convert seq_lengths to tensor with consistent dtype (long)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        
        return {
            'input_ids': batch_encoding['input_ids'],
            'attention_mask': batch_encoding['attention_mask'],
            'labels': padded_labels,
            'seq_length': seq_lengths
        }
    
    def train_dataloader(self):
        print(f"[DEBUG] NARDataModule.train_dataloader: Creating with batch_size={self.config.batch_size}")
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
        print(f"[DEBUG] NARDataModule.val_dataloader: Creating with batch_size={self.config.batch_size}")
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            persistent_workers=(self.config.num_workers > 0),
            pin_memory=True,
            collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer)
        )
    
    def test_dataloader(self):
        print(f"[DEBUG] NARDataModule.test_dataloader: Creating with batch_size={self.config.batch_size}")
        if self.test_dataset is None:
            print(f"[WARNING] NARDataModule.test_dataloader: test_dataset is None, setting up explicitly")
            self.setup(stage='test')
            
        if self.test_dataset is None:
            raise ValueError("test_dataset is still None after explicit setup!")
            
        return DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            persistent_workers=(self.config.num_workers > 0),
            pin_memory=True,
            collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer)
        )