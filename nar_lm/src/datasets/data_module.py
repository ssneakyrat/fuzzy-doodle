import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
from transformers import AutoTokenizer

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
            
            self.train_dataset = SimpleTextDataset(
                train_texts, 
                self.tokenizer, 
                self.config.max_position_embeddings
            )
            
            self.val_dataset = SimpleTextDataset(
                val_texts, 
                self.tokenizer, 
                self.config.max_position_embeddings
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
        """Create synthetic data for testing"""
        synthetic_texts = [
            "This is a synthetic text for training the non-autoregressive model.",
            "We are using PyTorch Lightning to train our model efficiently.",
            "This model has encoder and decoder components with a latent space.",
            "Non-autoregressive generation produces all tokens in parallel.",
            "TensorBoard logging helps us track the training progress and text generation."
        ]
        
        # Repeat to reach desired count
        texts = synthetic_texts * (num_samples // len(synthetic_texts) + 1)
        return texts[:num_samples]
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )