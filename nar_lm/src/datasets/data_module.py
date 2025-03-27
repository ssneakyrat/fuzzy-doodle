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
            
            # Create dataset instances (now just storing raw texts)
            max_len = self.config.max_position_embeddings
            print(f"[DEBUG] NARDataModule.setup: Using max_length={max_len}")
            
            print(f"[DEBUG] NARDataModule.setup: Creating train dataset with {len(train_texts)} texts")
            self.train_dataset = SimpleTextDataset(
                train_texts, 
                self.tokenizer, 
                max_len
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
    def collate_fn(batch, tokenizer, max_length=None):
        """Improved collate function that handles tokenization in a single step"""
        try:
            # Extract raw texts from the batch
            texts = [item['text'] for item in batch]
            
            # Debug info
            if len(batch) > 0:
                print(f"[DEBUG] collate_fn: Processing batch of {len(texts)} texts")
            
            # Perform tokenization with padding in a single step
            # This addresses the tokenizer warning by using the __call__ method with padding
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
            
            # Debug info
            if len(batch) > 0:
                print(f"[DEBUG] collate_fn: Tokenized batch with seq_lengths dtype={seq_lengths.dtype}, shape={seq_lengths.shape}")
                # Print sample of sequence lengths
                if len(seq_lengths) > 0:
                    print(f"[DEBUG] collate_fn: Sample seq_lengths: {seq_lengths[:min(5, len(seq_lengths))]}")
            
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': labels,
                'seq_length': seq_lengths
            }
        except Exception as e:
            print(f"[ERROR] Error in collate_fn: {e}")
            import traceback
            traceback.print_exc()
            # Return empty tensors as a fallback to avoid crashing the training
            return {
                'input_ids': torch.zeros((1, 1), dtype=torch.long),
                'attention_mask': torch.zeros((1, 1), dtype=torch.long),
                'labels': torch.zeros((1, 1), dtype=torch.long),
                'seq_length': torch.ones((1,), dtype=torch.long)
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
            collate_fn=partial(self.collate_fn, 
                             tokenizer=self.tokenizer, 
                             max_length=self.config.max_position_embeddings)
        )
    
    def val_dataloader(self):
        print(f"[DEBUG] NARDataModule.val_dataloader: Creating with batch_size={self.config.batch_size}")
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
            collate_fn=partial(self.collate_fn, 
                             tokenizer=self.tokenizer, 
                             max_length=self.config.max_position_embeddings)
        )