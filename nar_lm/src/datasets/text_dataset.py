from torch.utils.data import Dataset
import torch

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Debug print for tokenizer configuration
        print(f"[DEBUG] SimpleTextDataset: tokenizer={tokenizer.__class__.__name__}, pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")
        print(f"[DEBUG] SimpleTextDataset: max_length={max_length}, sample_text='{texts[0][:30]}...'")
        
        # Ensure pad_token is properly set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[DEBUG] SimpleTextDataset: Set pad_token to eos_token={self.tokenizer.pad_token}")
        
        # Tokenize all texts with dynamic padding
        self.encodings = tokenizer(
            texts, 
            truncation=True if max_length else False,
            padding='longest',  # Use dynamic padding instead of fixed max_length
            max_length=max_length if max_length else None, 
            return_tensors='pt'
        )
        
        # Store original sequence lengths for potential use in model
        self.seq_lengths = self.encodings['attention_mask'].sum(dim=1)
        
        # Debug sequence lengths
        print(f"[DEBUG] SimpleTextDataset: encoded {len(texts)} texts")
        print(f"[DEBUG] SimpleTextDataset: sequence lengths - min={self.seq_lengths.min().item()}, max={self.seq_lengths.max().item()}, mean={self.seq_lengths.float().mean().item():.2f}")
        print(f"[DEBUG] SimpleTextDataset: sequence length dtype={self.seq_lengths.dtype}")
        
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        # Ensure seq_length is type long (int64) to prevent dtype mismatches
        seq_length = self.seq_lengths[idx].long()
        
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx].clone(),
            'seq_length': seq_length  # Explicitly convert to long type
        }