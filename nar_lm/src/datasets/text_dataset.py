from torch.utils.data import Dataset
import torch

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=None):
        """
        Improved dataset that stores raw texts instead of tokenized data
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance (used for debugging/statistics only)
            max_length: Maximum sequence length (optional)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Debug print for tokenizer configuration
        print(f"[DEBUG] SimpleTextDataset: tokenizer={tokenizer.__class__.__name__}, pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")
        print(f"[DEBUG] SimpleTextDataset: max_length={max_length}, sample_text='{texts[0][:30]}...'")
        
        # Ensure pad_token is properly set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[DEBUG] SimpleTextDataset: Set pad_token to eos_token={self.tokenizer.pad_token}")
        
        # Only tokenize a small sample to get length statistics (not for actual use)
        # Use a subset to avoid slow initialization with large datasets
        sample_size = min(100, len(texts))
        sample_texts = texts[:sample_size]
        
        # Get sample length statistics without storing the encodings
        sample_encodings = tokenizer(
            sample_texts,
            truncation=True if max_length else False,
            padding=False,  # No padding at this stage
            max_length=max_length if max_length else None,
            return_tensors=None,  # Return python lists
            return_length=True    # Return sequence lengths
        )
        
        seq_lengths = sample_encodings['length']
        
        # Debug sequence lengths
        print(f"[DEBUG] SimpleTextDataset: stored {len(texts)} raw texts")
        if len(seq_lengths) > 0:
            print(f"[DEBUG] SimpleTextDataset: sequence lengths (sample) - min={min(seq_lengths)}, max={max(seq_lengths)}, mean={sum(seq_lengths)/len(seq_lengths):.2f}")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Simply return the raw text - tokenization happens in collate_fn
        return {
            'text': self.texts[idx]
        }