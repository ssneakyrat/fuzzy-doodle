from torch.utils.data import Dataset

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
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
        
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx].clone(),
            'seq_length': self.seq_lengths[idx]  # Include original sequence length
        }