"""
Dataset class for plausibility prediction
"""

import json
import torch
from torch.utils.data import Dataset


class SimplePlausibilityDataset(Dataset):
    """Simple dataset for plausibility prediction"""
    
    def __init__(self, data_file, tokenizer, max_length=512, is_test=False):
        # Load data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = list(data.values())
        
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.truncation_side = "left"

        self.is_test = is_test
        if self.is_test:
            self.std_devs = [0]*len(data)
        else:
            self.std_devs = [sample['std_score'] for sample in data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get text
        precontext = ' '.join(sample['precontext_sentences'])
        sentence = sample['ambiguous_sentence']
        word_sense = sample['word_sense']
        ending = sample.get('ending', '')
        
        # Create input text
        text = f"{precontext} {sentence} {ending} [SEP] {word_sense}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if self.is_test:
            return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(0, dtype=torch.float),
            'std_devs': torch.tensor(0, dtype=torch.float)  
            }

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['avg_score'], dtype=torch.float),
            'std_devs': torch.tensor(sample['std_score'], dtype=torch.float)  
        }