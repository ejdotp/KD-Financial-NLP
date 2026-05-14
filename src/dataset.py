# src/dataset.py
# PyTorch Dataset class for Financial PhraseBank
# Part of: Black-Box KD for Financial Sentiment and Urgency Classification

import torch
from torch.utils.data import Dataset


class FinancialDataset(Dataset):
    """
    PyTorch Dataset for dual-task financial text classification.
    Handles tokenisation internally for clean DataLoader integration.

    Args:
        texts           : List or array of financial sentences
        sentiment_labels: Integer sentiment labels (0=negative, 1=neutral, 2=positive)
        urgency_labels  : Integer urgency labels (0=non-urgent, 1=urgent). Optional.
        tokenizer       : HuggingFace tokenizer instance
        max_length      : Maximum token length (default 128)
    """
    def __init__(self, texts, sentiment_labels, urgency_labels=None,
                 tokenizer=None, max_length=128):

        self.encodings = tokenizer(
            list(texts),
            truncation      = True,
            padding         = True,
            max_length      = max_length,
            return_tensors  = 'pt'
        )
        self.sentiment_labels = torch.tensor(sentiment_labels, dtype=torch.long)
        self.urgency_labels   = torch.tensor(
            urgency_labels, dtype=torch.long
        ) if urgency_labels is not None else None

    def __len__(self):
        return len(self.sentiment_labels)

    def __getitem__(self, idx):
        item = {
            'input_ids'       : self.encodings['input_ids'][idx],
            'attention_mask'  : self.encodings['attention_mask'][idx],
            'sentiment_label' : self.sentiment_labels[idx],
        }
        if self.urgency_labels is not None:
            item['urgency_label'] = self.urgency_labels[idx]
        return item