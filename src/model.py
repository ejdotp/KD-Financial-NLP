# src/model.py
# Dual-Head DistilBERT model for financial sentiment and urgency classification
# Part of: Black-Box KD for Financial Sentiment and Urgency Classification

import torch.nn as nn
from transformers import DistilBertModel


class DualHeadDistilBERT(nn.Module):
    """
    DistilBERT with two independent classification heads sharing
    a single encoder backbone.

    Architecture:
        Input → DistilBERT encoder → CLS token → Dropout
                                                 ├── Linear(768, 3) → Sentiment logits
                                                 └── Linear(768, 2) → Urgency logits

    Parameters: 66M (vs 8B teacher = 98.3% reduction)

    Args:
        num_sentiment: Number of sentiment classes (default 3)
        num_urgency  : Number of urgency classes (default 2)
        dropout      : Dropout rate applied to CLS representation (default 0.3)
    """
    def __init__(self, num_sentiment=3, num_urgency=2, dropout=0.3):
        super(DualHeadDistilBERT, self).__init__()

        self.distilbert     = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout        = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(768, num_sentiment)
        self.urgency_head   = nn.Linear(768, num_urgency)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through shared encoder and dual classification heads.

        Args:
            input_ids      : Tokenised input tensor [batch, seq_len]
            attention_mask : Attention mask tensor  [batch, seq_len]

        Returns:
            sentiment_logits: [batch, 3] raw logits for sentiment
            urgency_logits  : [batch, 2] raw logits for urgency
        """
        outputs    = self.distilbert(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        return self.sentiment_head(cls_output), self.urgency_head(cls_output)