# src/inference.py
# CPU inference pipeline for local deployment
# Part of: Black-Box KD for Financial Sentiment and Urgency Classification

import torch
import time
import numpy as np
from transformers import DistilBertTokenizerFast
from src.model import DualHeadDistilBERT


# Label maps
SENTIMENT_LABELS = {0: "negative", 1: "neutral",    2: "positive"}
URGENCY_LABELS   = {0: "non-urgent", 1: "urgent"}


def load_model(model_dir: str):
    """
    Load fine-tuned DualHeadDistilBERT and tokenizer from directory.

    Args:
        model_dir: Path to directory containing model_weights.pt and tokenizer files

    Returns:
        model    : Loaded model in eval mode on CPU
        tokenizer: Loaded tokenizer
    """
    tokenizer  = DistilBertTokenizerFast.from_pretrained(model_dir)
    model      = DualHeadDistilBERT()
    state_dict = torch.load(
        f"{model_dir}/model_weights.pt",
        map_location = 'cpu'
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def predict(sentence: str, model, tokenizer) -> dict:
    """
    Run single-sample inference on one financial sentence.

    Args:
        sentence : Financial news sentence
        model    : Loaded DualHeadDistilBERT
        tokenizer: Loaded tokenizer

    Returns:
        dict with keys: sentiment, urgency, latency_ms
    """
    inputs = tokenizer(
        sentence,
        return_tensors = 'pt',
        truncation     = True,
        padding        = True,
        max_length     = 128
    )

    start = time.perf_counter()
    with torch.no_grad():
        s_logits, u_logits = model(
            inputs['input_ids'],
            inputs['attention_mask']
        )
    end = time.perf_counter()

    return {
        "sentiment"  : SENTIMENT_LABELS[torch.argmax(s_logits).item()],
        "urgency"    : URGENCY_LABELS[torch.argmax(u_logits).item()],
        "latency_ms" : (end - start) * 1000
    }


def benchmark(sentences: list, model, tokenizer) -> dict:
    """
    Run latency benchmark over a list of sentences.

    Args:
        sentences: List of financial sentences
        model    : Loaded model
        tokenizer: Loaded tokenizer

    Returns:
        dict with latency statistics
    """
    latencies = []
    for sentence in sentences:
        result = predict(sentence, model, tokenizer)
        latencies.append(result["latency_ms"])

    latencies = np.array(latencies)
    return {
        "mean_ms"   : latencies.mean(),
        "median_ms" : np.median(latencies),
        "min_ms"    : latencies.min(),
        "max_ms"    : latencies.max(),
        "std_ms"    : latencies.std(),
        "n_samples" : len(latencies)
    }