# Optimizing Inference Latency in Enterprise NLP via Task-Specific Knowledge Distillation

**Black-Box Knowledge Distillation for Financial Sentiment and Urgency Classification**

Group 15 | Section 2241044 | ITER, Siksha 'O' Anusandhan University

---

## Overview

This project implements a Black-Box Knowledge Distillation (KD) pipeline
that transfers financial sentiment and urgency classification capability
from a large teacher model (Llama-3.1-8B, 8B parameters) to a lightweight
student model (DistilBERT-base-uncased, 66M parameters) deployable on
standard CPU hardware without access to teacher model weights or internal logits.

**Key Results:**

| Metric | Value |
|---|---|
| Student Test Macro F1 | 0.7452 |
| Student Val Macro F1 | 0.8028 |
| Teacher Cohen Kappa | 0.6362 |
| CPU Inference Latency | 83.86ms (AMD Ryzen 5 2500U) |
| Latency Reduction vs Teacher | 9.5x faster |
| Parameter Reduction | 98.3% (8B to 66M) |
| Peak RAM at Inference | 825.1 MB |
| Model Size on Disk | 253.20 MB |
| Failed Pseudo-Labels | 0 / 3876 |

---

## Pipeline Architecture

```
Financial PhraseBank (4,846 samples)
        |
        v
Text Preprocessing (HuggingFace tokenizer, max_length=128)
        |
        v
Llama-3.1-8B Teacher (Groq API, black-box mode)
        |
        v
Pseudo-Labels: Sentiment (3-class) + Urgency (binary)
        |
        v
Confidence Filtering (threshold = 0.70)
        |
        v
DistilBERT Fine-tuning (Google Colab T4, 5 epochs, dual-head)
        |
        v
CPU Inference Engine (local laptop deployment)
        |
        v
Output: Sentiment + Urgency labels
```

---

## Repository Structure

```
KD-Financial-NLP/
|
├── README.md
├── requirements.txt
|
├── data/
|   └── .gitkeep                         # Data stored on Google Drive, not in repo
|
├── notebooks/
|   ├── 01_dataset_setup.ipynb           # Dataset loading, EDA, train/val/test split
|   ├── 02_pseudo_label_generation.ipynb # Llama-3.1-8B prompting, label generation
|   ├── 03_distilbert_finetuning.ipynb   # Dual-head DistilBERT training
|   └── 04_evaluation_inference.ipynb    # Test evaluation, CPU latency benchmark
|
├── src/
|   ├── prompt.py                        # Pseudo-label generation functions
|   ├── dataset.py                       # PyTorch FinancialDataset class
|   ├── model.py                         # DualHeadDistilBERT architecture
|   └── inference.py                     # CPU inference pipeline
|
└── results/
    ├── training_history.csv             # Epoch-wise train/val loss and F1
    ├── classification_report.txt        # Final test set classification report
    ├── latency_local.csv                # Per-sample latency measurements
    └── results_summary.csv             # Full project results table
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset

Financial PhraseBank (Malo et al., 2014) — 4,846 financial news sentences
annotated by 16 domain experts (MBA students and finance professionals)
for 3-class sentiment classification.

- Source: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
- Classes: Positive (28.1%), Neutral (59.4%), Negative (12.5%)
- Split: 80% train / 10% val / 10% test (stratified)

> Data files are not included in this repository.
> Download from Kaggle and place in the data/ directory.

---

## Model Weights

Model weights (253MB) are not included in this repository due to file size limits.

Download from Google Drive: [add your Drive link here]

Place the downloaded folder at: `./distilbert_finetuned/`

---

## Reproducing Results

### Step 1 — Dataset Setup
Run `notebooks/01_dataset_setup.ipynb` in Google Colab.
Saves train/val/test splits to Google Drive.

### Step 2 — Pseudo-Label Generation
Run `notebooks/02_pseudo_label_generation.ipynb` in Google Colab.
Requires a free Groq API key from https://console.groq.com
Saves pseudo_labels_final.csv to Google Drive.
Expected runtime: ~25 minutes on free Groq tier.

### Step 3 — Fine-Tuning
Run `notebooks/03_distilbert_finetuning.ipynb` in Google Colab (T4 GPU).
Saves model_weights.pt and tokenizer to Google Drive.
Expected runtime: ~10 minutes on T4 GPU.

### Step 4 — Evaluation and Inference
Run `notebooks/04_evaluation_inference.ipynb` in Google Colab for test evaluation.
Run `local_benchmark.py` locally for CPU latency benchmark.

---

## Hardware Requirements

| Stage | Hardware | Notes |
|---|---|---|
| Pseudo-label generation | Google Colab (CPU) + Groq API | Free tier sufficient |
| Fine-tuning | Google Colab T4 GPU | Free tier sufficient |
| Inference | CPU-only laptop, min 2GB RAM available | No GPU required |

---

## Key Design Decisions

**Black-Box KD:** Llama-3.1-8B accessed via API only — no model weights, no logits.
Labels generated through structured prompting with JSON output enforcement.

**Dual-Head Architecture:** Single DistilBERT encoder with two independent
classification heads (sentiment + urgency) sharing representations.
Combined loss: L_total = 0.6 * L_sentiment + 0.4 * L_urgency

**Class-Weighted Loss:** Applied to both tasks to handle imbalance.
Sentiment: neutral 4.8x larger than negative.
Urgency: non-urgent 29x larger than urgent.

**Confidence Filtering:** Samples with teacher confidence below 0.70 removed.
Note: llama-3.1-8b-instant exhibited systematic overconfidence (mean=0.95),
rendering this filter ineffective in practice — documented as a finding.

---

## Results

### Sentiment Classification (Test Set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.70 | 0.80 | 0.75 | 61 |
| Neutral | 0.80 | 0.87 | 0.83 | 288 |
| Positive | 0.76 | 0.57 | 0.66 | 136 |
| Macro avg | 0.75 | 0.75 | 0.75 | 485 |

### Hardware Benchmark (AMD Ryzen 5 2500U, Windows 11, 14.9GB RAM)

| Metric | Value | Target | Status |
|---|---|---|---|
| Mean latency | 83.86ms | <= 50ms | Hardware-dependent |
| Latency vs teacher | 9.5x faster | -- | Strong |
| Peak RAM | 825.1 MB | <= 1500MB | Pass |
| Model size | 253.20 MB | <= 300MB | Pass |
| Parameter reduction | 98.3% | >= 98% | Pass |

---

## Known Limitations

1. **Teacher quality ceiling:** llama-3.1-8b-instant achieved Cohen Kappa of 0.636
   against gold labels, primarily due to positive/neutral confusion in subtle
   financial language. Student performance is upper-bounded by teacher quality.

2. **Latency on legacy hardware:** 83.86ms exceeds the 50ms SLA target on a
   2018-era AMD mobile processor. Modern Intel 12th-gen hardware is expected
   to meet the target based on published DistilBERT benchmarks (Sanh et al., 2019).

3. **Urgency label imbalance:** 96.7% non-urgent in training data limits
   urgency classification reliability despite class-weighted loss.

4. **LLM overconfidence:** Confidence scores from llama-3.1-8b-instant are
   not well-calibrated, making confidence-based filtering ineffective.

---

## Future Work

- Vanilla DistilBERT baseline (gold labels only) for KD contribution verification
- Multi-seed training for statistical significance reporting
- Ablation study: impact of class weighting and dual-task training
- Domain extension: legal clause classification, ESG sentiment, clinical triage
- INT8 quantisation for further latency reduction on legacy hardware

---

## References

1. Malo et al. (2014) — Financial PhraseBank dataset
2. Sanh et al. (2019) — DistilBERT
3. Devlin et al. (2019) — BERT
4. Hinton et al. (2015) — Knowledge Distillation
5. Gu et al. (2024) — Black-Box KD methodology
6. Li et al. (2024) — LLKD confidence filtering
7. Touvron et al. (2023) — Llama 2/3

Full reference list available in the project report.

---

## Authors

Group 15, Section 2241044
ITER, Siksha 'O' Anusandhan University
