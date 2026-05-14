# Optimizing Inference Latency in Enterprise NLP via Task-Specific Knowledge Distillation

**Black-Box Knowledge Distillation for Financial Sentiment and Urgency Classification**

Group 15 | Section 2241044 | ITER, Siksha 'O' Anusandhan University

---

## Overview

This project implements a Black-Box Knowledge Distillation (KD) pipeline that transfers
financial sentiment and urgency classification capability from a large teacher model
(Llama-3.1-8B, 8B parameters) to a lightweight student model (DistilBERT-base-uncased,
66M parameters) deployable on standard CPU hardware without access to teacher model
weights or internal logits.

---

## Key Results (3-Seed Statistical Summary)

| Metric | KD Pipeline | Vanilla Baseline | Gap |
|---|---|---|---|
| Macro F1 | 0.7539 ± 0.0055 | 0.8093 ± 0.0096 | -0.055 |
| Accuracy | 0.7883 ± 0.0060 | 0.8241 ± 0.0078 | -0.036 |
| Negative F1 | 0.7556 ± 0.0130 | 0.8128 ± 0.0240 | -0.057 |
| Neutral F1 | 0.8444 ± 0.0074 | 0.8591 ± 0.0081 | -0.015 |
| Positive F1 | 0.6619 ± 0.0072 | 0.7559 ± 0.0104 | -0.094 |
| Annotation Cost | Zero (pseudo-labels) | High (gold labels) | — |

Seeds: 42, 7, 123 | Epochs: 5 | lr: 2e-5 | batch_size: 16

### Hardware Benchmark (AMD Ryzen 5 2500U, Windows 11, 14.9 GB RAM)

| Metric | Value | Target | Status |
|---|---|---|---|
| Mean latency | 83.86 ms | <= 50ms | Hardware-dependent |
| Latency vs teacher (~800ms) | 9.5x faster | — | Strong |
| Peak RAM at inference | 825.1 MB | <= 1,500 MB | Pass |
| Model size on disk | 253.20 MB | <= 300 MB | Pass |
| Parameter reduction | 98.3% (8B to 66M) | >= 98% | Pass |
| Failed pseudo-labels | 0 / 3,876 | 0 | Pass |

### Teacher Quality (Llama-3.1-8B via Groq API)

| Metric | Value | Target |
|---|---|---|
| Cohen's Kappa vs gold labels | 0.636 | >= 0.75 |
| Label accuracy | 80.4% | >= 88% |
| Failed labels | 0 / 3,876 | 0 |
| Mean confidence score | 0.95 (overconfident) | Calibrated |

---

## Key Findings

**Finding 1 — Teacher quality ceiling is the binding constraint:**
The 8B teacher achieved k=0.636, primarily due to positive/neutral confusion (teacher
positive recall: 0.62). The KD pipeline Macro F1 of 0.754 is upper-bounded by this
teacher quality, consistent with theoretical predictions in Gu et al. (2024).

**Finding 2 — 94% of the KD-vanilla gap comes from positive class:**
Neutral F1 gap = 0.015 (near-equivalent). Positive F1 gap = 0.094 (dominant).
Black-box KD is effective where teacher labels are reliable (neutral), and degrades
where they are noisy (positive/neutral confusion in subtle financial language).

**Finding 3 — Results are statistically stable:**
KD pipeline std = +/-0.0055 across 3 seeds. The gap of 0.055 is 5x larger than the
model's own variance — a genuine reproducible difference, not noise.

**Finding 4 — Hardware targets met except latency on legacy CPU:**
83.86ms on a 2018 AMD mobile processor. Published DistilBERT benchmarks (Sanh et al.,
2019) report 45ms on modern Intel hardware. RAM (825MB) and model size (253MB) pass.

**Finding 5 — LLM confidence scores are not calibrated:**
Llama-3.1-8b-instant assigned 0.95 confidence to virtually all samples regardless of
actual uncertainty, rendering confidence-based filtering (LLKD, Li et al. 2024)
ineffective. Documented as a limitation for future work.

---

## Pipeline Architecture

```
Financial PhraseBank (4,846 samples)
        |
        v
Text Preprocessing (HuggingFace tokenizer, max_length=128)
        |
        v  [STAGE 1: Google Colab T4 GPU]
Llama-3.1-8B Teacher (Groq API, black-box mode)
        |
        v
Pseudo-Labels: Sentiment (3-class) + Urgency (binary)
        |
        v
Confidence Filtering (threshold=0.70)
        |
        v
Dual-Head DistilBERT Fine-tuning
Loss: 0.6 * CE(sentiment) + 0.4 * CE(urgency) [class-weighted]
        |
        v  [STAGE 2: Local CPU Analyst Laptop]
CPU Inference Engine (batch_size=1, no GPU)
        |
        v
Output: Sentiment + Urgency labels at 83.86ms mean latency
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
|   └── .gitkeep
|
├── notebooks/
|   ├── 01_dataset_setup.ipynb
|   ├── 02_pseudo_label_generation.ipynb
|   ├── 03_distilbert_finetuning.ipynb
|   ├── 04_evaluation_inference.ipynb
|   ├── 05_vanilla_baseline.ipynb
|   └── 06_statistical_significance.ipynb
|
├── src/
|   ├── prompt.py
|   ├── dataset.py
|   ├── model.py
|   └── inference.py
|
└── results/
    ├── classification_report.txt
    ├── training_history.csv
    ├── baseline_comparison.csv
    ├── statistical_significance.csv
    ├── statistical_summary.csv
    ├── latency_local.csv
    └── results_summary.csv
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset

Financial PhraseBank (Malo et al., 2014) — 4,846 financial news sentences
annotated by 16 domain experts (MBA students and finance professionals).

- Source: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
- Classes: Positive (28.1%), Neutral (59.4%), Negative (12.5%)
- Split: 80% train (3,876) / 10% val (485) / 10% test (485) — stratified

Data files are not included in this repository.
Download from Kaggle and place in the data/ directory before running notebooks.

---

## Model Weights

Fine-tuned model weights (253MB) are not included due to file size limits.

Download from Google Drive: [add your Drive link here]

After downloading, unzip and place at: ./distilbert_finetuned/

---

## Reproducing Results

### Prerequisites
- Google Colab account (free tier sufficient)
- Groq API key (free at https://console.groq.com)
- Python 3.10+ for local inference

### Step 1 — Dataset Setup
Run notebooks/01_dataset_setup.ipynb in Google Colab.

### Step 2 — Pseudo-Label Generation
Run notebooks/02_pseudo_label_generation.ipynb in Google Colab.
Requires Groq API key. Expected runtime: ~25 minutes (free tier).
Note: model string is llama-3.1-8b-instant (llama3-8b-8192 is decommissioned as of Aug 2025).

### Step 3 — Fine-Tuning
Run notebooks/03_distilbert_finetuning.ipynb in Google Colab (T4 GPU).
Expected runtime: ~10 minutes on T4.

### Step 4 — Evaluation
Run notebooks/04_evaluation_inference.ipynb for test set evaluation.
Run local_benchmark.py on your laptop for CPU latency measurement:

```bash
python local_benchmark.py
```

### Step 5 — Baseline and Statistics
Run notebooks/05_vanilla_baseline.ipynb for vanilla baseline.
Run notebooks/06_statistical_significance.ipynb for 3-seed results.
Expected runtime: ~60 minutes for all 6 training runs.

---

## Hardware Requirements

| Stage | Hardware | Notes |
|---|---|---|
| Pseudo-label generation | Colab CPU + Groq API | Free tier sufficient |
| Fine-tuning | Colab T4 GPU (16GB) | Free tier sufficient |
| Inference | CPU-only, min 2GB RAM | No GPU required |

---

## Model Architecture

```
Input -> DistilBERT (6 layers, 768 hidden, 66M params) -> CLS token -> Dropout(0.3)
                                                                         |
                                              +--------------------------+
                                              |                          |
                                      Linear(768, 3)             Linear(768, 2)
                                              |                          |
                                     Sentiment logits            Urgency logits
                                   (neg / neu / pos)         (non-urgent / urgent)
```

Training loss:
```
L_total = 0.6 * CrossEntropy(sentiment, weights=[2.67, 0.56, 1.19])
        + 0.4 * CrossEntropy(urgency,   weights=[0.52, 15.02])
```

---

## Known Limitations

1. Teacher quality ceiling (k=0.636): positive/neutral confusion limits student F1
2. LLM overconfidence: confidence scores not calibrated, filtering was ineffective
3. Latency on legacy hardware: 83.86ms exceeds 50ms target on 2018 AMD mobile CPU
4. Urgency imbalance (29:1): limits urgency classification reliability

---

## Future Work

- Error analysis: categorise misclassified samples by class
- Ablation study: remove class weights to quantify their contribution
- Second laptop benchmark: modern Intel CPU
- Domain-fine-tuned teacher: FinBERT as teacher to improve positive-class recall
- INT8 quantisation: further latency and size reduction
- Domain extension: legal NLP, ESG sentiment, clinical triage

---

## References

1. Malo et al. (2014) — Financial PhraseBank
2. Sanh et al. (2019) — DistilBERT
3. Devlin et al. (2019) — BERT
4. Hinton et al. (2015) — Knowledge Distillation
5. Gu et al. (2024) — Black-Box KD methodology
6. Li et al. (2024) — LLKD confidence filtering
7. Touvron et al. (2023) — Llama 2/3

Full IEEE-formatted reference list in the project report.

---

## Authors

Group 15, Section 2241044
ITER, Siksha 'O' Anusandhan University
