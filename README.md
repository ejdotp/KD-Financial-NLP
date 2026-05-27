# Optimising Inference Latency in Enterprise NLP via Task-Specific Knowledge Distillation

**Final Year Capstone Project — BTech Computer Science**  
Group 15 | Section 2241044 | ITER, Siksha 'O' Anusandhan University

---

## Overview

This project implements a two-stage black-box Knowledge Distillation (KD) pipeline for financial sentiment classification. A large language model (Llama-3.1-8B, accessed via the Groq API) acts as the teacher and generates pseudo-labels for the Financial PhraseBank corpus. A lightweight DualHeadDistilBERT student (66M parameters) is then fine-tuned on those labels and deployed for CPU-only inference on analyst laptops.

The core question: *can a general-purpose LLM, accessed only through a public API with no access to weights or logits, generate pseudo-labels of sufficient quality to train a production-ready financial text classifier?*

---

## Key Results

| Metric | Value |
|---|---|
| Student Macro F1 (KD pipeline) | 0.754 ± 0.006 |
| Student Macro F1 (gold-label baseline) | 0.809 ± 0.010 |
| Accuracy gap (zero annotation cost) | −0.055 |
| Inference latency — Intel 12th/13th Gen | 25.70 ms/sample ✓ |
| Inference latency — AMD Ryzen 5 2500U | 83.86 ms/sample ✗ |
| 50 ms SLA | Met on current-gen hardware |
| Model size on disk | 253 MB |
| Peak inference RAM | 825 MB |
| Parameter reduction vs. teacher | ~99.2% (8,030M → 66M) |
| Teacher label accuracy vs. gold | 80.44% (κ = 0.636) |

---

## Repository Structure

```
.
├── data/
│   └── financial_phrasebank/          # Raw corpus (4,846 sentences)
│       └── Sentences_AllAgree.txt
│
├── stage1_colab/                      # Run on Google Colab (T4 GPU)
│   ├── 01_pseudolabel_generation.ipynb   # Groq API prompting + checkpointing
│   ├── 02_student_finetuning.ipynb       # DualHeadDistilBERT training
│   └── checkpoints/                      # Saved every 100 samples to Drive
│
├── stage2_inference/                  # Run locally (CPU only)
│   ├── inference_benchmark.py            # 100-pass latency + RAM profiling
│   └── predict.py                        # Single-sample inference script
│
├── model/
│   ├── dual_head_distilbert.py           # Model architecture definition
│   └── saved/                            # Exported student model (253 MB)
│
├── results/
│   ├── pseudolabels_3876.csv             # Generated pseudo-labels
│   ├── training_log_seed42.csv           # Per-epoch loss + F1 (seed 42)
│   ├── training_log_seed7.csv
│   ├── training_log_seed123.csv
│   └── error_analysis_30samples.csv      # 30 misclassified test samples
│
├── figures/                           # All manuscript figures (EPS + PDF)
│   ├── fig3_training.eps / .pdf
│   ├── fig4_f1.eps / .pdf
│   ├── fig5_latency.eps / .pdf
│   └── fig6_errors.eps / .pdf
│
├── manuscript/
│   ├── frp_manuscript.tex             # LNCS LaTeX source
│   ├── frp_manuscript.pdf             # Compiled paper
│   └── llncs.cls                      # Replace with official Springer cls
│
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier sufficient)
- Google Colab account with Google Drive (for Stage 1)
- CPU-only machine for Stage 2 inference benchmarking

### Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt** covers:
```
torch>=2.0.0
transformers>=4.40.0
datasets
scikit-learn
pandas
numpy
tqdm
groq
psutil
matplotlib
```

---

## Running the Pipeline

### Stage 1A — Pseudo-label generation (Colab)

Open `stage1_colab/01_pseudolabel_generation.ipynb`.  
Set your Groq API key:

```python
GROQ_API_KEY = "your_key_here"
```

The notebook prompts Llama-3.1-8B at temperature 0, requests a JSON response with `sentiment`, `urgency`, and `confidence` for each sentence, checkpoints to Google Drive every 100 samples, and retries failed calls up to 3 times with exponential back-off.

Output: `results/pseudolabels_3876.csv`

### Stage 1B — Student fine-tuning (Colab, T4 GPU)

Open `stage1_colab/02_student_finetuning.ipynb`.

Trains DualHeadDistilBERT for 5 epochs with:
- Combined loss: `L = 0.6 × CE_sentiment + 0.4 × CE_urgency`
- Class-weighted cross-entropy (sklearn balanced mode)
- AdamW, lr = 2e-5, weight decay = 0.01
- Linear warmup (10%) + linear decay
- Gradient clipping at max norm 1.0
- Batch size 16, seeds 42 / 7 / 123

Best checkpoint selected by validation Macro F1. Exported to `model/saved/`.

### Stage 2 — CPU inference benchmark (local)

```bash
python stage2_inference/inference_benchmark.py \
    --model_path model/saved/ \
    --n_passes 100
```

Reports mean latency, std, throughput, and peak RAM.

---

## Model Architecture

```
Input text
    ↓
DistilBERT Encoder  (6 Transformer layers, 768 hidden dim, 12 heads)
    ↓
[CLS] token  (768-dim)
    ↓
Dropout (p = 0.3)
    ↓         ↓
Sentiment   Urgency
Linear      Linear
(768→3)     (768→2)
    ↓         ↓
Neg/Neu/Pos  Urgent/Non-urgent
```

Total parameters: 66M. Teacher (Llama-3.1-8B): 8,030M. Parameter reduction: ~99.2%.

---

## Notable Findings

**Teacher overconfidence.** Llama-3.1-8B assigns mean confidence 0.95 to every sample regardless of actual label quality. The confidence threshold filter (designed to remove low-quality pseudo-labels at < 0.70) was completely inactive as a result. Any pipeline relying on LLM self-reported confidence for filtering should validate calibration on a pilot sample first.

**Error partition.** Manual analysis of 30 misclassified test samples found only two error types: T1 (Ambiguous Language, n=20) affecting negative and neutral classes, and T2 (Subtle Positive, n=10) affecting the positive class exclusively. Zero errors of types T3–T5 (context-dependency, negation, domain jargon). The student's linguistic competence is not the bottleneck — the positive-class gap traces directly to the teacher's 62% positive-class recall.

**Hardware dependency.** The 50 ms SLA is met on Intel 12th/13th Gen hardware (25.70 ms) but missed on a 2018 AMD Ryzen 5 2500U (83.86 ms). For this model size and architecture, latency compliance is a procurement question, not a model design question.

---

## Manuscript

The paper is formatted to Springer LNCS standards.

```
manuscript/frp_manuscript.tex   ← source
manuscript/frp_manuscript.pdf   ← compiled output
```

**Before submission:** replace the `llncs.cls` stub in `manuscript/` with the official class file from [springer.com](https://www.springer.com/gp/authors-editors/conference-proceedings/conference-proceedings-guidelines).

---

## Citation

If you build on this work, please cite:

```bibtex
@inproceedings{patro2025optimising,
  title     = {Optimising Inference Latency in Enterprise {NLP} via
               Task-Specific Knowledge Distillation},
  author    = {Patro, E. Jagadeeswar and Mohanty, Subham and
               Jha, Anisha and Mohanty, Udipta and Sahoo, Mohinikanta},
  booktitle = {Proceedings of [Conference Name]},
  year      = {2025},
  publisher = {Springer},
  series    = {Lecture Notes in Computer Science}
}
```

---

## Acknowledgements

Groq free-tier API access was used for all pseudo-label generation. Google Colab free-tier T4 GPU was used for student fine-tuning. No paid compute was used at any stage of this project.
