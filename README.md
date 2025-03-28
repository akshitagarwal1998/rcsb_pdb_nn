# Protein Similarity Classification using BioZernike Descriptors

This repository provides a PyTorch-based framework for classifying protein structural similarity using BioZernike descriptors. Developed as part of the RCSB PDB Data Science internship task, the pipeline supports efficient dataset handling, model training, and evaluation workflows at scale.

---

## 📁 Directory Overview

```
rcsb_pdb_nn/
├── model.py               # Defines the neural network model (MLP)
├── train.py               # Training loop and evaluation logic
├── dataset.py             # Dataset classes (streaming vs. precomputed cache)
├── cache_utils.py         # Utilities for caching feature-label pairs
├── tensorboard_utils.py   # Setup for TensorBoard logging
├── util.py                # Descriptor classes and distance metrics
├── weight_strategy.py     # (Optional) Weighted sampling strategies
│
├── data/
│   ├── cath_moments.tsv   # CATH dataset (geometric + Zernike features)
│   └── ecod_moments.tsv   # ECOD dataset for testing
│
├── cache/                 # Contains partitioned and merged caches
├── logs/                  # Log files per training run
├── modelData/             # Stores best and per-epoch model checkpoints
└── Main.ipynb             # Notebook for exploration and testing
```

---

## ⚙️ Setup

```bash
# Clone repository
git clone https://github.com/akshitagarwal1998/rcsb_pdb_nn.git
cd rcsb_pdb_nn

# Setup environment
conda create -n tf_keras python=3.12
conda activate tf_keras
pip install -r requirements.txt
```

---

## 🚀 Modes of Operation

### 1. Precomputed Mode (`streaming=False`)

Load precomputed pairwise feature and label tensors from disk for efficient training.

```python
from cache_utils import load_cached_parts
from train import train_model

features, labels = load_cached_parts("./cache/cath_2685")
model = train_model(features=features, labels=labels, streaming=False, input_dim=3924)
```

To generate a precomputed cache:

```python
from cache_utils import cache_pairwise_data
cache_pairwise_data(df, cache_dir="./cache/cath_2685", buffer_limit_mb=100)
```

### 2. Streaming Mode (`streaming=True`)

Stream protein pairs and compute distances on-the-fly, avoiding large memory usage.

```python
from train import train_model

df = pd.read_csv("data/cath_moments.tsv", sep='\t', header=None).dropna(axis=1)
model = train_model(protein_df=df, streaming=True)
```

---

## Evaluation

To evaluate the model on the ECOD dataset:

```python
from train import test_model_on_ecod

ecod_df = pd.read_csv("data/ecod_moments.tsv", sep='\t', header=None).dropna(axis=1)
test_model_on_ecod(model, ecod_df)
```

---

## TensorBoard Logging

All metrics are logged via TensorBoard. To launch the dashboard:

```bash
tensorboard --logdir=tensorboard_logs
```

Visit [http://localhost:6006](http://localhost:6006) in your browser to view training progress.

---

## Features

- Support for both cached and streaming data pipelines
- Efficient `nC2` sampling via generator or precompute mode
- TensorBoard integration for real-time training monitoring
- Saves best and per-epoch model checkpoints for reproducibility
- Compatible with both CPU and GPU

---

## Notes

- Use streaming mode if cache files exceed available memory (recommended for >10M pairs).
- Set `persistent_workers=True` and tune `prefetch_factor` in `DataLoader` for large-scale runs.
- Best model is saved in `modelData/best_model.pt`.

---

## Author

**Akshit Agarwal**  
MS in Computer Science, UC San Diego  
Email: aka002@ucsd.edu  
GitHub: [akshitagarwal1998](https://github.com/akshitagarwal1998)