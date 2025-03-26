# RCSB PDB Neural Network Task

## Overview
This project is a solution for the RCSB PDB - SDSC Data Science Assistant task. The goal is to train and evaluate a fully connected neural network that can classify whether two proteins have similar 3D structures based on 1D BioZernike descriptors.

## Objective
Train a neural network using PyTorch to predict structural similarity between pairs of proteins using pairwise distance features computed from geometric and Zernike moments.

## Dataset
Two datasets are provided:
- `cath_moments.tsv`: Training dataset
- `ecod_moments.tsv`: Evaluation dataset

### Each row contains:
- Column 0: Protein shape class identifier
- Columns 1–17: Geometric descriptors
- Columns 18–393: Zernike moments

## Preprocessing
- Pairs are generated between all proteins in the training set.
- For each pair, two distances are computed:
  - Geometric distance: L2 norm between geometric descriptors
  - Zernike distance: L2 norm between Zernike moments
- The label is 1 if proteins have the same shape class, otherwise 0.

## Model
- Fully connected neural network trained on distance features
- Model variants:
  - Single neuron (logistic regression)
  - Fully connected with 32, 64, 128, 256 neurons
- L2 regularization applied only to weights

## Training
- Training is conducted over 100 epochs
- Different loss functions (e.g., BCE, BCEWithLogits) and optimizers (e.g., SGD, Adam) are experimented with
- Training data is imbalanced; handled using `WeightedRandomSampler`

## Evaluation
- Evaluation performed after every epoch using `ecod_moments.tsv`
- Metrics logged:
  - ROC AUC
  - PR AUC
  - MCC (Matthews Correlation Coefficient)

## Tools & Libraries
- PyTorch
- TensorBoard (for logging metrics and histograms)
- scikit-learn (for evaluation metrics)

## Project Structure
```
├── dataset.py        # Dataset and DataLoader logic
├── model.py          # Neural network architecture
├── train.py          # Training and evaluation loop
├── requirements.txt  # Required packages
├── results/          # Screenshots and result summaries
└── tensorboard_logs/ # Logs for visualization in TensorBoard
```

## How to Run
Instructions will be added after implementation.

## Results
This section will include summary observations and screenshots from TensorBoard once training is completed.

## Contributors
- Akshit Agarwal

## License
MIT License
