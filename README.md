# Exploring the Efficacy of the Mixup Approach in Graph Neural Networks

This repository contains implementations of Graph Convolutional Networks (GCN) and Graph Isomorphism Network (GIN) for graph classification, including Mixup augmentation techniques. The code is designed to support multiple datasets and features efficient training using PyTorch and PyTorch Geometric.

## Features
- Implementation of **Graph Convolutional Networks (GCN)** and **Graph Isomorphism Network (GIN)** for graph classification.
- Support for **Mixup** augmentation method in the embedding space.
- Support for modifying the training set ratio in GCN training.
- GPU acceleration using PyTorch and multiprocessing for faster training.

## Installation

### Dependencies(with python == 3.8.2)

#### Ensure you have the following dependencies installed:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install scikit-learn numpy matplotlib optuna tqdm
```
## Data Preparation
For GCN and GIN just create a folder named 'cache_new' on the same level with this repo's folder, for GCN_with_r create a folder named 'cache_r'. Then the left things will be done by the code automatically.

## Running the Code
We take GCN as an example:
```bash
#running without mixup
CUDA_VISIBLE_DEVICES=0 python3 GCN/main.py --dataset=COLLAB

#running with mixup
CUDA_VISIBLE_DEVICES=0 python3 GCN/main.py --dataset=COLLAB --mixup
```

When running GCN_with_r, there is one more parameter to be set
```bash
#running without mixup
CUDA_VISIBLE_DEVICES=0 python3 GCN_with_r/main.py --dataset=COLLAB --r=0.8

#running with mixup
CUDA_VISIBLE_DEVICES=0 python3 GCN_with_r/main.py --dataset=COLLAB --mixup --r=0.8
```
## Hyperparameter Tuning
The optimal hyperparameter configurations are stored separately in the hyper and hyper_gin directories. To update them, please delete the existing hyperparameter configurations in the respective folders before proceeding.


