# HuBERT-ECG Training Guide

This document provides comprehensive guidance on training HuBERT-ECG models, including pre-training and fine-tuning procedures.

## Overview

HuBERT-ECG follows a two-stage training approach:
1. **Self-supervised Pre-training**: Learn general ECG representations using masked prediction
2. **Supervised Fine-tuning**: Adapt pre-trained models for specific downstream tasks

## Pre-training

### Self-Supervised Learning Pipeline

The pre-training process follows these steps:

1. **Feature Extraction**: Extract MFCC and time-frequency features from ECG signals
2. **Clustering**: Apply K-means clustering to create pseudo-labels
3. **Masked Prediction**: Train model to predict masked ECG segments

### Running Pre-training

```bash
# Basic pre-training
python training/pretrain.py \
    --data_path /path/to/ecg/data \
    --output_dir ./pretrained_models \
    --num_epochs 100 \
    --batch_size 32

# Advanced pre-training with custom parameters
python training/pretrain.py \
    --data_path /path/to/ecg/data \
    --output_dir ./pretrained_models \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --mask_prob 0.15 \
    --num_clusters 100 \
    --feature_type mfcc
```

### Pre-training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_path` | Path to ECG training data | Required |
| `--output_dir` | Directory to save models | `./models` |
| `--num_epochs` | Number of training epochs | 100 |
| `--batch_size` | Training batch size | 32 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--mask_prob` | Masking probability | 0.15 |
| `--num_clusters` | Number of K-means clusters | 100 |
| `--feature_type` | Feature type for clustering | `mfcc` |

## Fine-tuning

### Supervised Fine-tuning for Downstream Tasks

Fine-tuning adapts pre-trained models for specific classification tasks:

```bash
# Fine-tune on PTB-XL dataset
python training/finetune.py \
    --dataset ptbxl \
    --task all \
    --pretrained_model ./pretrained_models/hubert_ecg.pt \
    --output_dir ./finetuned_models \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-5

# Fine-tune on Chapman dataset
python training/finetune.py \
    --dataset chapman \
    --task rhythm \
    --pretrained_model ./pretrained_models/hubert_ecg.pt \
    --output_dir ./finetuned_models \
    --num_epochs 30 \
    --batch_size 16
```

### Fine-tuning Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (ptbxl, chapman, etc.) | Required |
| `--task` | Task type (all, rhythm, form, etc.) | Required |
| `--pretrained_model` | Path to pre-trained model | Required |
| `--output_dir` | Directory to save fine-tuned models | `./models` |
| `--num_epochs` | Number of fine-tuning epochs | 50 |
| `--batch_size` | Training batch size | 16 |
| `--learning_rate` | Learning rate | 1e-5 |
| `--freeze_encoder` | Freeze encoder layers | False |

### Supported Datasets and Tasks

#### PTB-XL Dataset
- **all**: All diagnostic statements
- **diag**: Diagnostic classes
- **diag_subclass**: Diagnostic subclasses
- **diag_superclass**: Diagnostic superclasses
- **form**: Form statements
- **rhythm**: Rhythm statements

#### Chapman Dataset
- **rhythm**: Rhythm classification
- **all**: All available labels

#### Other Datasets
- **CPSC**: China Physiological Signal Challenge
- **Georgia**: Georgia 12-lead ECG Challenge
- **Hefei**: Hefei Database
- **Ningbo**: Ningbo Database

## Model Architectures

### Pre-training Architecture (`training/models/hubert_ecg.py`)
- Transformer encoder with 12 layers
- Projection and lookup embedding matrices
- Masked prediction head
- Self-supervised learning objectives

### Fine-tuning Architecture (`training/models/hubert_ecg_classification.py`)
- Pre-trained transformer encoder
- Classification head for downstream tasks
- Task-specific output layers

## Training Scripts

### Shell Scripts for Easy Training

The project includes convenient shell scripts:

```bash
# Fine-tuning script
./code/finetune.sh

# Testing script
./code/test.sh
```

### Example Fine-tuning Script
```bash
#!/bin/bash
python training/finetune.py \
    --dataset ptbxl \
    --task all \
    --pretrained_model hubert_ecg_base \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --num_epochs 50 \
    --output_dir ./results/ptbxl_all
```

## Data Preparation

### ECG Data Format
- **Input**: Raw ECG signals as .npy files
- **Shape**: (n_leads, n_samples) for multi-lead ECG
- **Sampling Rate**: 500 Hz (recommended)
- **Duration**: 10 seconds (5000 samples)

### Preprocessing Pipeline
```python
from preprocessors.ecg_preprocessing import prepare_ecg_for_inference

# Preprocess ECG data
processed_ecg = prepare_ecg_for_inference(
    raw_ecg_data,
    sampling_rate=500,
    target_length=5000,
    downsampling_factor=5,
    normalize=True
)
```

## Feature Extraction and Clustering

### Extract Features for Pre-training
```bash
# Extract MFCC features
python tools/feature_extraction.py \
    --data_path /path/to/ecg/data \
    --feature_type mfcc \
    --output_path ./features/mfcc_features.pkl

# Extract time-frequency features
python tools/feature_extraction.py \
    --data_path /path/to/ecg/data \
    --feature_type time_freq \
    --output_path ./features/tf_features.pkl
```

### Perform Clustering
```bash
# K-means clustering
python tools/clustering.py \
    --features_path ./features/mfcc_features.pkl \
    --num_clusters 100 \
    --output_path ./clusters/kmeans_model.pkl
```

## Evaluation and Testing

### Model Evaluation
```bash
# Evaluate fine-tuned model
python tools/test.py \
    --model_path ./finetuned_models/ptbxl_all.pt \
    --dataset ptbxl \
    --task all \
    --test_split test
```

### Metrics
- **AUROC**: Area Under ROC Curve
- **AUPRC**: Area Under Precision-Recall Curve
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Classification accuracy

## Best Practices

### Pre-training Tips
1. Use large, diverse ECG datasets for pre-training
2. Experiment with different masking strategies
3. Monitor clustering quality during training
4. Use appropriate feature types for your domain

### Fine-tuning Tips
1. Start with lower learning rates (1e-5 to 1e-4)
2. Use smaller batch sizes for fine-tuning
3. Consider freezing encoder layers for small datasets
4. Apply data augmentation for better generalization

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ system RAM
- **Storage**: SSD recommended for faster data loading

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or sequence length
2. **Slow Training**: Use mixed precision training or smaller models
3. **Poor Performance**: Check data preprocessing and normalization
4. **Convergence Issues**: Adjust learning rate or optimizer settings

### Performance Optimization
- Use DataLoader with multiple workers
- Enable mixed precision training with AMP
- Use gradient accumulation for larger effective batch sizes
- Consider model parallelism for very large models

This guide provides the foundation for successfully training HuBERT-ECG models. For specific use cases, refer to the reproducibility documentation and example scripts.
