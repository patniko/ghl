# HuBERT-ECG Code Organization

This document explains the organization and functionality of the HuBERT-ECG codebase after restructuring.

## Core Components

### Feature Extraction (`tools/feature_extraction.py`)
Previously `dumping.py`, this module contains the code and entry points to compute and dump feature descriptors of raw ECG fragments. These descriptors include:
- Time-frequency features
- 39 MFCC coefficients
- Time-frequency features + 13 MFCC coefficients
- Latent representations extracted from $i^{th}$ encoding layer, $i = 0, 1, 2..., 11$

### Clustering (`tools/clustering.py`)
After dumping ECG feature descriptors, one can proceed with the offline clustering step, that is, clustering the feature descriptor and fit a K-means clustering model. 
This module implements such a step, saves the resulting model, which is necessary to produce labels to use in the pre-training, and provides evaluation functions to quantify the clustering quality.

### Dataset (`training/dataset.py`)
The ECGDataset implementation, responsible for iterating over a CSV file representing an ECG dataset (normally train/val/test sets) and providing the data loader with ECGs, ECG feature descriptors, and ECG up/downstream labels.

### Model Architectures (`training/models/`)

#### Pre-training Architecture (`training/models/hubert_ecg.py`)
The architecture of HuBERT-ECG used during pre-training, featuring projection & look-up embedding matrices for self-supervised learning.

#### Fine-tuning Architecture (`training/models/hubert_ecg_classification.py`)
The architecture used during fine-tuning or training from scratch, where the projection & look-up embedding matrices are replaced by a classification head.

### Training Components (`training/`)

#### Pre-training (`training/pretrain.py`)
Contains the code to pre-train HuBERT-ECG in a self-supervised manner. Run `python training/pretrain.py --help` for detailed usage information.

#### Fine-tuning (`training/finetune.py`)
Contains the code to fine-tune and train from scratch HuBERT-ECG in a supervised manner. Run `python training/finetune.py --help` for detailed usage information. Also see `training/finetune.sh` for example usage.

#### Training Utilities (`training/utils.py`)
Contains utility functions used throughout the training process.

### Evaluation (`tools/`)

#### Testing/Evaluation (`tools/test.py`)
Contains the code to evaluate fine-tuned or fully trained HuBERT-ECG instances on test data. Run `python tools/test.py --help` for detailed usage information. Also see `training/test.sh` for example usage.

#### Evaluation Metrics (`tools/evaluation.py`)
Previously `metrics.py`, contains functions for computing various evaluation metrics including AUROC, AUPRC, F1-score, and accuracy.

### Preprocessing (`preprocessors/`)

#### ECG Preprocessing (`preprocessors/ecg_preprocessing.py`)
Contains functions for preprocessing ECG data according to HuBERT-ECG specifications, including normalization, filtering, and data preparation for model inference.

### Inference (`inference/`)

#### Demonstration Notebooks
- `inference/HuBERT-ECG-Demo.ipynb`: Main demonstration notebook
- `inference/EchoPrimeDemo.ipynb`: Additional demo notebook

## Usage Examples

### Feature Extraction
```bash
python tools/feature_extraction.py --data_path /path/to/ecg/data --feature_type mfcc
```

### Clustering
```bash
python tools/clustering.py --features_path ./features.pkl --num_clusters 100
```

### Pre-training
```bash
python training/pretrain.py --data_path /path/to/data --num_epochs 100
```

### Fine-tuning
```bash
python training/finetune.py --dataset ptbxl --task all --pretrained_model ./model.pt
```

### Testing
```bash
python tools/test.py --model_path ./model.pt --dataset ptbxl --task all
```

## Shell Scripts

Convenient shell scripts are provided in the `training/` directory:
- `training/finetune.sh`: Example fine-tuning commands
- `training/test.sh`: Example testing commands

This reorganized structure provides better separation of concerns and makes the codebase more maintainable and easier to navigate.
