# HuBERT-ECG Project Structure

This document describes the organizational structure of the HuBERT-ECG project, which has been restructured to follow clean architecture principles and match the organization of other model projects.

## Project Structure

```
models.hubertecg/
├── Project files (Dockerfile, README.md, pyproject.toml, etc.)
├── data/                        # Sample data and datasets
│   └── __init__.py
├── docs/                        # Documentation
│   ├── __init__.py
│   ├── STRUCTURE.md             # This file - project organization
│   ├── TRAINING.md              # Training documentation
│   └── REPRODUCIBILITY.md      # Reproducibility guidelines
├── inference/                   # Inference scripts and demos
│   ├── __init__.py
│   ├── HuBERT-ECG-Demo.ipynb   # Main demonstration notebook
│   └── EchoPrimeDemo.ipynb     # Additional demo notebook
├── preprocessors/               # Data preprocessing utilities
│   ├── __init__.py
│   └── ecg_preprocessing.py    # ECG data preprocessing functions
├── results/                     # Outputs, results, and visualizations
│   ├── __init__.py
│   └── figures/                # Research figures and plots
├── tools/                       # Utility scripts and helper functions
│   ├── __init__.py
│   ├── clustering.py           # K-means clustering for self-supervised learning
│   ├── feature_extraction.py   # Feature descriptor computation and dumping
│   ├── evaluation.py          # Metrics and evaluation functions
│   └── test.py                # Testing and evaluation scripts
├── training/                    # Training-related files
│   ├── __init__.py
│   ├── pretrain.py            # Self-supervised pre-training
│   ├── finetune.py            # Supervised fine-tuning
│   ├── dataset.py             # ECG dataset implementation
│   ├── utils.py               # Training utility functions
│   └── models/                # Model architectures
│       ├── __init__.py
│       ├── hubert_ecg.py      # Pre-training architecture
│       └── hubert_ecg_classification.py # Fine-tuning architecture
├── scripts/                     # Setup and utility scripts
│   └── (existing script files)
├── reproducibility/            # Reproducibility data and splits
│   └── (existing reproducibility files)
└── weights/                     # Model weights and checkpoints
    └── __init__.py
```

## Module Organization

### Core Components

#### `training/`
Contains all training-related functionality:
- **`pretrain.py`**: Self-supervised pre-training using HuBERT methodology
- **`finetune.py`**: Supervised fine-tuning for downstream tasks
- **`dataset.py`**: ECGDataset implementation for data loading
- **`utils.py`**: Training utility functions
- **`models/`**: Model architecture definitions
  - **`hubert_ecg.py`**: Pre-training architecture with projection & lookup embedding matrices
  - **`hubert_ecg_classification.py`**: Fine-tuning architecture with classification head

#### `tools/`
Utility tools and helper functions:
- **`clustering.py`**: K-means clustering for self-supervised learning labels
- **`feature_extraction.py`**: Computing and dumping ECG feature descriptors (MFCC, time-frequency features, latent representations)
- **`evaluation.py`**: Metrics and evaluation functions
- **`test.py`**: Testing and evaluation scripts

#### `preprocessors/`
Data preprocessing utilities:
- **`ecg_preprocessing.py`**: ECG data preprocessing according to HuBERT-ECG specifications
  - Signal normalization, filtering, resampling
  - Data preparation for model inference
  - Multi-lead ECG handling

#### `inference/`
Inference and demonstration:
- **`HuBERT-ECG-Demo.ipynb`**: Main demonstration notebook
- **`EchoPrimeDemo.ipynb`**: Additional demo notebook
- Ready-to-use inference examples

#### `results/`
Outputs and visualizations:
- **`figures/`**: Research figures, plots, and visualizations
- Storage for experimental results and outputs

### Data Flow

1. **Raw ECG Data** → `preprocessors/` → **Preprocessed Data**
2. **Preprocessed Data** → `tools/feature_extraction.py` → **Feature Descriptors**
3. **Feature Descriptors** → `tools/clustering.py` → **Cluster Labels**
4. **Data + Labels** → `training/pretrain.py` → **Pre-trained Model**
5. **Pre-trained Model** → `training/finetune.py` → **Fine-tuned Model**
6. **Fine-tuned Model** → `inference/` → **Predictions**

### Key Features

- **Clear Separation of Concerns**: Each directory has a specific purpose
- **Modular Design**: Components can be used independently
- **Comprehensive Documentation**: Each module is well-documented
- **Reproducibility**: All splits and configurations preserved
- **Easy Navigation**: Intuitive structure for new users

## Usage Examples

### Training a Model
```bash
# Pre-training
python training/pretrain.py --config config.yaml

# Fine-tuning
python training/finetune.py --dataset ptbxl --task all
```

### Running Inference
```bash
# Start Jupyter notebook
make jupyter

# Open inference/HuBERT-ECG-Demo.ipynb
```

### Preprocessing Data
```python
from preprocessors.ecg_preprocessing import prepare_ecg_for_inference

processed_ecg = prepare_ecg_for_inference(raw_ecg_data)
```

### Feature Extraction
```python
from tools.feature_extraction import extract_features

features = extract_features(ecg_data, method='mfcc')
```

This structure provides a clean, maintainable, and scalable foundation for the HuBERT-ECG project while preserving all existing functionality.
