# Asymmetric DINO V2 for ECG Analysis

A self-supervised learning framework for ECG signals where:
- Teacher model processes 12-lead ECG + patient metadata
- Student model processes only lead I
- Goal: Student matches teacher's diagnostic capability

## Documentation
- [ML Model](docs/ml_model.md) - Overview of the ML Model for healthcare applications
- [API Reference](docs/api.md) - Detailed API documentation
- [Getting Started](docs/getting_started.md) - Installation and usage guide
- [Metadata Schema](docs/metadata_schema.md) - Metadata integration specifications

## Features
- Pretraining on PTB-XL dataset
- Fine-tuning on custom datasets
- Metadata integration (age, sex, height, weight)
- Configurable training pipeline

## Installation
```bash
git clone https://github.com/yourusername/ecg_dino.git
cd ecg_dino
bash scripts/setup.sh
