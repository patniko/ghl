# Getting Started Guide

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ecg_dino.git
   cd ecg_dino
   ```

2. Set up the Poetry environment:
   ```bash
   bash scripts/setup_environment.sh
   ```
   
   This script will install Poetry if it's not already installed, and then install all project dependencies.

3. Download the PTB-XL dataset:
   ```bash
   bash scripts/download_ptbxl.sh
   ```

## Usage

### Pretraining

1. Configure pretraining parameters in `configs/pretrain.yaml`
2. Run pretraining:
   ```bash
   # Using the Makefile
   make pretrain
   
   # Or directly with the script
   bash scripts/train.sh pretrain
   ```

### Fine-tuning

1. Configure fine-tuning parameters in `configs/finetune.yaml`
2. Run fine-tuning:
   ```bash
   # Using the Makefile
   make finetune
   
   # Or directly with the script
   bash scripts/train.sh finetune
   ```

## Working with Custom Datasets

To use your own ECG dataset:

1. Prepare your data following the format described in the [Metadata Schema](metadata_schema.md)
2. Create a custom dataset instance:
   ```python
   from data.custom_dataset import CustomECGDataset
   
   dataset = CustomECGDataset(
       data_path='/path/to/your/data',
       student_leads=['I'],  # Leads for student model
       teacher_leads=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],  # Leads for teacher model
       metadata=True  # Include metadata if available
   )
   ```

3. Use the dataset for training:
   ```python
   from torch.utils.data import DataLoader
   
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
   ```

## Development with Poetry

This project uses Poetry for dependency management. Here are some common commands:

1. Activate the Poetry shell:
   ```bash
   poetry shell
   ```

2. Run a command within the Poetry environment without activating the shell:
   ```bash
   poetry run python train/train.py --config configs/pretrain.yaml
   ```

3. Add a new dependency:
   ```bash
   poetry add package-name
   ```

4. Add a development dependency:
   ```bash
   poetry add --group dev package-name
   ```

5. Update dependencies:
   ```bash
   poetry update
   ```

6. Run tests:
   ```bash
   # Using the Makefile
   make test
   
   # Or directly with Poetry
   poetry run pytest tests/
   ```

## ML Model Integration

For information on the ML Model and its integration with this framework, see the [ML Model documentation](ml_model.md).

## API Reference

For detailed information about the available classes and functions, see the [API Reference](api.md).
