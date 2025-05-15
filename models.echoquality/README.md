# EchoQuality

A deep learning tool for assessing the quality of echocardiogram videos.

## Overview

EchoQuality uses a pre-trained R(2+1)D model to classify the quality of echocardiogram videos. The model analyzes DICOM files and predicts whether the video quality is acceptable, helping to filter out low-quality studies that might lead to inaccurate diagnoses.

## Features

- Automatic masking of non-ultrasound regions in echo videos
- Quality assessment with detailed scoring and feedback
- MLflow integration for experiment tracking
- GradCAM visualization to understand model decisions
- Comprehensive training pipeline with data augmentation
- Jupyter notebook for interactive exploration

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/echoquality.git
cd echoquality

# Install dependencies using Poetry
make setup
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/echoquality.git
cd echoquality

# Install dependencies using pip
pip install -r requirements.txt
```

## Quick Start

### Running the Demo Notebook

The easiest way to get started is to run the demo Jupyter notebook:

```bash
make jupyter
```

Then open `EchoQualityDemo.ipynb` in your browser.

### Command Line Usage

For batch processing of DICOM files:

```bash
python EchoPrime_qc.py
```

The script will process all DICOM files in the `./model_data/example_study` directory by default.

For more advanced inference options:

```bash
python inference.py --input /path/to/dicom/files --output /path/to/save/results --gradcam
```

## Model Training

To train the model on your own dataset:

1. Prepare a CSV file with annotations (see `TRAINING.md` for details)
2. Configure training parameters in `train_quality_model.py`
3. Run the training script:

```bash
make train
```

For more detailed training instructions, see [TRAINING.md](TRAINING.md).

## Project Structure

```
.
├── EchoPrime_qc.py           # Main script for quality assessment
├── echo_data_augmentation.py # Data augmentation utilities
├── echo_model_evaluation.py  # Model evaluation and visualization tools
├── example_training.py       # Example script for training
├── inference.py              # Script for batch inference
├── train_quality_model.py    # Main training script
├── EchoQualityDemo.ipynb     # Demo Jupyter notebook
├── Makefile                  # Makefile with useful commands
├── pyproject.toml            # Poetry configuration
├── README.md                 # This file
├── TRAINING.md               # Detailed training instructions
├── video_quality_model.pt    # Pre-trained model weights
├── mask_images/              # Directory for masked image outputs
└── model_data/               # Directory for model data and examples
```

## Makefile Commands

The project includes a Makefile with useful commands:

```bash
# Show available commands
make help

# Setup environment
make setup
make setup-dev          # Setup with development dependencies
make update-lock        # Update poetry.lock file

# Run commands
make jupyter            # Run Jupyter notebook
make train              # Run model training
make inference          # Run inference on example data

# Docker commands
make build-docker       # Build the Docker image
make build-jupyter      # Build the Docker image for Jupyter
make run-docker         # Run Docker container
make run-jupyter        # Run Jupyter Docker container

# MLflow commands
make mlflow-server      # Start MLflow tracking server

# Utility commands
make clean              # Clean up temporary files
```

## Docker Support

The project includes Docker support for easy deployment and reproducibility:

```bash
# Build and run the main Docker image
make build-docker
make run-docker

# Build and run the Jupyter-specific Docker image
make build-jupyter
make run-jupyter
```

The Jupyter Docker container mounts the current directory as a volume, allowing you to edit files on your host machine and have the changes reflected in the container.

## Dependencies

- torch
- numpy
- tqdm
- opencv-python
- pydicom
- torchvision
- mlflow
- matplotlib
- scikit-learn
- pandas
- seaborn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
[Citation information]
```

## Acknowledgments

- The model architecture is based on the R(2+1)D model from torchvision
- [Add other acknowledgments as needed]
