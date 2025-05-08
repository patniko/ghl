# HuBERT-ECG: A Self-Supervised Foundation Model for Broad and Scalable Cardiac Application

[![medrXiv](https://img.shields.io/badge/medRxiv-green)](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v1)
License: CC BY-NC 4.0


📢 [[Models](https://huggingface.co/Edoardo-BS)] 

## Abstract
Deep learning models have shown remarkable performance in electrocardiogram (ECG) analysis, but their success has been constrained by the limited availability and size of ECG datasets, resulting in systems that are more task specialists than versatile generalists. In this work, we introduce HuBERT-ECG, a foundation ECG model pre-trained in a self-supervised manner on a large and diverse dataset of 9.1 million 12-lead ECGs encompassing 164 cardiovascular conditions. By simply adding an output layer, HuBERT-ECG can be fine-tuned for a wide array of downstream tasks, from diagnosing diseases to predicting future cardiovascular events. Across diverse real-world scenarios, HuBERT-ECG achieves AUROCs from 84.3% in low-data settings to 99% in large-scale setups. When trained to detect 164 overlapping conditions simultaneously, our model delivers AUROCs above 90% and 95% for 140 and 94 diseases, respectively. HuBERT-ECG also predicts death events within a 2-year follow-up with an AUROC of 93.4%. We release models and code.

## News
- [12/2024] Reproducibility has never been easier! Training, validation, and test splits ready to use in the reproducibility folder!
- [12/2024] Pre-trained models are easily downloadable from Hugging Face using `AutoModel.from_pretrained`
- [11/2024] Pre-trained models are freely available on HuggingFace
- [11/2024] This repository has been made public!

## Model weights
All our models are accessible on Hugging Face [(https://huggingface.co/Edoardo-BS)] under CC BY-NC 4.0 license

**Remember to pre-process your data before feeding HuBERT-ECG. Take a look at Data and Preprocessing section in the paper**

## Installation

### Using Poetry (Recommended)
We now use Poetry for dependency management. To install the project with Poetry:

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

### Using pip
Alternatively, you can still install dependencies using pip:

```bash
pip install -r requirements.txt
```

Full installation time may take up to 1 minute.

### Setting up the PTB-XL Dataset
To use the HuBERT-ECG model with the PTB-XL dataset, you need to download and process the dataset. We provide a convenient script to do this:

```bash
# Using Poetry
make ptbxl-setup

# Or directly
poetry run python scripts/setup_ptbxl.py
```

This will download the PTB-XL dataset from PhysioNet, extract it, and process the ECG data to create .npy files that match the filenames in the CSV files. The processed data will be stored in the `data/ptbxl/processed` directory.

## Docker Integration

We provide Docker integration for easy setup and reproducibility. You can run HuBERT-ECG with Jupyter Notebook using Docker:

```bash
# Build the Docker image
make build-jupyter

# Run Jupyter in a Docker container
make run-jupyter
```

This will start a Jupyter Notebook server accessible at http://localhost:8888.

## Jupyter Notebook

We provide a demo notebook (`HuBERT-ECG-Demo.ipynb`) that shows how to use the HuBERT-ECG model for ECG classification. You can run it using:

```bash
# Using Poetry
make jupyter

# Or directly
python run_jupyter.py
```

## Reproducibility
In the `reproducibility` folder you can find all train, validation, and test splits we used in our work as .csv files. You simply have to follow the instructions in the `reproducibility/README.md` to reproduce our results.
As an example, you can easily fine-tune and evaluate an instance of HuBERT-ECG on PTB-XL All dataset, as shown in .sh scripts `finetune.sh` and `test.sh`.
Prediction on a single instance takes less than 1 second on an A100 GPU node.

## Makefile Commands

We provide a Makefile with useful commands:

```bash
# Show available commands
make help

# Setup Poetry environment
make setup

# Setup Poetry environment with dev dependencies
make setup-dev

# Run Jupyter notebook
make jupyter

# Build Docker image for Jupyter
make build-jupyter

# Run Jupyter in a Docker container
make run-jupyter

# Clean up temporary files
make clean

# Run tests
make test
```

## 📚 Citation
If you use our models or find our work useful, please consider citing us:
```
doi: https://doi.org/10.1101/2024.11.14.24317328
```
