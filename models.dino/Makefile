# Asymmetric DINO V2 for ECG Analysis
# Makefile for common operations

# Poetry command
POETRY := poetry

# Directories
SRC_DIR := .
DATA_DIR := data
MODELS_DIR := models
TRAIN_DIR := train
UTILS_DIR := utils
TESTS_DIR := tests
SCRIPTS_DIR := scripts
CONFIGS_DIR := configs
DOCS_DIR := docs

# Configuration files
PRETRAIN_CONFIG := $(CONFIGS_DIR)/pretrain.yaml
FINETUNE_CONFIG := $(CONFIGS_DIR)/finetune.yaml

# Default number of GPUs
GPUS := 1

# Default phase
PHASE := pretrain

# Help target
.PHONY: help
help:
	@echo "Asymmetric DINO V2 for ECG Analysis"
	@echo ""
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  help                 Show this help message"
	@echo "  setup                Set up the Poetry environment and install dependencies"
	@echo "  download-data        Download the PTB-XL dataset"
	@echo "  test                 Run all tests"
	@echo "  test-unit            Run unit tests"
	@echo "  test-integration     Run integration tests"
	@echo "  lint                 Run linting checks"
	@echo "  format               Format code using black and isort"
	@echo "  pretrain             Run pretraining with default config"
	@echo "  finetune             Run finetuning with default config"
	@echo "  train                Run training with specified phase and config"
	@echo "                       Usage: make train PHASE=[pretrain|finetune] CONFIG=path/to/config.yaml GPUS=n"
	@echo "  clean                Remove build artifacts and temporary files"
	@echo "  clean-poetry         Remove Poetry environment"
	@echo "  clean-data           Remove downloaded datasets"
	@echo "  clean-all            Remove all generated files (venv, data, etc.)"

# Setup Poetry environment and install dependencies
.PHONY: setup
setup:
	@echo "Setting up Poetry environment..."
	$(POETRY) install
	@echo "Setup complete. Run commands with: poetry run <command> or poetry shell"

# Download PTB-XL dataset
.PHONY: download-data
download-data:
	@echo "Downloading PTB-XL dataset..."
	bash $(SCRIPTS_DIR)/download_ptbxl.sh

# Run all tests
.PHONY: test
test:
	@echo "Running all tests..."
	$(POETRY) run pytest $(TESTS_DIR)

# Run unit tests
.PHONY: test-unit
test-unit:
	@echo "Running unit tests..."
	$(POETRY) run pytest $(TESTS_DIR)/unit

# Run integration tests
.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	$(POETRY) run pytest $(TESTS_DIR)/integration -m integration

# Run linting
.PHONY: lint
lint:
	@echo "Running linting checks..."
	$(POETRY) run flake8 $(SRC_DIR)
	$(POETRY) run pylint $(MODELS_DIR) $(DATA_DIR) $(TRAIN_DIR) $(UTILS_DIR)

# Format code
.PHONY: format
format:
	@echo "Formatting code..."
	$(POETRY) run black $(SRC_DIR)
	$(POETRY) run isort $(SRC_DIR)

# Run pretraining
.PHONY: pretrain
pretrain:
	@echo "Running pretraining..."
	bash $(SCRIPTS_DIR)/train.sh pretrain -g $(GPUS)

# Run finetuning
.PHONY: finetune
finetune:
	@echo "Running finetuning..."
	bash $(SCRIPTS_DIR)/train.sh finetune -g $(GPUS)

# Run training with specified phase and config
.PHONY: train
train:
	@echo "Running $(PHASE) with config $(CONFIG) on $(GPUS) GPUs..."
	bash $(SCRIPTS_DIR)/train.sh $(PHASE) -c $(CONFIG) -g $(GPUS)

# Clean build artifacts and temporary files
.PHONY: clean
clean:
	@echo "Cleaning build artifacts and temporary files..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +

# Clean Poetry environment
.PHONY: clean-poetry
clean-poetry:
	@echo "Removing Poetry environment..."
	$(POETRY) env remove --all

# Clean downloaded data
.PHONY: clean-data
clean-data:
	@echo "Removing downloaded datasets..."
	rm -rf $(DATA_DIR)/physionet.org

# Clean everything
.PHONY: clean-all
clean-all: clean clean-poetry clean-data
	@echo "All generated files removed."
