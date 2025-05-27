# Makefile for HuBERT-ECG project
# Provides easy commands for running Jupyter, and Docker tasks

# Variables
PYTHON := python3
POETRY := poetry
JUPYTER_PORT := 8888

# Colors for terminal output
YELLOW := \033[1;33m
GREEN := \033[1;32m
NC := \033[0m # No Color

# Help command
.PHONY: help
help:
	@echo "${YELLOW}HuBERT-ECG Makefile Commands:${NC}"
	@echo ""
	@echo "${GREEN}Setup Commands:${NC}"
	@echo "  make setup              Setup Poetry environment"
	@echo "  make setup-dev          Setup Poetry environment with dev dependencies"
	@echo "  make ptbxl-setup        Download and setup the PTB-XL dataset"
	@echo "  make dicom-setup        Convert DICOM ECG files to NPY format"
	@echo "  make dicom-inference    Run HuBERT-ECG inference on converted DICOM files"
	@echo "  make dicom-metadata     Extract comprehensive metadata from DICOM files"
	@echo "  make dicom-visualize    Create ECG signal visualizations"
	@echo ""
	@echo "${GREEN}Run Commands:${NC}"
	@echo "  make jupyter            Run Jupyter notebook"
	@echo ""
	@echo "${GREEN}Docker Commands:${NC}"
	@echo "  make build-jupyter      Build the Docker image for Jupyter"
	@echo "  make run-jupyter        Run Jupyter in a Docker container"
	@echo ""
	@echo "${GREEN}Utility Commands:${NC}"
	@echo "  make clean              Clean up temporary files"
	@echo "  make test               Run tests"
	@echo ""
	@echo "Example usage:"
	@echo "  make jupyter JUPYTER_PORT=8889"

# Setup commands
.PHONY: setup
setup:
	@echo "${YELLOW}Setting up Poetry environment...${NC}"
	@$(POETRY) install

.PHONY: setup-dev
setup-dev:
	@echo "${YELLOW}Setting up Poetry environment with dev dependencies...${NC}"
	@$(POETRY) install --with dev

.PHONY: ptbxl-setup
ptbxl-setup:
	@echo "${YELLOW}Setting up PTB-XL dataset...${NC}"
	@$(POETRY) run python scripts/setup_ptbxl.py
	@echo "${GREEN}PTB-XL dataset setup complete!${NC}"

.PHONY: dicom-setup
dicom-setup:
	@echo "${YELLOW}Converting DICOM ECG files to NPY format...${NC}"
	@$(POETRY) run python scripts/dicom_to_npy.py
	@echo "${GREEN}DICOM conversion complete!${NC}"

.PHONY: dicom-inference
dicom-inference:
	@echo "${YELLOW}Running HuBERT-ECG inference on converted DICOM files...${NC}"
	@$(POETRY) run python scripts/run_dicom_inference.py
	@echo "${GREEN}DICOM inference complete!${NC}"

.PHONY: dicom-metadata
dicom-metadata:
	@echo "${YELLOW}Extracting comprehensive metadata from DICOM files...${NC}"
	@$(POETRY) run python scripts/dump_dicom_metadata.py
	@echo "${GREEN}DICOM metadata extraction complete!${NC}"

.PHONY: dicom-visualize
dicom-visualize:
	@echo "${YELLOW}Creating ECG signal visualizations...${NC}"
	@$(POETRY) run python scripts/visualize_ecg_signals.py
	@echo "${GREEN}ECG visualization complete!${NC}"

# Run commands
.PHONY: jupyter
jupyter:
	@echo "${YELLOW}Starting Jupyter notebook on port $(JUPYTER_PORT)${NC}"
	@$(POETRY) run python run_jupyter.py --port $(JUPYTER_PORT)

# Docker commands
.PHONY: build-jupyter
build-jupyter:
	@echo "${YELLOW}Building Docker image for Jupyter...${NC}"
	@docker build -t hubert-ecg-jupyter:latest -f Dockerfile.jupyter .

.PHONY: run-jupyter
run-jupyter:
	@echo "${YELLOW}Running Jupyter container on port $(JUPYTER_PORT)...${NC}"
	@docker run -it --rm -p $(JUPYTER_PORT):8888 -v $(PWD):/workspace hubert-ecg-jupyter:latest

# Utility commands
.PHONY: clean
clean:
	@echo "${YELLOW}Cleaning up temporary files...${NC}"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".DS_Store" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name "*.egg" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".coverage" -exec rm -rf {} +
	@find . -type d -name "htmlcov" -exec rm -rf {} +
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@rm -rf build/
	@rm -rf dist/
	@rm -rf .coverage
	@echo "${GREEN}Cleanup complete!${NC}"

.PHONY: test
test:
	@echo "${YELLOW}Running tests...${NC}"
	@$(POETRY) run pytest

# Default target
.DEFAULT_GOAL := help
