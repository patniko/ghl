# Makefile for EchoQuality project
# Provides easy commands for setting up environment, running Jupyter, and training models

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
	@echo "${YELLOW}EchoQuality Makefile Commands:${NC}"
	@echo ""
	@echo "${GREEN}Setup Commands:${NC}"
	@echo "  make setup              Setup Poetry environment"
	@echo ""
	@echo "${GREEN}Run Commands:${NC}"
	@echo "  make jupyter            Run Jupyter notebook"
	@echo "  make train              Run model training with default parameters"
	@echo "  make inference          Run inference on all folders in raw_data/ directory"

	@echo ""
	@echo "${GREEN}Docker Commands:${NC}"
	@echo "  make build-docker       Build the Docker image"
	@echo "  make build-jupyter      Build the Docker image for Jupyter"
	@echo "  make run-docker         Run Docker container"
	@echo "  make run-jupyter        Run Jupyter Docker container"
	@echo ""
	@echo "${GREEN}Utility Commands:${NC}"
	@echo "  make clean              Clean up temporary files"
	@echo ""
	@echo "Example usage:"
	@echo "  make jupyter JUPYTER_PORT=8889"

# Setup commands
.PHONY: setup
setup:
	@echo "${YELLOW}Setting up Poetry environment...${NC}"
	@$(POETRY) install
	@echo "${GREEN}Poetry environment setup complete!${NC}"

# Run commands
.PHONY: jupyter
jupyter:
	@echo "${YELLOW}Starting Jupyter notebook on port $(JUPYTER_PORT)${NC}"
	@$(POETRY) run python run_jupyter.py --port $(JUPYTER_PORT)

.PHONY: train
train:
	@echo "${YELLOW}Running model training...${NC}"
	@$(POETRY) run python -m training.train_quality_model
	@echo "${GREEN}Training complete!${NC}"

.PHONY: inference
inference:
	@echo "${YELLOW}Running EchoQuality inference on all folders in raw_data/ directory...${NC}"
	@$(POETRY) run python -m inference.inference --data_dir ./raw_data --output ./results/inference_output
	@echo "${GREEN}Inference complete! Results saved to ./results/inference_output${NC}"

# Docker commands
.PHONY: build-docker
build-docker:
	@echo "${YELLOW}Building Docker image...${NC}"
	@docker build -t echoquality .
	@echo "${GREEN}Docker image built successfully!${NC}"

.PHONY: build-jupyter
build-jupyter:
	@echo "${YELLOW}Building Docker image for Jupyter...${NC}"
	@docker build -t echoquality-jupyter -f Dockerfile.jupyter .
	@echo "${GREEN}Jupyter Docker image built successfully!${NC}"

.PHONY: run-docker
run-docker:
	@echo "${YELLOW}Running Docker container...${NC}"
	@docker run -d --name echoquality-container -p $(JUPYTER_PORT):8888 echoquality
	@echo "${GREEN}Docker container started!${NC}"
	@echo "You can access Jupyter notebook at http://localhost:$(JUPYTER_PORT)"
	@echo "To attach to the container: docker exec -it echoquality-container /bin/bash"

.PHONY: run-jupyter
run-jupyter:
	@echo "${YELLOW}Running Jupyter Docker container...${NC}"
	@docker run -it --rm -p $(JUPYTER_PORT):8888 -v $(PWD):/workspace echoquality-jupyter
	@echo "${GREEN}Jupyter Docker container started!${NC}"

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
	@rm -rf quality_results.json
	@echo "${GREEN}Cleanup complete!${NC}"

# Default target
.DEFAULT_GOAL := help
