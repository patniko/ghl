# Makefile for EchoPrime project
# Provides easy commands for downloading model data, setting up environment, and running Jupyter

# Variables
PYTHON := python3
POETRY := poetry
JUPYTER_PORT := 8888
CANDIDATES_DATA_URL := https://github.com/patniko/ghl/releases/download/v1.0.0/candidates_data.zip
EMBEDDING_P1_URL := https://github.com/patniko/ghl/releases/download/v1.0.0/candidate_embeddings_p1.pt
EMBEDDING_P2_URL := https://github.com/patniko/ghl/releases/download/v1.0.0/candidate_embeddings_p2.pt
ECHO_PRIME_ENCODER_URL := https://github.com/patniko/ghl/releases/download/v1.0.0/echo_prime_encoder.pt
VIEW_CLASSIFIER_URL := https://github.com/patniko/ghl/releases/download/v1.0.0/view_classifier.ckpt

# Colors for terminal output
YELLOW := \033[1;33m
GREEN := \033[1;32m
NC := \033[0m # No Color

# Help command
.PHONY: help
help:
	@echo "${YELLOW}EchoPrime Makefile Commands:${NC}"
	@echo ""
	@echo "${GREEN}Setup Commands:${NC}"
	@echo "  make init               Setup Poetry environment"
	@echo "  make download-weights   Download model weights and embeddings"
	@echo "  make setup-all          Setup environment and download weights"
	@echo ""
	@echo "${GREEN}Run Commands:${NC}"
	@echo "  make inference          Run inference on all folders in raw_data/ directory"
	@echo "  make train              Run model training"
	@echo "  make visualize          Create comprehensive result visualizations"
	@echo ""
	@echo "${GREEN}Docker Commands:${NC}"
	@echo "  make build-docker       Build the Docker image"
	@echo "  make build-jupyter      Build the Docker image for Jupyter"
	@echo "  make run-docker         Run Docker container"
	@echo "  make run-jupyter        Run Jupyter Docker container"
	@echo ""
	@echo "${GREEN}Utility Commands:${NC}"
	@echo "  make jupyter            Run Jupyter notebook"
	@echo "  make clean              Clean up temporary files"
	@echo ""
	@echo "Example usage:"
	@echo "  make jupyter JUPYTER_PORT=8889"

# Setup commands
.PHONY: init
setup:
	@echo "${YELLOW}Setting up Poetry environment...${NC}"
	@$(POETRY) install
	@echo "${GREEN}Poetry environment setup complete!${NC}"

.PHONY: download-weights
download-weights:
	@echo "${YELLOW}Downloading model weights and embeddings...${NC}"
	@mkdir -p weights/candidates_data
	
	@if [ ! -f "weights/echo_prime_encoder.pt" ]; then \
		echo "Downloading echo_prime_encoder.pt..."; \
		wget -q --show-progress $(ECHO_PRIME_ENCODER_URL) -O weights/echo_prime_encoder.pt; \
	else \
		echo "echo_prime_encoder.pt already exists."; \
	fi
	
	@if [ ! -f "weights/view_classifier.ckpt" ]; then \
		echo "Downloading view_classifier.ckpt..."; \
		wget -q --show-progress $(VIEW_CLASSIFIER_URL) -O weights/view_classifier.ckpt; \
	else \
		echo "view_classifier.ckpt already exists."; \
	fi
	
	@if [ ! -f "weights/candidates_data/candidates_data.zip" ]; then \
		echo "Downloading candidates_data.zip..."; \
		wget -q --show-progress $(CANDIDATES_DATA_URL) -O weights/candidates_data/candidates_data.zip; \
		echo "Extracting candidates_data.zip..."; \
		cd weights/candidates_data && unzip -q candidates_data.zip && rm candidates_data.zip; \
	else \
		echo "candidates_data.zip already exists."; \
	fi
	
	@if [ ! -f "weights/candidates_data/candidate_embeddings_p1.pt" ]; then \
		echo "Downloading candidate_embeddings_p1.pt..."; \
		wget -q --show-progress $(EMBEDDING_P1_URL) -O weights/candidates_data/candidate_embeddings_p1.pt; \
	else \
		echo "candidate_embeddings_p1.pt already exists."; \
	fi
	
	@if [ ! -f "weights/candidates_data/candidate_embeddings_p2.pt" ]; then \
		echo "Downloading candidate_embeddings_p2.pt..."; \
		wget -q --show-progress $(EMBEDDING_P2_URL) -O weights/candidates_data/candidate_embeddings_p2.pt; \
	else \
		echo "candidate_embeddings_p2.pt already exists."; \
	fi
	
	@echo "${GREEN}Model weights and embeddings download complete!${NC}"

# Run commands
.PHONY: jupyter
jupyter:
	@echo "${YELLOW}Starting Jupyter notebook on port $(JUPYTER_PORT)${NC}"
	@$(POETRY) run jupyter notebook --port $(JUPYTER_PORT)

.PHONY: inference
inference:
	@echo "${YELLOW}Cleaning results and preprocessed_data for clean run...${NC}"
	@rm -rf ./results/*
	@rm -rf ./preprocessed_data/*
	@echo "${YELLOW}Running EchoPrime inference on all folders in raw_data/ directory...${NC}"
	@$(POETRY) run python -m inference.inference --data_dir ./raw_data --output ./results/inference_output
	@echo "${GREEN}Inference complete! Results saved to ./results/inference_output${NC}"

.PHONY: visualize
visualize:
	@echo "${YELLOW}Creating comprehensive result visualizations...${NC}"
	@$(POETRY) run python scripts/visualize_results.py --results_dir ./results/inference_output --output_dir ./results/visualization_output
	@echo "${GREEN}Result visualization complete! Check ./results/visualization_output${NC}"

.PHONY: train
train:
	@echo "${YELLOW}Running EchoPrime model training...${NC}"
	@$(POETRY) run python -m training.echoprime_finetune train
	@echo "${GREEN}Training complete!${NC}"

# Docker commands
.PHONY: build-docker
build-docker:
	@echo "${YELLOW}Building Docker image...${NC}"
	@docker build -t echoprime .
	@echo "${GREEN}Docker image built successfully!${NC}"

.PHONY: build-jupyter
build-jupyter:
	@echo "${YELLOW}Building Docker image for Jupyter...${NC}"
	@docker build -t echoprime-jupyter -f Dockerfile.jupyter .
	@echo "${GREEN}Jupyter Docker image built successfully!${NC}"

.PHONY: run-docker
run-docker:
	@echo "${YELLOW}Running Docker container...${NC}"
	@docker run -d --name echoprime-container --gpus all -p $(JUPYTER_PORT):8888 echoprime
	@echo "${GREEN}Docker container started!${NC}"
	@echo "You can access Jupyter notebook at http://localhost:$(JUPYTER_PORT)"
	@echo "To attach to the container: docker exec -it echoprime-container /bin/bash"

.PHONY: run-jupyter
run-jupyter:
	@echo "${YELLOW}Running Jupyter Docker container...${NC}"
	@docker run -it --rm --gpus all -p $(JUPYTER_PORT):8888 -v $(PWD):/workspace echoprime-jupyter
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
	@echo "${GREEN}Cleanup complete!${NC}"

# All-in-one setup command
.PHONY: setup-all
all: setup download-weights

# Default target
.DEFAULT_GOAL := help
