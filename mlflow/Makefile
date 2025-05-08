.PHONY: init mlflow-start run-test clean help

# Default target
all: init

# Install Poetry and dependencies
init:
	@echo "Checking if Poetry is installed..."
	@if ! command -v poetry &> /dev/null; then \
		echo "Installing Poetry..."; \
		curl -sSL https://install.python-poetry.org | python3 -; \
	else \
		echo "Poetry is already installed."; \
	fi
	@echo "Installing project dependencies..."
	@poetry install

# Start MLflow server
mlflow-start:
	@echo "Starting MLflow server..."
	@poetry run mlflow server --host 127.0.0.1 --port 5000

# Run the test script
run-test:
	@echo "Running test script..."
	@poetry run python test.py

# Clean MLflow artifacts and runs (use with caution)
clean:
	@echo "Cleaning MLflow artifacts and runs..."
	@echo "This will remove all experiment data. Are you sure? [y/N]"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		rm -rf mlruns/* mlartifacts/*; \
		echo "MLflow data cleaned."; \
	else \
		echo "Clean operation cancelled."; \
	fi

# Help command
help:
	@echo "Available commands:"
	@echo "  make init          - Install Poetry and project dependencies"
	@echo "  make mlflow-start  - Start the MLflow server"
	@echo "  make run-test      - Run the test script"
	@echo "  make clean         - Clean MLflow artifacts and runs (use with caution)"
	@echo "  make help          - Show this help message"
