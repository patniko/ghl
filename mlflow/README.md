# MLflow Project

This project uses MLflow to track machine learning experiments.

## Setup and Running

There are several ways to set up and run the MLflow server:

### Using Make (Recommended)

```bash
# Install Poetry and dependencies
make init

# Start the MLflow server
make mlflow-start

# Run the test script
make run-test

# Clean MLflow artifacts and runs (use with caution)
make clean

# Show help
make help
```

### Using the Shell Script

```bash
./start_mlflow.sh
```

### Using the Python Script

```bash
./start_mlflow.py
```

Or:

```bash
python start_mlflow.py
```

All methods will:
1. Install the required dependencies using Poetry (including the project package)
2. Start the MLflow server on http://127.0.0.1:5000

## Running Experiments

After starting the MLflow server, you can run experiments using:

```bash
# Using Make
make run-test

# Or directly with Poetry
poetry run python test.py
```

This will train a RandomForestRegressor model and log the results to MLflow.

## Accessing the MLflow UI

Once the server is running, you can access the MLflow UI by opening http://127.0.0.1:5000 in your web browser.
