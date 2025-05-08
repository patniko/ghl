#!/bin/bash

# Install dependencies including the root package
poetry install

# Start the MLflow server
poetry run mlflow server --host 127.0.0.1 --port 5000
