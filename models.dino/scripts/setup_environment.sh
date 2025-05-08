#!/bin/bash

echo "Setting up Poetry environment..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Install dependencies
echo "Installing project dependencies..."
poetry install

echo "Environment ready. Run commands with: poetry run <command> or poetry shell"
