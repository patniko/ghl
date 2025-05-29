# Use the NVIDIA PyTorch container as a base
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace
ENV PIP_ROOT_USER_ACTION=ignore

# Install Poetry
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/

# Configure Poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Copy poetry configuration
COPY pyproject.toml ./

# Install dependencies
# Using --no-root because we've set package-mode = false in pyproject.toml
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the HuBERT-ECG files
COPY . /workspace/

# Set up environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Create directories for project structure if they don't exist
RUN mkdir -p results preprocessed_data raw_data training_data weights

# Default command to keep container running
CMD ["bash"]
