# Dockerfile for EchoQuality
# Provides a containerized environment for running the EchoQuality model

FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install Jupyter
RUN pip install --no-cache-dir jupyter matplotlib

# Set up Jupyter configuration
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py

# Create directories for project structure if they don't exist
RUN mkdir -p results/mask_images/original results/mask_images/before results/mask_images/after preprocessed_data/example_study weights

# Expose port for Jupyter
EXPOSE 8888

# Expose port for MLflow
EXPOSE 5000

# Set entrypoint to run Jupyter by default
ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0"]
