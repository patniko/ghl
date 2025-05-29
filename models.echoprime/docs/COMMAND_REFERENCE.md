# 📖 EchoPrime Command Reference

Complete reference for all available commands, options, and usage patterns in EchoPrime.

## 🛠️ Makefile Commands

### Setup Commands

#### `make init`
Initialize the Poetry environment and create necessary directories.

```bash
make init
```

**What it does:**
- Installs all Python dependencies via Poetry
- Creates project directories: `raw_data/`, `preprocessed_data/`, `results/`, `weights/`, `training_data/`

**Prerequisites:** Poetry must be installed

---

#### `make download-weights`
Download all required model weights and embeddings.

```bash
make download-weights
```

**Downloads:**
- `echo_prime_encoder.pt` (~138MB) - Main video encoder
- `view_classifier.ckpt` (~350MB) - View classification model  
- `candidates_data.zip` (~500MB) - Knowledge base for report generation
- `candidate_embeddings_p1.pt` & `candidate_embeddings_p2.pt` - Pre-computed embeddings

**Total size:** ~1GB

---

### Inference Commands

#### `make inference`
Run EchoPrime analysis on all studies in the `raw_data/` directory.

```bash
make inference
```

**What it does:**
1. Cleans previous results and preprocessed data
2. Processes all DICOM files in `raw_data/`
3. Generates clinical reports and predictions
4. Saves results to `results/inference_output/`

**Equivalent to:**
```bash
poetry run python -m inference.inference --data_dir ./raw_data --output ./results/inference_output
```

---

#### `make visualize`
Create comprehensive visualizations of inference results.

```bash
make visualize
```

**What it does:**
- Generates charts and analysis from `results/inference_output/`
- Creates visualizations in `results/visualization_output/`
- Includes quality comparisons, view distributions, and metric summaries

**Equivalent to:**
```bash
poetry run python scripts/visualize_results.py --results_dir ./results/inference_output --output_dir ./results/visualization_output
```

---

### Training Commands

#### `make train`
Run EchoPrime model training.

```bash
make train
```

**What it does:**
- Runs the fine-tuning pipeline on data in `training_data/`
- Uses configuration from training scripts

**Equivalent to:**
```bash
poetry run python -m training.echoprime_finetune train
```

---

### Docker Commands

#### `make build-docker`
Build the standard Docker image.

```bash
make build-docker
```

**Creates:** `echoprime` Docker image

---

#### `make build-jupyter`
Build Docker image optimized for Jupyter notebooks.

```bash
make build-jupyter
```

**Creates:** `echoprime-jupyter` Docker image

---

#### `make run-docker`
Run EchoPrime in a Docker container.

```bash
make run-docker
```

**Options:**
```bash
# Custom port
make run-docker JUPYTER_PORT=8889
```

---

#### `make run-jupyter`
Run Jupyter notebook in Docker with GPU support.

```bash
make run-jupyter
```

**Features:**
- GPU support with `--gpus all`
- Volume mounting for data persistence
- Interactive mode

---

### Utility Commands

#### `make jupyter`
Start Jupyter notebook server locally.

```bash
make jupyter
```

**Options:**
```bash
# Custom port
make jupyter JUPYTER_PORT=8889
```

**Default:** Runs on port 8888

---

#### `make clean`
Clean up temporary files and directories.

```bash
make clean
```

**Removes:**
- Python cache files (`__pycache__/`, `*.pyc`)
- Jupyter checkpoints
- Build artifacts
- Contents of `results/` and `preprocessed_data/`

---

#### `make help`
Display all available Makefile commands with descriptions.

```bash
make help
```

---

## 🐍 Python Module Commands

### Inference Module

#### Basic Inference
```bash
poetry run python -m inference.inference [OPTIONS]
```

**Required Arguments:**
- `--data_dir PATH` - Directory containing DICOM files or study folders
- `--output PATH` - Output directory for results

**Optional Arguments:**
- `--device {cuda,cpu}` - Processing device (default: auto-detect)
- `--min_quality FLOAT` - Minimum quality threshold for filtering (default: varies by metric)
- `--batch_size INT` - Batch size for processing (default: 4)
- `--num_workers INT` - Number of data loading workers (default: 4)

**Examples:**
```bash
# Basic usage
poetry run python -m inference.inference --data_dir ./raw_data --output ./results

# Force CPU processing
poetry run python -m inference.inference --data_dir ./raw_data --output ./results --device cpu

# Custom quality threshold
poetry run python -m inference.inference --data_dir ./raw_data --output ./results --min_quality 0.6

# Single study processing
poetry run python -m inference.inference --data_dir ./raw_data/patient_001 --output ./results/single_study
```

---

#### Per-Patient Inference
```bash
poetry run python -m inference.inference_per_patient [OPTIONS]
```

**Arguments:**
- `--data_dir PATH` - Directory containing patient study folders
- `--output PATH` - Output directory for results
- `--patient_id STR` - Specific patient ID to process (optional)

**Example:**
```bash
# Process all patients
poetry run python -m inference.inference_per_patient --data_dir ./raw_data --output ./results/per_patient

# Process specific patient
poetry run python -m inference.inference_per_patient --data_dir ./raw_data --output ./results --patient_id patient_001
```

---

### Training Module

#### Fine-tuning
```bash
poetry run python -m training.echoprime_finetune [COMMAND] [OPTIONS]
```

**Commands:**
- `train` - Start training process
- `eval` - Evaluate trained model
- `prepare` - Prepare training data

**Training Options:**
- `--config PATH` - Configuration file path
- `--data_dir PATH` - Training data directory
- `--output_dir PATH` - Output directory for model checkpoints
- `--batch_size INT` - Training batch size
- `--learning_rate FLOAT` - Learning rate
- `--num_epochs INT` - Number of training epochs
- `--mode {full,lora,linear_probe}` - Fine-tuning mode

**Examples:**
```bash
# Basic training
poetry run python -m training.echoprime_finetune train --data_dir ./training_data --output_dir ./trained_models

# LoRA fine-tuning
poetry run python -m training.echoprime_finetune train --mode lora --learning_rate 1e-3

# Custom configuration
poetry run python -m training.echoprime_finetune train --config ./my_config.yaml
```

---

### Preprocessing Scripts

#### DICOM Unpacking
```bash
poetry run python scripts/unpack_dicom_files.py [OPTIONS]
```

**Arguments:**
- `--input_dir PATH` - Directory containing DICOM files
- `--output_dir PATH` - Output directory for extracted videos
- `--format {avi,mp4}` - Output video format (default: avi)

**Example:**
```bash
poetry run python scripts/unpack_dicom_files.py --input_dir ./raw_dicom --output_dir ./extracted_videos
```

---

#### EchoPrime DICOM Unpacking
```bash
poetry run python scripts/unpack_echoprime_dicom.py [OPTIONS]
```

**Arguments:**
- `--input_dir PATH` - Directory containing DICOM files
- `--output_dir PATH` - Output directory
- `--study_name STR` - Name for the study (optional)

---

#### Visualization
```bash
poetry run python scripts/visualize_results.py [OPTIONS]
```

**Arguments:**
- `--results_dir PATH` - Directory containing inference results
- `--output_dir PATH` - Output directory for visualizations
- `--format {png,pdf,svg}` - Output format for plots (default: png)

**Example:**
```bash
poetry run python scripts/visualize_results.py --results_dir ./results/inference_output --output_dir ./visualizations --format pdf
```

---

## 🔧 Configuration Options

### Environment Variables

#### `CUDA_VISIBLE_DEVICES`
Control which GPUs are available to EchoPrime.

```bash
# Use only GPU 0
CUDA_VISIBLE_DEVICES=0 make inference

# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 make inference

# Force CPU only
CUDA_VISIBLE_DEVICES="" make inference
```

#### `PYTORCH_CUDA_ALLOC_CONF`
Configure CUDA memory allocation.

```bash
# Reduce memory fragmentation
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 make inference
```

---

### Poetry Commands

#### Environment Management
```bash
# Install dependencies
poetry install

# Update dependencies
poetry update

# Add new dependency
poetry add package_name

# Remove dependency
poetry remove package_name

# Show installed packages
poetry show

# Activate virtual environment
poetry shell
```

#### Running Commands
```bash
# Run any command in Poetry environment
poetry run command_name

# Run Python script
poetry run python script.py

# Run with specific Python version
poetry env use python3.9
```

---

## 📊 Output Formats

### JSON Output Structure

#### Summary JSON
```json
{
  "total_studies": 3,
  "successful_studies": 3,
  "failed_studies": 0,
  "total_dicom_files": 12,
  "processing_time_seconds": 45.2,
  "studies": [
    {
      "study_name": "patient_001_study_001",
      "dicom_files": 4,
      "status": "success",
      "processing_time": 15.1
    }
  ]
}
```

#### Metrics Predictions JSON
```json
{
  "ejection_fraction": 62.5,
  "left_ventricle_size": "normal",
  "binary_predictions": {
    "aortic_stenosis": 0.05,
    "mitral_regurgitation": 0.12,
    "pacemaker_present": 0.02
  },
  "quality_metrics": {
    "overall_study_quality": 0.78,
    "usable_views": 4,
    "total_views": 4
  },
  "view_distribution": {
    "A4C": 1,
    "A2C": 1,
    "Parasternal_Long": 1,
    "Subcostal": 1
  }
}
```

---

## 🚨 Error Handling

### Common Exit Codes

- **0**: Success
- **1**: General error
- **2**: Invalid arguments
- **3**: File not found
- **4**: CUDA/GPU error
- **5**: Model loading error
- **6**: Insufficient memory

### Error Messages

#### "CUDA out of memory"
```bash
# Solution 1: Use CPU
poetry run python -m inference.inference --device cpu

# Solution 2: Reduce batch size
poetry run python -m inference.inference --batch_size 1
```

#### "Model weights not found"
```bash
# Solution: Download weights
make download-weights
```

#### "No DICOM files found"
```bash
# Check directory structure
ls -la raw_data/
```

---

## 🔍 Debug and Verbose Options

### Verbose Output
```bash
# Enable verbose logging
poetry run python -m inference.inference --verbose --data_dir ./raw_data --output ./results

# Debug mode
poetry run python -m inference.inference --debug --data_dir ./raw_data --output ./results
```

### Logging Configuration
```bash
# Set log level
export ECHOPRIME_LOG_LEVEL=DEBUG
make inference

# Log to file
poetry run python -m inference.inference --log_file ./inference.log --data_dir ./raw_data --output ./results
```

---

## 📈 Performance Tuning

### Memory Optimization
```bash
# Reduce memory usage
poetry run python -m inference.inference --batch_size 1 --num_workers 1

# Enable memory efficient mode
poetry run python -m inference.inference --memory_efficient
```

### Speed Optimization
```bash
# Increase batch size (if memory allows)
poetry run python -m inference.inference --batch_size 8

# Use more workers
poetry run python -m inference.inference --num_workers 8

# Enable mixed precision
poetry run python -m inference.inference --mixed_precision
```

---

## 🔗 Integration Examples

### Bash Script Integration
```bash
#!/bin/bash
# Process multiple directories
for dir in /path/to/studies/*/; do
    echo "Processing $dir"
    cd /path/to/echoprime
    poetry run python -m inference.inference --data_dir "$dir" --output "./results/$(basename "$dir")"
done
```

### Python API Usage
```python
from inference.inference import EchoPrimeInference

# Initialize model
model = EchoPrimeInference(device='cuda')

# Process single study
results = model.process_study('/path/to/study')

# Generate report
report = model.generate_report(results)
```

---

This command reference provides comprehensive coverage of all EchoPrime functionality. For specific use cases or advanced configurations, refer to the individual guide documents.
