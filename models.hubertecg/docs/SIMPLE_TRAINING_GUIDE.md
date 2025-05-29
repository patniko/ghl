# Simple Training Guide for HuBERT-ECG

This guide shows you how to easily train a HuBERT-ECG model with your own ECG data using the new simplified training workflow.

## Quick Start

### 1. Prepare Your ECG Data

First, create the training data directory and add your ECG files:

```bash
# Create the directory structure
mkdir -p training_data/raw

# Copy your ECG files (.npy format) to the raw directory
cp your_ecg_files/*.npy training_data/raw/
```

**ECG File Format Requirements:**
- Files must be in `.npy` format (NumPy arrays)
- Shape: `(12, n_samples)` where 12 represents the 12 ECG leads
- Sampling rate: 500 Hz (recommended)
- Minimum duration: 5 seconds (2500 samples)
- Data type: float32 or float64

### 2. Prepare Training Data

Run the data preparation script:

```bash
make prepare-data
```

This will:
- Process your ECG files and validate their format
- Create train/validation splits
- Generate CSV files with metadata and default labels
- Create a training configuration file

### 3. Edit Labels (Important!)

After data preparation, you need to edit the labels in the generated CSV files:

```bash
# Edit the training labels
nano training_data/metadata/train_dataset.csv
nano training_data/metadata/val_dataset.csv
```

The CSV files will have these columns:
- `filename`: Name of the processed ECG file
- `original_path`: Path to the original file
- `shape`: Shape of the ECG data
- `duration_seconds`: Duration in seconds
- `normal`: Binary label (0 or 1) - edit this!
- `abnormal`: Binary label (0 or 1) - edit this!
- `arrhythmia`: Binary label (0 or 1) - edit this!

**Replace the random default labels with your actual labels!**

### 4. Train the Model

Once you've edited the labels, start training:

```bash
# Train with pretrained weights (recommended)
make train

# Or train from scratch (random initialization)
make train-from-scratch
```

### 5. Monitor Training

Training progress will be displayed in the terminal. Checkpoints are saved to:
```
training_data/checkpoints/simple_training/
```

## Advanced Usage

### Custom Training Configuration

You can modify the training configuration by editing:
```
training_data/metadata/training_config.json
```

Available parameters:
- `vocab_size`: Number of labels/classes
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `patience`: Early stopping patience
- `task`: "multi_label", "multi_class", or "regression"
- `target_metric`: Metric to optimize ("f1_score", "accuracy", etc.)

### Using Pretrained Models

To use a pretrained model from HuggingFace:

```bash
# Download a pretrained model first
# Then specify the path when training
poetry run python scripts/simple_train.py --pretrained_model path/to/model.pt
```

### Custom Labels

You can add your own label columns to the CSV files. Just make sure to:
1. Update `vocab_size` in the config to match the number of labels
2. Update `label_start_index` if needed (default is 4)

## File Structure

After running the training workflow, your directory will look like:

```
training_data/
├── raw/                          # Your original ECG files
│   ├── ecg_001.npy
│   ├── ecg_002.npy
│   └── ...
├── processed/                    # Processed ECG files
│   ├── ecg_001.npy
│   ├── ecg_002.npy
│   └── ...
├── metadata/                     # Training metadata and configs
│   ├── training_config.json
│   ├── training_dataset.csv
│   ├── train_dataset.csv
│   └── val_dataset.csv
└── checkpoints/                  # Saved model checkpoints
    └── simple_training/
        └── hubert_1_iteration_*.pt
```

## Troubleshooting

### Common Issues

1. **"No ECG files found"**
   - Make sure your files are in `.npy` format
   - Check that files are in `training_data/raw/`

2. **"Wrong number of leads"**
   - ECG files must have exactly 12 leads
   - Shape should be `(12, n_samples)`

3. **"ECG too short"**
   - Minimum duration is 5 seconds (2500 samples at 500Hz)
   - Files will be automatically padded if too short

4. **Training fails**
   - Check that you've edited the labels in the CSV files
   - Ensure you have enough GPU memory (reduce batch_size if needed)

### Getting Help

- Check the main README.md for general setup instructions
- Look at the existing training scripts in `training/` for advanced options
- Review the original paper for model details

## Example Workflow

Here's a complete example:

```bash
# 1. Setup environment
make setup

# 2. Add your ECG files
mkdir -p training_data/raw
cp /path/to/your/ecgs/*.npy training_data/raw/

# 3. Prepare data
make prepare-data

# 4. Edit labels (replace random labels with real ones)
nano training_data/metadata/train_dataset.csv
nano training_data/metadata/val_dataset.csv

# 5. Train the model
make train

# 6. Check results
ls training_data/checkpoints/simple_training/
```

That's it! You now have a trained HuBERT-ECG model ready for inference.
