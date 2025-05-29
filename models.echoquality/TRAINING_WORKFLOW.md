# EchoQuality Training Workflow

This document explains the complete workflow for building a training dataset from inference results and running training for the EchoQuality model.

## Overview

The EchoQuality project now includes tools to automatically create training datasets from your inference results. This allows you to leverage the quality assessments from your inference runs to build robust training datasets with both high and low quality samples.

## Workflow Steps

### 1. Run Inference (if not already done)

First, ensure you have inference results available:

```bash
make inference
```

This will process all DICOM files in `raw_data/` and create quality assessments in `results/inference_output/`.

### 2. Prepare Training Data

Use the automated script to extract high and low quality samples:

```bash
make prepare-data
```

Or run with custom parameters:

```bash
poetry run python prepare_training_data.py --max-high-quality 800 --max-low-quality 300
```

This script will:
- **Clear** the existing `training_data/` folder
- **Analyze** inference results to identify high and low quality samples
- **Copy** selected DICOM files to `training_data/echo_videos/`
- **Create** `training_data/echo_annotations.csv` with quality labels
- **Generate** dataset analysis and statistics

### 3. Run Training

Once the training data is prepared, start training:

```bash
make train
```

This will use the existing training pipeline with your prepared dataset.

## Dataset Structure

After running `make prepare-data`, you'll have:

```
training_data/
├── echo_annotations.csv          # Training annotations
├── dataset_analysis.json         # Dataset statistics
└── echo_videos/                  # DICOM files
    ├── hq_epiq7_0000_*.dcm       # High quality samples
    ├── hq_affiniti70g_0001_*.dcm
    ├── lq_vivids70_0050_*.dcm    # Low quality samples
    └── ...
```

### Annotations Format

The `echo_annotations.csv` file contains:

| Column | Description |
|--------|-------------|
| `video_path` | Path to DICOM file (relative to training_data/) |
| `quality_score` | Binary quality label (1=high, 0=low) |
| `filename` | DICOM filename |
| `machine_type` | Ultrasound machine type |
| `score` | Original inference score (0-1) |
| `status` | Original inference status (PASS/FAIL) |
| `assessment` | Original quality assessment |
| `folder` | Original folder name |
| `file_id` | Original file identifier |

## Quality Criteria

### High Quality Samples (Label = 1)
- **Score**: ≥ 0.95
- **Status**: PASS
- **Assessment**: Contains "Excellent quality"

### Low Quality Samples (Label = 0)
- **Score**: ≤ 0.1
- **Status**: FAIL
- **Assessment**: Contains "Critical issues"

## Dataset Statistics

Based on your inference results, you can expect:

- **Total Available**: 3,440 processed samples
- **High Quality**: 2,463 samples (71.6%)
- **Low Quality**: 525 samples (15.3%)
- **Machine Types**: 7 different ultrasound machines

### Machine Distribution
- **Vivid Series**: 53 folders (VIVIDE95, VIVIDS60, VIVIDS70)
- **Epiq7**: 17 folders
- **Affiniti 70G**: 16 folders
- **Affiniti 50G**: 15 folders

## Customization Options

### Sample Limits
```bash
poetry run python prepare_training_data.py \
    --max-high-quality 800 \
    --max-low-quality 300
```

### Quality Thresholds
```bash
poetry run python prepare_training_data.py \
    --high-quality-threshold 0.90 \
    --low-quality-threshold 0.15
```

## Training Configuration

The training pipeline uses:
- **Model**: R(2+1)D video quality assessment
- **Loss**: Binary Cross Entropy
- **Optimizer**: Adam with learning rate scheduling
- **Data Augmentation**: Brightness, contrast, rotation, translation
- **Validation Split**: Automatic train/validation/test split
- **MLflow Tracking**: Experiment logging and metrics

## File Organization

### Key Files
- `prepare_training_data.py` - Main dataset preparation script
- `analyze_data.py` - Analyze inference results
- `Makefile` - Easy commands for common tasks
- `training/train_quality_model.py` - Training pipeline

### Generated Files
- `training_data/echo_annotations.csv` - Training annotations
- `training_data/dataset_analysis.json` - Dataset statistics
- `training_data/echo_videos/*.dcm` - Copied DICOM files

## Example Workflow

```bash
# 1. Analyze available data
poetry run python analyze_data.py

# 2. Prepare training dataset
make prepare-data

# 3. Review dataset statistics
cat training_data/dataset_analysis.json

# 4. Start training
make train

# 5. Monitor training (in another terminal)
mlflow ui
```

## Expected Results

### Dataset Quality
- **Balanced Classes**: ~67% high quality, ~33% low quality
- **Machine Diversity**: Samples from all ultrasound machine types
- **Quality Range**: High quality (0.95-1.0), Low quality (0.0-0.1)

### Training Benefits
- **Real Data**: Uses actual inference results, not synthetic data
- **Quality Diversity**: Clear distinction between good and bad samples
- **Machine Robustness**: Training across different ultrasound machines
- **Scalable**: Easy to regenerate with different parameters

## Troubleshooting

### Common Issues

1. **No inference results**: Run `make inference` first
2. **Insufficient samples**: Lower quality thresholds or increase limits
3. **Missing DICOM files**: Check that raw_data paths are correct
4. **Training errors**: Verify annotations CSV format

### Performance Tips

1. **Start Small**: Use smaller sample limits for initial testing
2. **Monitor Training**: Use MLflow to track training progress
3. **Adjust Thresholds**: Experiment with different quality criteria
4. **Balance Classes**: Ensure reasonable ratio of high/low quality samples

## Next Steps

After successful training:

1. **Evaluate Model**: Review training metrics and validation performance
2. **Test Inference**: Run inference on new data with trained model
3. **Iterate Dataset**: Adjust quality criteria and retrain if needed
4. **Deploy Model**: Integrate trained model into production pipeline

The automated workflow makes it easy to continuously improve your model by leveraging new inference results and adjusting training parameters based on performance.
