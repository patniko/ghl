# EchoQuality Training Dataset Builder

This guide explains how to build a comprehensive training dataset from your inference results and run training for the EchoQuality model.

## Overview

The training dataset builder extracts high and low quality samples from your inference results, generates synthetic degraded samples, and creates a balanced dataset for training. This approach leverages your existing inference results to create a robust training dataset.

## Files Created

- `build_training_dataset.py` - Main script to build the training dataset
- `run_training.py` - Script to run the training pipeline
- `TRAINING_DATASET_README.md` - This documentation

## Quick Start

### 1. Build the Training Dataset

```bash
cd models.echoquality
python build_training_dataset.py
```

This will:
- Extract ~800 high-quality samples (score ≥ 0.95, status = PASS)
- Extract ~300 low-quality samples (score ≤ 0.1, status = FAIL)
- Generate ~300 synthetic degraded samples
- Create organized directory structure
- Generate comprehensive annotations CSV

### 2. Run Training

```bash
python run_training.py
```

This will start training with the enhanced dataset using the existing training pipeline.

## Detailed Usage

### Building the Dataset

The `build_training_dataset.py` script offers several customization options:

```bash
python build_training_dataset.py \
    --results-dir results/inference_output \
    --raw-data-dir raw_data \
    --output-dir training_data_enhanced \
    --high-quality-threshold 0.95 \
    --low-quality-threshold 0.1 \
    --max-high-quality 800 \
    --max-low-quality 300 \
    --num-synthetic 300
```

**Arguments:**
- `--results-dir`: Directory containing your inference results (default: `results/inference_output`)
- `--raw-data-dir`: Directory containing raw DICOM files (default: `raw_data`)
- `--output-dir`: Output directory for training dataset (default: `training_data_enhanced`)
- `--high-quality-threshold`: Minimum score for high quality samples (default: 0.95)
- `--low-quality-threshold`: Maximum score for low quality samples (default: 0.1)
- `--max-high-quality`: Maximum number of high quality samples (default: 800)
- `--max-low-quality`: Maximum number of low quality samples (default: 300)
- `--num-synthetic`: Number of synthetic samples to generate (default: 300)

### Dataset Structure

The script creates the following directory structure:

```
training_data_enhanced/
├── echo_annotations_enhanced.csv          # Comprehensive annotations
├── high_quality/                          # High quality samples
│   ├── hq_affiniti50g_0000_12345678.dcm
│   ├── hq_affiniti70g_0001_87654321.dcm
│   └── ...
├── low_quality/
│   ├── natural_failures/                  # Naturally low quality samples
│   │   ├── lq_epiq7_0000_11111111.dcm
│   │   └── ...
│   └── synthetic_degraded/                 # Synthetically degraded samples
│       ├── synthetic_affiniti50g_0000.dcm
│       └── ...
└── validation_set/                        # Reserved for future use
```

### Annotations CSV Format

The generated `echo_annotations_enhanced.csv` contains:

| Column | Description |
|--------|-------------|
| video_path | Relative path to DICOM file |
| quality_score | Binary quality label (1=high, 0=low) |
| filename | DICOM filename |
| machine_type | Ultrasound machine type (affiniti50g, affiniti70g, epiq7, vivid) |
| score | Original inference score (0-1) |
| status | Original inference status (PASS/FAIL) |
| assessment | Original quality assessment |
| folder | Original folder name |
| file_id | Original file identifier |
| synthetic | Boolean flag for synthetic samples |
| source_sample | Source file ID for synthetic samples |

### Running Training

The `run_training.py` script provides a convenient wrapper for the training pipeline:

```bash
python run_training.py \
    --annotations-csv training_data_enhanced/echo_annotations_enhanced.csv \
    --model-weights weights/video_quality_model.pt \
    --output-dir trained_models_enhanced \
    --batch-size 8 \
    --epochs 30 \
    --learning-rate 1e-4
```

**Arguments:**
- `--annotations-csv`: Path to annotations CSV (default: `training_data_enhanced/echo_annotations_enhanced.csv`)
- `--model-weights`: Path to pre-trained weights (default: `weights/video_quality_model.pt`)
- `--output-dir`: Directory to save trained models (default: `trained_models_enhanced`)
- `--batch-size`: Training batch size (default: 8)
- `--epochs`: Number of training epochs (default: 30)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--no-mlflow`: Disable MLflow tracking

## Dataset Quality Analysis

Based on your inference results, the dataset will include:

### High Quality Samples (Label = 1)
- **Criteria**: Score ≥ 0.95, Status = PASS, Assessment = "Excellent quality"
- **Expected Count**: ~800 samples
- **Machine Distribution**: Balanced across Affiniti 50G/70G, Epiq7, Vivid series
- **Quality Characteristics**: Clear imaging, proper views, minimal artifacts

### Low Quality Samples (Label = 0)
- **Natural Failures**: Score ≤ 0.1, Status = FAIL, Assessment = "Critical issues"
- **Synthetic Degraded**: Generated from high quality samples with:
  - Noise injection (20-40% level)
  - Blur kernels (5-9 pixels)
  - Contrast reduction (30-70%)
- **Expected Count**: ~600 samples total (300 natural + 300 synthetic)

### Machine Type Distribution
From your inference results, you have data from:
- **Affiniti 50G**: 15 folders
- **Affiniti 70G**: 17 folders  
- **Epiq7**: 16 folders
- **Vivid Series**: 53 folders (VIVIDE95, VIVIDS60, VIVIDS70)

## Synthetic Data Generation

The script generates synthetic low-quality samples by applying degradations to high-quality samples:

### Degradation Types
1. **Noise Injection**: Gaussian noise (σ = 0.2-0.4)
2. **Blur**: Motion blur with kernels 5-9 pixels
3. **Contrast Reduction**: 30-70% contrast reduction
4. **Combined Effects**: Random combinations of above

### Benefits
- **Data Augmentation**: Increases dataset size
- **Balanced Classes**: Ensures sufficient negative examples
- **Controlled Degradation**: Known failure modes for better training
- **Machine Diversity**: Synthetic samples from all machine types

## Training Configuration

### Recommended Settings
- **Batch Size**: 8-16 (depending on GPU memory)
- **Learning Rate**: 1e-4 with ReduceLROnPlateau scheduler
- **Epochs**: 30-50 with early stopping
- **Model**: Fine-tune existing R(2+1)D model
- **Loss Function**: Binary Cross Entropy with Logits
- **Optimizer**: Adam with weight decay

### Data Splits
The training script automatically splits data:
- **Training**: 70% of samples
- **Validation**: 15% of samples  
- **Test**: 15% of samples

### Augmentation
Training uses data augmentation:
- Brightness/contrast variations (±10%)
- Rotation (±5 degrees)
- Translation (±5%)
- Zoom (±5%)
- Temporal masking (5% probability)

## Expected Results

### Dataset Statistics
- **Total Samples**: ~1,400 samples
- **High Quality**: ~800 samples (57%)
- **Low Quality**: ~600 samples (43%)
- **Machine Coverage**: All 4 ultrasound machine types
- **Quality Distribution**: Balanced for robust training

### Training Improvements
- **Better Generalization**: Diverse machine types and quality levels
- **Reduced Overfitting**: Synthetic augmentation and balanced classes
- **Improved Robustness**: Exposure to various failure modes
- **Enhanced Performance**: More training data from existing results

## Troubleshooting

### Common Issues

1. **File Not Found Errors**
   - Ensure `results/inference_output/summary.json` exists
   - Check that raw DICOM files are accessible
   - Verify file paths in inference results

2. **Memory Issues**
   - Reduce batch size in training
   - Limit number of synthetic samples
   - Process dataset in smaller chunks

3. **DICOM Reading Errors**
   - Some DICOM files may be corrupted
   - Script will skip problematic files and continue
   - Check error messages for specific issues

4. **Insufficient Samples**
   - Lower quality thresholds to include more samples
   - Increase synthetic sample generation
   - Check inference results for data availability

### Performance Tips

1. **GPU Usage**: Ensure CUDA is available for training
2. **Storage**: Use SSD for faster DICOM file access
3. **Memory**: Monitor RAM usage during dataset building
4. **Parallel Processing**: Consider multiprocessing for large datasets

## Next Steps

After building the dataset and running training:

1. **Evaluate Results**: Check training metrics and validation performance
2. **Model Analysis**: Use GradCAM to understand model focus areas
3. **Hyperparameter Tuning**: Experiment with different learning rates and architectures
4. **Dataset Expansion**: Add more samples or different quality criteria
5. **Deployment**: Integrate trained model into inference pipeline

## Support

For issues or questions:
1. Check the training logs for error messages
2. Verify dataset statistics in the generated CSV
3. Review MLflow experiments for training metrics
4. Examine sample files to ensure quality

The enhanced training dataset provides a solid foundation for improving your EchoQuality model with real-world data from your inference results.
