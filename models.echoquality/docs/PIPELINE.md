# EchoQuality Inference Pipeline Documentation

## Overview

The `make inference` command provides a comprehensive pipeline for running quality assessment on echocardiogram DICOM files. This pipeline processes raw DICOM files, extracts and saves processed images, runs AI-based quality assessment, and generates detailed reports with error tracking.

## Command Usage

```bash
make inference
```

This command executes:
```bash
poetry run python -m inference.inference --data_dir ./raw_data --output ./results/inference_output
```

## Directory Structure and Data Flow

### Input Directory: `raw_data/`
- **Purpose**: Contains raw DICOM files organized by patient/device/study folders
- **Organization**: Each folder represents **one patient using one device in one study**
- **Structure**: 
  ```
  raw_data/
  ├── patient_001_device_A_study_20240101/
  │   ├── view1.dcm
  │   ├── view2.dcm
  │   ├── view3.dcm
  │   └── ...
  ├── patient_002_device_B_study_20240102/
  │   ├── apical_4ch.dcm
  │   ├── parasternal_long.dcm
  │   └── ...
  └── patient_003_device_A_study_20240103/
      └── ...
  ```
- **File Types**: DICOM files (.dcm) containing echocardiogram video data
- **Important**: All DICOM files within a single folder must be from the same patient, using the same device, during the same study session

### Output Directory: `preprocessed_data/`
- **Purpose**: Stores extracted and processed images from DICOM files
- **Structure**:
  ```
  preprocessed_data/
  ├── device_1/
  │   ├── study1_frame_00.png
  │   ├── study1_frame_01.png
  │   ├── study1_frame_02.png
  │   ├── study1_frame_03.png
  │   ├── study1_frame_04.png
  │   ├── study2_frame_00.png
  │   └── ...
  └── device_2/
      └── ...
  ```
- **Content**: PNG images extracted from DICOM files (up to 5 sample frames per study)

### Results Directory: `results/inference_output/`
- **Purpose**: Contains analysis results, visualizations, and error logs
- **Structure**:
  ```
  results/inference_output/
  ├── summary.json                    # Overall summary across all devices
  ├── device_1/
  │   ├── folder_summary.json         # Device-specific summary
  │   ├── inference_results.json      # Detailed per-file results
  │   ├── score_distribution.png      # Quality score histogram
  │   ├── pass_fail_distribution.png  # Pass/fail pie chart
  │   ├── mask_images/                # Debug mask images (if enabled)
  │   │   ├── original/               # Raw frames before processing
  │   │   ├── before/                 # Frames after color conversion
  │   │   └── after/                  # Frames after masking
  │   ├── failed_files/               # Error tracking
  │   │   └── device_1_failed_files.json
  │   └── gradcam/                    # GradCAM visualizations (if enabled)
  │       └── *.png
  └── device_2/
      └── ...
  ```

## Processing Pipeline

### 1. **File Discovery**
- Recursively scans the `raw/` directory for all files
- Identifies potential DICOM files for processing
- Organizes files by device/folder structure

### 2. **DICOM Processing**
For each DICOM file, the pipeline:

#### a. **Validation**
- Attempts to read the file as a DICOM
- Validates pixel array accessibility
- Checks image dimensions (must be 3D+ video data)
- Tracks validation errors by type

#### b. **Preprocessing**
- Converts single-channel to 3-channel if needed
- Applies ultrasound region masking to remove artifacts
- Crops and scales frames to 112x112 pixels
- Normalizes pixel values

#### c. **Image Extraction**
- Extracts up to 5 sample frames per study
- Saves frames as PNG images to `preprocessed_data/{device_name}/`
- Converts to uint8 format for storage

#### d. **Quality Assessment**
- Runs frames through the trained R2+1D CNN model
- Generates quality probability scores (0-1)
- Applies threshold (default: 0.3) for pass/fail classification
- Provides quality assessment descriptions

### 3. **Error Tracking**
The pipeline tracks six types of errors:

| Error Type | Description |
|------------|-------------|
| `not_dicom` | File cannot be read as DICOM |
| `empty_pixel_array` | DICOM has no pixel data |
| `invalid_dimensions` | Wrong image dimensions (not video) |
| `masking_error` | Error during ultrasound masking |
| `scaling_error` | Error during frame scaling/cropping |
| `other_errors` | Unexpected processing errors |

### 4. **Results Generation**

#### a. **Per-Device Results**
- **folder_summary.json**: Device statistics and metadata
- **inference_results.json**: Detailed per-file quality scores
- **Visualizations**: Score distributions and pass/fail charts

#### b. **Error Logs**
- **failed_files.json**: Detailed error information per device
- Lists failed files with specific error reasons
- Includes frame-level error tracking

#### c. **Debug Images** (Optional)
- **mask_images/**: Before/after masking visualizations
- **gradcam/**: Model attention visualizations

#### d. **Overall Summary**
- **summary.json**: Aggregated statistics across all devices
- Total files processed, pass rates, error summaries

## Quality Assessment Scoring

### Score Interpretation
| Score Range | Quality Level | Description |
|-------------|---------------|-------------|
| 0.8 - 1.0 | Excellent | High-quality images suitable for analysis |
| 0.6 - 0.8 | Good | Acceptable quality with minor issues |
| 0.3 - 0.6 | Acceptable | Usable but may have quality concerns |
| 0.2 - 0.3 | Poor | Significant quality issues |
| 0.1 - 0.2 | Very Poor | Major acquisition problems |
| 0.0 - 0.1 | Critical | Severe technical issues |

### Pass/Fail Threshold
- **Default Threshold**: 0.3
- **Pass**: Score ≥ 0.3 (acceptable quality or better)
- **Fail**: Score < 0.3 (poor quality)

## Command Line Options

The inference pipeline supports several command-line arguments:

```bash
python -m inference.inference [OPTIONS]
```

### Available Options
- `--data_dir`: Input directory containing DICOM files (default: "data")
- `--model`: Path to model weights (default: "weights/video_quality_model.pt")
- `--output`: Output directory for results (default: "results/inference_output")
- `--threshold`: Quality threshold for pass/fail (default: 0.3)
- `--gradcam`: Enable GradCAM visualizations
- `--device`: Computation device (auto/cpu/cuda, default: "auto")

### Examples

```bash
# Basic inference with default settings
make inference

# Custom threshold and enable GradCAM
poetry run python -m inference.inference --threshold 0.5 --gradcam

# Process different input directory
poetry run python -m inference.inference --data_dir ./custom_raw --output ./custom_results
```

## Output Files Reference

### summary.json
```json
{
  "total_folders": 3,
  "successful_folders": 3,
  "failed_folders": 0,
  "total_files": 150,
  "total_processed": 145,
  "total_pass": 120,
  "total_fail": 25,
  "overall_pass_rate": 82.8,
  "folder_results": [...]
}
```

### folder_summary.json (per device)
```json
{
  "folder": "device_1",
  "status": "success",
  "num_files": 50,
  "num_processed": 48,
  "pass_count": 40,
  "fail_count": 8,
  "pass_rate": 83.3,
  "results": {...},
  "error_stats": {...}
}
```

### inference_results.json (per device)
```json
{
  "study1.dcm": {
    "score": 0.85,
    "status": "PASS",
    "assessment": "Excellent quality",
    "path": "/path/to/study1.dcm"
  },
  "study2.dcm": {
    "score": 0.15,
    "status": "FAIL",
    "assessment": "Very poor quality - significant issues",
    "path": "/path/to/study2.dcm"
  }
}
```

## Performance Considerations

### Processing Speed
- **GPU Acceleration**: Automatically uses CUDA if available
- **Batch Processing**: Processes files sequentially per device
- **Memory Management**: Efficient tensor operations with cleanup

### Storage Requirements
- **Input**: Original DICOM files (varies by study)
- **Extracted Images**: ~5 PNG files per study (~500KB each)
- **Results**: JSON files and visualizations (~1-10MB per device)
- **Debug Images**: Optional mask images (~2MB per study if enabled)

## Troubleshooting

### Common Issues

1. **No DICOM files found**
   - Check that `raw/` directory exists and contains .dcm files
   - Verify file permissions

2. **Model loading errors**
   - Ensure `weights/video_quality_model.pt` exists
   - Check model file integrity

3. **Memory errors**
   - Reduce batch size or use CPU instead of GPU
   - Ensure sufficient system memory

4. **Permission errors**
   - Verify write permissions for `data/` and `results/` directories

### Error Logs
Check the following for debugging:
- Console output during processing
- `failed_files.json` for specific file errors
- `folder_summary.json` for device-level statistics

## Integration with Other Tools

### Preprocessing
- Can be used after DICOM acquisition and organization
- Compatible with standard DICOM viewers and tools

### Post-processing
- Results can be imported into analysis pipelines
- JSON format enables easy integration with other tools
- Extracted images can be used for manual review or additional processing

## Best Practices

1. **Data Organization**: Organize raw DICOM files by device/study for clear results
2. **Regular Cleanup**: Monitor `data/` and `results/` directory sizes
3. **Quality Thresholds**: Adjust threshold based on specific use case requirements
4. **Error Review**: Regularly review failed files to identify systematic issues
5. **Backup**: Maintain backups of original DICOM files before processing
