# Echo Quality Assessment Tools

This repository contains tools for assessing the quality of echocardiogram (echo) DICOM files and debugging image scaling issues.

## Overview

EchoQuality uses a pre-trained R(2+1)D model to classify the quality of echocardiogram videos. The model analyzes DICOM files and predicts whether the video quality is acceptable, helping to filter out low-quality studies that might lead to inaccurate diagnoses.

## Features

- Automatic masking of non-ultrasound regions in echo videos
- Quality assessment with detailed scoring and feedback
- GradCAM visualization to understand model decisions
- Comprehensive training pipeline with data augmentation
- Jupyter notebook for interactive exploration

## EchoPrime Quality Control

The main tool, `EchoPrime_qc.py`, is used to assess the quality of echo DICOM files. It processes DICOM files, applies masking to isolate the ultrasound region, and uses a pre-trained model to evaluate the quality of each file.

### Usage

```bash
# Process a specific folder of DICOM files
python EchoPrime_qc.py --folders ./path/to/dicom/folder

# Process all device folders in model_data/study_data
python EchoPrime_qc.py --study_data

# Disable saving mask images
python EchoPrime_qc.py --no-mask-images
```

### Command Line Arguments

- `--folders`: List of device folders to process
- `--study_data`: Process all device folders in model_data/study_data
- `--no-mask-images`: Disable saving mask images

### Output

The script generates:
- A summary of quality assessment results in the terminal
- A JSON file (`quality_results.json`) with detailed results
- JSON files in `./model_data/failed_files/` with information about failed files
- If enabled, mask images in `./mask_images/` showing the original, before, and after masking

## Debug Tools

The repository includes several debug tools for investigating image scaling issues in DICOM files:

### 1. Basic Debug Tool (`debug_scaling.py`)

Provides detailed visualization and logging of each step in the crop_and_scale process.

```bash
# Run with a specific DICOM file
python debug_scaling.py ./path/to/dicom/file.dcm

# Import and use programmatically
import debug_scaling
debug_scaling.main('./path/to/dicom/file.dcm')
```

Output is saved to `./debug_images/` and `debug_scaling.log`.

### 2. Interactive Debug Tool (`debug_scaling_interactive.py`)

Provides a step-by-step interactive approach to debug the crop_and_scale function with Python debugger (pdb) breakpoints.

```bash
# Run with a specific DICOM file
python debug_scaling_interactive.py ./path/to/dicom/file.dcm

# Import and use programmatically
import debug_scaling_interactive
debug_scaling_interactive.main('./path/to/dicom/file.dcm')
```

Output is saved to `./debug_images_interactive/`.

### 3. Visual Debug Tool (`debug_scaling_visual.py`)

Provides a visual approach with side-by-side comparisons and detailed visualizations of problematic areas.

```bash
# Run with a specific DICOM file
python debug_scaling_visual.py ./path/to/dicom/file.dcm

# Import and use programmatically
import debug_scaling_visual
debug_scaling_visual.main('./path/to/dicom/file.dcm')
```

Output is saved to `./debug_images_visual/` and `debug_scaling_visual.log`.

### 4. Specialized Debug Tool (`debug_scaling_specialized.py`)

Focuses on the specific problem of very narrow images being scaled to square outputs.

```bash
# Run with a specific DICOM file
python debug_scaling_specialized.py ./path/to/dicom/file.dcm

# Import and use programmatically
import debug_scaling_specialized
debug_scaling_specialized.main('./path/to/dicom/file.dcm')
```

Output is saved to `./debug_images_specialized/` and `debug_scaling_specialized.log`.

## Example Workflow

1. Run quality assessment on a folder of DICOM files:
   ```bash
   python EchoPrime_qc.py --folders ./model_data/study_data/epiq7
   ```

2. If issues are detected, use the debug tools to investigate:
   ```bash
   # Start with basic debugging
   python debug_scaling.py ./model_data/study_data/epiq7/problematic_file.dcm
   
   # For more detailed visual analysis
   python debug_scaling_visual.py ./model_data/study_data/epiq7/problematic_file.dcm
   
   # For interactive debugging
   python debug_scaling_interactive.py ./model_data/study_data/epiq7/problematic_file.dcm
   
   # For specialized debugging of narrow images
   python debug_scaling_specialized.py ./model_data/study_data/epiq7/problematic_file.dcm
   ```

3. Review the debug output to identify and fix issues.

## Requirements

- Python 3.6+
- PyTorch
- OpenCV (cv2)
- pydicom
- numpy
- matplotlib
- tqdm

## Model

The quality assessment uses a pre-trained R(2+1)D model located at `./video_quality_model.pt`.
