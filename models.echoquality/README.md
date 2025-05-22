# Echo Quality Assessment Tools

This repository contains tools for assessing the quality of echocardiogram (echo) DICOM files and debugging image scaling issues.

## Overview

EchoQuality uses a pre-trained R(2+1)D model to classify the quality of echocardiogram videos. The model analyzes DICOM files and predicts whether the video quality is acceptable, helping to filter out low-quality studies that might lead to inaccurate diagnoses.

## Project Structure

The project is organized into the following directories:

- `data/`: Contains study data organized by device type
- `weights/`: Contains model weights, embeddings, and other model-related data
- `inference/`: Contains code for running inference with the quality assessment model
- `training/`: Contains code and documentation for training the model
- `tools/`: Contains debugging tools for investigating image scaling issues
- `preprocessors/`: Contains code for preprocessing echo videos (masking, scaling, etc.)
- `results/`: Contains output from quality assessment runs and debugging tools

## Features

- Automatic masking of non-ultrasound regions in echo videos
- Quality assessment with detailed scoring and feedback
- GradCAM visualization to understand model decisions
- Comprehensive training pipeline with data augmentation
- Jupyter notebook for interactive exploration

## EchoPrime Quality Control

The main tool, `inference/EchoPrime_qc.py`, is used to assess the quality of echo DICOM files. It processes DICOM files, applies masking to isolate the ultrasound region, and uses a pre-trained model to evaluate the quality of each file.

### Usage

```bash
# Process a specific folder of DICOM files
python -m inference.EchoPrime_qc --folders ./path/to/dicom/folder

# Process all device folders in data/
python -m inference.EchoPrime_qc --study_data

# Disable saving mask images
python -m inference.EchoPrime_qc --no-mask-images
```

### Command Line Arguments

- `--folders`: List of device folders to process
- `--study_data`: Process all device folders in data/
- `--no-mask-images`: Disable saving mask images

### Output

The script generates:
- A summary of quality assessment results in the terminal
- A JSON file (`quality_results.json`) with detailed results
- JSON files in `./results/failed_files/` with information about failed files
- If enabled, mask images in `./results/mask_images/` showing the original, before, and after masking

## Debug Tools

The repository includes several debug tools for investigating image scaling issues in DICOM files:

### 1. Basic Debug Tool (`tools/debug_scaling.py`)

Provides detailed visualization and logging of each step in the crop_and_scale process.

```bash
# Run with a specific DICOM file
python -m tools.debug_scaling ./path/to/dicom/file.dcm

# Import and use programmatically
from tools import debug_scaling
debug_scaling.main('./path/to/dicom/file.dcm')
```

Output is saved to `./results/debug_images/` and `debug_scaling.log`.

### 2. Interactive Debug Tool (`tools/debug_scaling_interactive.py`)

Provides a step-by-step interactive approach to debug the crop_and_scale function with Python debugger (pdb) breakpoints.

```bash
# Run with a specific DICOM file
python -m tools.debug_scaling_interactive ./path/to/dicom/file.dcm

# Import and use programmatically
from tools import debug_scaling_interactive
debug_scaling_interactive.main('./path/to/dicom/file.dcm')
```

Output is saved to `./results/debug_images_interactive/`.

### 3. Visual Debug Tool (`tools/debug_scaling_visual.py`)

Provides a visual approach with side-by-side comparisons and detailed visualizations of problematic areas.

```bash
# Run with a specific DICOM file
python -m tools.debug_scaling_visual ./path/to/dicom/file.dcm

# Import and use programmatically
from tools import debug_scaling_visual
debug_scaling_visual.main('./path/to/dicom/file.dcm')
```

Output is saved to `./results/debug_images_visual/` and `debug_scaling_visual.log`.

### 4. Specialized Debug Tool (`tools/debug_scaling_specialized.py`)

Focuses on the specific problem of very narrow images being scaled to square outputs.

```bash
# Run with a specific DICOM file
python -m tools.debug_scaling_specialized ./path/to/dicom/file.dcm

# Import and use programmatically
from tools import debug_scaling_specialized
debug_scaling_specialized.main('./path/to/dicom/file.dcm')
```

Output is saved to `./results/debug_images_specialized/` and `debug_scaling_specialized.log`.

## Example Workflow

1. Run quality assessment on a folder of DICOM files:
   ```bash
   python -m inference.EchoPrime_qc --folders ./data/epiq7
   ```

2. If issues are detected, use the debug tools to investigate:
   ```bash
   # Start with basic debugging
   python -m tools.debug_scaling ./data/epiq7/problematic_file.dcm
   
   # For more detailed visual analysis
   python -m tools.debug_scaling_visual ./data/epiq7/problematic_file.dcm
   
   # For interactive debugging
   python -m tools.debug_scaling_interactive ./data/epiq7/problematic_file.dcm
   
   # For specialized debugging of narrow images
   python -m tools.debug_scaling_specialized ./data/epiq7/problematic_file.dcm
   ```

3. Review the debug output to identify and fix issues.

## Training

For information on training the model, see [TRAINING.md](training/TRAINING.md).

## Requirements

- Python 3.6+
- PyTorch
- OpenCV (cv2)
- pydicom
- numpy
- matplotlib
- tqdm

## Model

The quality assessment uses a pre-trained R(2+1)D model located at `./weights/video_quality_model.pt`.
