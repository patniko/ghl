# Echo Quality Assessment Tools

This repository contains tools for assessing the quality of echocardiogram (echo) DICOM files and debugging image scaling issues.

## Overview

EchoQuality uses a pre-trained R(2+1)D model to classify the quality of echocardiogram videos. The model analyzes DICOM files and predicts whether the video quality is acceptable, helping to filter out low-quality studies that might lead to inaccurate diagnoses.

## Project Structure

The project is organized into the following directories:

- `preprocessed_data/`: Contains study data organized by device type
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

## Quick Start

### Running Quality Assessment

The easiest way to run quality assessment is using the Makefile:

```bash
# Run inference on all DICOM files in raw_data/ directory
make inference
```

This command:
- Processes DICOM files from `raw_data/` directory
- Extracts processed images to `preprocessed_data/` directory  
- Saves analysis results to `results/inference_output/`

### Directory Structure

**Important**: Each folder in `raw/` represents one patient using one device in one study.

```
raw/                                    # Input: Raw DICOM files
├── patient_001_device_A_study_001/     # One patient, one device, one study
│   ├── view1.dcm
│   ├── view2.dcm
│   └── view3.dcm
├── patient_002_device_B_study_001/     # Different patient/device/study
│   ├── apical_4ch.dcm
│   └── parasternal_long.dcm
└── patient_003_device_A_study_002/
    └── ...

data/                                   # Output: Extracted images
├── patient_001_device_A_study_001/
│   ├── view1_frame_00.png
│   ├── view1_frame_01.png
│   └── ...
├── patient_002_device_B_study_001/
│   ├── apical_4ch_frame_00.png
│   └── ...
└── patient_003_device_A_study_002/
    └── ...

results/inference_output/               # Analysis results
├── summary.json
├── patient_001_device_A_study_001/
│   ├── folder_summary.json
│   ├── inference_results.json
│   └── *.png (charts)
├── patient_002_device_B_study_001/
│   └── ...
└── patient_003_device_A_study_002/
    └── ...
```

## Documentation

- **[Inference Pipeline Guide](docs/inference_pipeline.md)**: Comprehensive documentation of the `make inference` command
- **[Quick Reference](docs/quick_reference.md)**: Quick reference for common tasks

## EchoPrime Quality Control (Legacy)

The legacy tool `inference/EchoPrime_qc.py` is still available for backward compatibility:

```bash
# Process a specific folder of DICOM files
python -m inference.EchoPrime_qc --folders ./path/to/dicom/folder

# Process all device folders in data/
python -m inference.EchoPrime_qc --study_data
```

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
