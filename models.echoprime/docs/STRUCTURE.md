# Project Structure

```
models.echoprime/
├── Project files (Dockerfile, README.md, pyproject.toml, etc.)
├── data/                        # Sample data organized
│   ├── __init__.py
│   └── example_study/           # Moved from model_data/
├── inference/                   # Inference scripts and demos
│   ├── __init__.py
│   ├── EchoPrimeDemo.ipynb     # Moved from root
│   └── ViewClassificationDemo.ipynb # Moved from root
├── preprocessors/               # NEW: Extracted from video_utils.py
│   ├── __init__.py
│   ├── image_scaling.py         # crop_and_scale, apply_zoom functions
│   └── ultrasound_masking.py    # mask_outside_ultrasound, downsample_and_crop
├── results/                     # NEW: For outputs and debug results
│   └── __init__.py
├── tools/                       # Utility scripts
│   ├── __init__.py
│   ├── report_processing.py     # Renamed from utils.py
│   └── video_io.py             # Video I/O functions from video_utils.py
├── training/                    # NEW: For training-related files
│   └── __init__.py
└── weights/                     # Model weights and candidate data
    ├── __init__.py
    ├── echo_prime_encoder.pt    # Moved from model_data/weights/
    ├── view_classifier.ckpt     # Moved from model_data/weights/
    └── candidates_data/         # Moved from model_data/
```

## Preprocessors and Tools
- `crop_and_scale()` → `preprocessors/image_scaling.py`
- `apply_zoom()` → `preprocessors/image_scaling.py`
- `mask_outside_ultrasound()` → `preprocessors/ultrasound_masking.py`
- `downsample_and_crop()` → `preprocessors/ultrasound_masking.py`
- `read_video()` → `tools/video_io.py`
- `write_video()` → `tools/video_io.py`
- `write_to_avi()` → `tools/video_io.py`
- `write_image()` → `tools/video_io.py`
- Color conversion functions → `tools/video_io.py`

## Utilities
- All report processing functions → `tools/report_processing.py`

This restructuring creates a clean, maintainable codebase that follows the same organizational principles as the successful echoquality project.

## Inference Capability

```bash
make inference
```

This command:
- Automatically discovers all folders in the `data/` directory
- Processes each folder as a separate device/study
- Generates EchoPrime reports and metrics for each folder
- Provides a comprehensive summary of all processed folders
- Saves individual results for each folder in `results/inference_output/`

### Usage Example
```bash
# Place your device folders in data/
data/
├── device1_study/
│   └── *.dcm files
├── device2_study/
│   └── *.dcm files
└── device3_study/
    └── *.dcm files

# Run inference on all folders
make inference

# Results will be saved to:
results/inference_output/
├── device1_study/
│   └── results.json
├── device2_study/
│   └── results.json
├── device3_study/
│   └── results.json
└── summary.json
```

The inference script provides:
- **Individual folder results**: Generated reports, predicted metrics, and processing status
- **Summary statistics**: Total folders processed, success/failure counts, total videos processed
- **Error handling**: Detailed error messages for failed folders
- **Progress tracking**: Real-time progress updates during processing
