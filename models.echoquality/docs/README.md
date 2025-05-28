# EchoQuality Documentation

Welcome to the EchoQuality documentation. This directory contains comprehensive guides for using the echo quality assessment pipeline.

## Documentation Overview

### 📚 Main Guides

- **[Inference Pipeline Guide](inference_pipeline.md)** - Complete documentation of the `make inference` command
  - Detailed process flow from raw_data DICOM files to analysis results
  - Directory structure and data flow explanation
  - Error tracking and troubleshooting
  - Command-line options and examples

- **[Quick Reference](quick_reference.md)** - Fast reference for common tasks
  - Basic usage commands
  - Directory flow diagram
  - Quality score interpretation
  - Common troubleshooting solutions

## Getting Started

### 1. Basic Usage
```bash
# Run quality assessment on all DICOM files
make inference
```

### 2. Directory Setup
Ensure your directory structure follows this pattern:
```
raw_data/                               # Place your DICOM files here
├── patient_001_device_A_study_001/     # One patient, one device, one study
│   ├── view1.dcm
│   ├── view2.dcm
│   └── view3.dcm
├── patient_002_device_B_study_001/     # Different patient/device/study
│   ├── apical_4ch.dcm
│   └── parasternal_long.dcm
└── patient_003_device_A_study_002/
    └── ...
```

**Important**: Each folder must contain DICOM files from the same patient, using the same device, during the same study session.

### 3. View Results
After running inference, check:
- `preprocessed_data/` - Extracted images from DICOM files
- `results/inference_output/` - Analysis results and reports

## Key Features

### 🔍 Quality Assessment
- AI-powered quality scoring (0-1 scale)
- Pass/fail classification with configurable thresholds
- Detailed quality descriptions and recommendations

### 📊 Comprehensive Reporting
- Overall summary across all devices
- Per-device detailed reports
- Visual charts and distributions
- Error tracking and failed file logs

### 🖼️ Image Processing
- Automatic ultrasound region masking
- Frame extraction and preprocessing
- Debug visualizations for troubleshooting

### 🛠️ Error Handling
- Six categories of error tracking
- Detailed failed file reports
- Frame-level error identification
- Comprehensive logging

## Advanced Usage

### Custom Parameters
```bash
# Custom quality threshold
poetry run python -m inference.inference --threshold 0.5

# Enable debug visualizations
poetry run python -m inference.inference --gradcam

# Process custom directory
poetry run python -m inference.inference --data_dir ./custom_raw
```

### Integration
- JSON output format for easy integration
- Compatible with standard DICOM tools
- Extensible pipeline architecture

## Support

### Troubleshooting
Common issues and solutions are documented in:
- [Inference Pipeline Guide - Troubleshooting Section](inference_pipeline.md#troubleshooting)
- [Quick Reference - Troubleshooting Table](quick_reference.md#troubleshooting)

### File Issues
If you encounter problems with specific files:
1. Check the `failed_files/` directory for error details
2. Review console output for specific error messages
3. Use the debug tools in the `tools/` directory for detailed analysis

## Related Documentation

- **[Main README](../README.md)** - Project overview and setup
- **[Training Documentation](../training/TRAINING.md)** - Model training information
- **[Debug Tools](../README.md#debug-tools)** - Specialized debugging utilities

---

For questions or issues, refer to the troubleshooting sections in the guides above or check the main project README.
