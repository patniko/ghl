# DICOM to NPY Conversion Script

This script converts DICOM ECG files to NPY format compatible with the HuBERT-ECG model.

## Overview

The `dicom_to_npy.py` script processes DICOM files containing ECG waveform data and converts them to the standardized NPY format expected by HuBERT-ECG:
- **Shape**: (12, 5000) - 12 ECG leads with 5000 samples each
- **Data type**: float64
- **Value range**: Normalized to approximately -1 to 1

## Features

- **Automatic ECG extraction**: Reads ECG waveform data from DICOM WaveformSequence
- **Flexible input handling**: Handles different sampling rates and lead counts
- **Intelligent resampling**: Automatically resamples data to 5000 samples
- **Lead normalization**: Handles cases with fewer or more than 12 leads
- **Data normalization**: Normalizes amplitude to match PTB-XL dataset format
- **Batch processing**: Processes all DICOM files in a directory

## Usage

### Using the Makefile (Recommended)

```bash
make dicom-setup
```

This will convert all DICOM files in `data/12L/` and save NPY files to `data/12L/processed/`.

### Using the script directly

```bash
# Convert files from default directories
poetry run python scripts/dicom_to_npy.py

# Specify custom input and output directories
poetry run python scripts/dicom_to_npy.py --input_dir /path/to/dicom/files --output_dir /path/to/output
```

### Command line options

- `--input_dir`: Directory containing DICOM files (default: `data/12L`)
- `--output_dir`: Directory to save NPY files (default: `data/12L/processed`)

## Input Requirements

The script expects DICOM files with:
- **Modality**: ECG
- **WaveformSequence**: Contains the ECG waveform data
- **Standard DICOM ECG format**: Compatible with Philips and other major manufacturers

## Output Format

Each DICOM file is converted to a corresponding NPY file with:
- **Filename**: `{original_name}.npy` (e.g., `93C7C5AD.dcm` → `93C7C5AD.npy`)
- **Shape**: (12, 5000)
- **Data type**: float64
- **Normalization**: DC offset removed, amplitude normalized

## Example

```python
import numpy as np

# Load converted ECG data
ecg_data = np.load('data/12L/processed/93C7C5AD.npy')
print(f"Shape: {ecg_data.shape}")  # (12, 5000)
print(f"Data type: {ecg_data.dtype}")  # float64
print(f"Value range: {ecg_data.min():.3f} to {ecg_data.max():.3f}")
```

## Technical Details

### Data Processing Pipeline

1. **DICOM Reading**: Uses pydicom to read DICOM files
2. **Waveform Extraction**: Extracts ECG data from WaveformSequence
3. **Data Type Conversion**: Converts to float64 to avoid read-only issues
4. **Scaling Application**: Applies channel sensitivity if available
5. **Lead Handling**: Pads or truncates to exactly 12 leads
6. **Resampling**: Linear interpolation to 5000 samples
7. **Normalization**: DC offset removal and amplitude scaling

### Supported DICOM Formats

- **Waveform DICOM**: Primary support for ECG stored in WaveformSequence
- **Philips ECG**: Tested with Philips Medical Systems DICOM files
- **Standard ECG DICOM**: Should work with most standard ECG DICOM implementations

### Error Handling

The script provides detailed error reporting for:
- Unreadable DICOM files
- Missing waveform data
- Extraction failures
- Normalization issues

## Troubleshooting

### Common Issues

1. **"No recognized ECG data format found"**
   - The DICOM file doesn't contain WaveformSequence
   - Try checking if the file contains pixel_array data (image-based ECG)

2. **"Error extracting from waveform sequence"**
   - The waveform data format is not standard
   - Check the DICOM file structure and manufacturer

3. **"Failed to normalize data"**
   - The extracted ECG data has unexpected format
   - Check the data shape and values

### Getting Help

If you encounter issues with specific DICOM formats, you can:
1. Check the console output for detailed error messages
2. Examine the DICOM file structure using pydicom
3. Modify the extraction logic for your specific format

## Dependencies

- pydicom: DICOM file reading
- numpy: Array operations
- scipy: Signal processing and interpolation
- tqdm: Progress bars

All dependencies are managed through Poetry and installed automatically.
