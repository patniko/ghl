# AliveCor JSON to NPY Conversion

This document explains how to convert AliveCor JSON ECG files to NPY format for use with HuBERT-ECG.

## Overview

The `alivecor_to_npy.py` script converts AliveCor JSON ECG files to the NPY format expected by the HuBERT-ECG model. It handles the 6-lead ECG data from AliveCor devices and normalizes it appropriately.

## Input Format

AliveCor JSON files contain:
- 6-lead ECG data: leadI, leadII, leadIII, AVR, AVL, AVF
- 300 Hz sampling frequency
- Both "raw" and "enhanced" data versions
- Amplitude resolution scaling factor (typically 500)
- Duration in milliseconds

## Output Format

The script produces NPY files with:
- Shape: (6, 5000) - 6 leads with 5000 samples each
- Normalized amplitude range similar to PTB-XL data
- Resampled to target length using linear interpolation
- DC offset removed and standardized scaling applied

## Usage

### Basic Usage

```bash
python preprocessors/alivecor_to_npy.py
```

This will:
- Read JSON files from `raw_data/PATIENT-ALIVECOR/`
- Use "enhanced" data by default
- Save NPY files to `preprocessed_data/PATIENT-ALIVECOR/`
- Target 5000 samples per lead

### Advanced Usage

```bash
python preprocessors/alivecor_to_npy.py \
    --input_dir raw_data \
    --output_dir preprocessed_data \
    --data_type enhanced \
    --target_length 5000
```

### Parameters

- `--input_dir`: Directory containing PATIENT-ALIVECOR folder (default: `raw_data`)
- `--output_dir`: Directory to save NPY files (default: `preprocessed_data`)
- `--data_type`: Use "raw" or "enhanced" data (default: `enhanced`)
- `--target_length`: Number of samples in output (default: `5000`)

## Data Processing Steps

1. **JSON Parsing**: Load and validate JSON structure
2. **Lead Extraction**: Extract 6-lead ECG data arrays
3. **Amplitude Scaling**: Apply amplitude resolution factor
4. **Resampling**: Interpolate to target sample length
5. **Normalization**: 
   - Remove DC offset (subtract mean)
   - Normalize by standard deviation with gentle scaling
   - Maintain physiological signal characteristics
6. **Output**: Save as NPY files with shape (6, 5000)

## File Structure

```
raw_data/
└── PATIENT-ALIVECOR/
    ├── alivecor-1.json
    ├── alivecor-2.json
    └── ...

preprocessed_data/
└── PATIENT-ALIVECOR/
    ├── alivecor-1.npy
    ├── alivecor-2.npy
    └── ...
```

## Differences from DICOM Processing

| Aspect | DICOM | AliveCor |
|--------|-------|----------|
| Input Format | Binary DICOM files | JSON text files |
| Lead Count | 12 leads (padded/derived) | 6 leads (native) |
| Data Extraction | WaveformSequence parsing | JSON key access |
| Sampling Rate | Variable | 300 Hz (typical) |
| Amplitude Scaling | DICOM sensitivity units | JSON amplitude resolution |

## Quality Checks

The script performs several validation steps:
- Verifies JSON structure and required fields
- Checks for missing leads and reports warnings
- Validates data array lengths and types
- Reports processing statistics and success rates

## Integration with Existing Pipeline

The AliveCor NPY files are compatible with:
- HuBERT-ECG feature extraction
- Existing analysis scripts
- Visualization tools
- Classification pipelines

Note: Some scripts may need minor modifications to handle 6-lead vs 12-lead data differences.

## Troubleshooting

### Common Issues

1. **Missing JSON fields**: Check that JSON files contain required structure
2. **Empty lead arrays**: Verify that lead data arrays are not empty
3. **Amplitude scaling**: Adjust normalization if signals appear too large/small
4. **Memory issues**: Process files in batches for large datasets

### Error Messages

- `"Enhanced data not found in JSON"`: JSON missing expected data section
- `"No valid lead data found"`: All lead arrays are empty or missing
- `"Invalid JSON format"`: File is not valid JSON

## Example Output

```
Processing AliveCor directory: raw_data/PATIENT-ALIVECOR

Processing AliveCor patient: PATIENT-ALIVECOR
  Found 10 JSON files

Processing: alivecor-1.json
  Patient ID: PATIENT123
  Duration: 14526 ms
  Recorded: 2022-06-21T14:59:14+05:30
  Sampling Rate: 300 Hz
  Amplitude Resolution: 500
  Number of Leads: 6
  Using: enhanced data
    leadI: 4358 samples
    leadII: 4358 samples
    leadIII: 4358 samples
    AVR: 4358 samples
    AVL: 4358 samples
    AVF: 4358 samples
  Extracted ECG shape: (6, 4358)
  Normalizing from (6, 4358) to (6, 5000)
  Expected samples from duration: 4358
    Saved: alivecor-1.npy with shape (6, 5000)

=== ALIVECOR CONVERSION SUMMARY ===
Data type used: enhanced
Target sample length: 5000
Total JSON files: 10
Successfully converted: 10
Success rate: 100.0%
