# DICOM File Analyzer Tool

This tool analyzes DICOM files in the data directory and provides comprehensive statistics about files, devices, and patients.

## Features

- Scans all DICOM files in the data/ directory and subdirectories
- Extracts metadata including patient IDs, device information, and study details
- Provides summary statistics for files and patients per device
- Outputs both console summary and detailed JSON results

## Usage

### Basic Usage
```bash
python tools/dicom_analyzer.py
```

### Analyze a specific directory
```bash
python tools/dicom_analyzer.py path/to/dicom/directory
```

### Make it executable and run directly
```bash
chmod +x tools/dicom_analyzer.py
./tools/dicom_analyzer.py
```

## Output

The tool provides two types of output:

### 1. Console Summary
A formatted summary showing:
- Total files scanned and valid DICOM files
- Number of devices found
- Total unique patients across all devices
- Files per device
- Unique patients per device
- Device information (manufacturer, model)

### 2. JSON Output File
A detailed JSON file (`dicom_analysis_results.json`) containing:
- All summary statistics
- Complete patient lists per device
- Detailed device information
- File counts and metadata

## Example Output

```
DICOM FILE ANALYSIS SUMMARY
============================================================

OVERALL STATISTICS:
  Total files scanned: 4159
  Valid DICOM files: 4159
  Total devices: 7
  Unique patients (across all devices): 101

FILES PER DEVICE:
  affiniti50g: 174 files
  affiniti70g: 666 files
  epiq7: 1482 files
  example_study: 50 files
  vivide95: 788 files
  vivids60: 334 files
  vivids70: 665 files

UNIQUE PATIENTS PER DEVICE:
  affiniti50g: 15 patients
  affiniti70g: 16 patients
  epiq7: 17 patients
  example_study: 1 patients
  vivide95: 17 patients
  vivids60: 17 patients
  vivids70: 18 patients

DEVICE INFORMATION:
  affiniti50g:
    Manufacturer: Philips Medical Systems
    Model: Affiniti 50G
    Station Name: Unknown
  affiniti70g:
    Manufacturer: Philips Medical Systems
    Model: Affiniti 70G
    Station Name: Unknown
  epiq7:
    Manufacturer: Philips Medical Systems
    Model: Biograph16_Horizon 3R
    Station Name: Unknown
  example_study:
    Manufacturer: Philips Medical Systems
    Model: EPIQ 7C
    Station Name: Unknown
  vivide95:
    Manufacturer: GE Vingmed Ultrasound
    Model: Vivid E95
    Station Name: Unknown
  vivids60:
    Manufacturer: GE Healthcare Ultrasound
    Model: Vivid S60
    Station Name: Unknown
  vivids70:
    Manufacturer: GE Healthcare Ultrasound
    Model: Vivid S70
    Station Name: Unknown
```

## Dependencies

- Python 3.6+
- pydicom (already included in requirements.txt)
- Standard library modules: os, sys, pathlib, collections, json

## Error Handling

The tool handles:
- Invalid DICOM files (skips with warning)
- Missing metadata fields (uses 'Unknown' as default)
- File access errors
- Directory not found errors

## Integration

This tool can be easily integrated into data processing pipelines or used as a standalone utility for DICOM dataset analysis.
