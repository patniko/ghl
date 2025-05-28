# DICOM File Organizer Tool

This tool reorganizes DICOM files from the current directory structure into patient-specific directories with the format `devicename-patientid`.

## Purpose

The organizer transforms the current structure:
```
data/
├── device1/
│   ├── file1.dcm
│   ├── file2.dcm
│   └── file3.dcm
└── device2/
    ├── file4.dcm
    └── file5.dcm
```

Into a patient-organized structure:
```
data_organized/
├── device1-patient001/
│   ├── file1.dcm
│   └── file2.dcm
├── device1-patient002/
│   └── file3.dcm
└── device2-patient001/
    ├── file4.dcm
    └── file5.dcm
```

## Features

- Reads DICOM metadata to extract patient IDs
- Creates patient-specific directories with format `devicename-patientid`
- Supports both copying and moving files (copying is default for safety)
- Provides detailed statistics and verification
- Handles both full data directory and individual device directories
- Creates comprehensive logs of the organization process

## Usage

### Basic Usage (Organize entire data directory)
```bash
python tools/dicom_organizer.py
```

### Organize specific device directory
```bash
python tools/dicom_organizer.py data/device_name data_organized
```

### Custom source and target directories
```bash
python tools/dicom_organizer.py source_directory target_directory
```

### Move files instead of copying (use with caution)
```bash
python tools/dicom_organizer.py data data_organized false
```

### Make executable and run directly
```bash
chmod +x tools/dicom_organizer.py
./tools/dicom_organizer.py
```

## Command Line Arguments

1. **Source Directory** (optional, default: `data`)
   - Directory containing DICOM files to organize
   
2. **Target Directory** (optional, default: `data_organized`)
   - Directory where organized files will be placed
   
3. **Copy Files** (optional, default: `true`)
   - `true`/`copy`/`yes`/`1`: Copy files (safer, preserves originals)
   - `false`/`move`/`no`/`0`: Move files (faster, but removes originals)

## Output

### Console Summary
Shows real-time progress and final statistics:
- Files processed and organized
- Patient directories created
- Device information
- Any failed files

### JSON Log File
Creates `dicom_organization_log.json` with detailed information:
- Complete statistics
- List of all patient directories created
- Failed files (if any)
- Device processing information

### Directory Verification
Automatically verifies the organization by:
- Counting patient directories created
- Verifying file counts match
- Ensuring proper directory structure

## Example Output

```
DICOM File Organizer
==============================
Source directory: data
Target directory: data_organized
Operation: Copy files

Organizing DICOM files from data to data_organized
============================================================
Processing device: affiniti50g
  Created directory: affiniti50g-Affiniti_50G-3
  Created directory: affiniti50g-Affiniti_50G-10
  ...
Processing device: vivide95
  Created directory: vivide95-VIVIDE95-022578-16
  ...

============================================================
DICOM FILE ORGANIZATION SUMMARY
============================================================

FILE STATISTICS:
  Total files processed: 4159
  Files successfully organized: 4159
  Files failed: 0

ORGANIZATION STATISTICS:
  Devices processed: 7
  Patient directories created: 101

DEVICES PROCESSED:
  - affiniti50g
  - affiniti70g
  - epiq7
  - example_study
  - vivide95
  - vivids60
  - vivids70

PATIENT DIRECTORIES CREATED (101):
  - affiniti50g-Affiniti_50G-2
  - affiniti50g-Affiniti_50G-3
  - affiniti50g-Affiniti_50G-4
  ...

Organization log saved to: dicom_organization_log.json

Verifying organization...
Verification complete:
  Patient directories: 101
  Total files organized: 4159

Organization complete!
```

## Safety Features

- **Default to copying**: Files are copied by default, preserving originals
- **Confirmation prompt**: When moving files, asks for user confirmation
- **Error handling**: Gracefully handles invalid DICOM files and access errors
- **Verification**: Automatically verifies organization results
- **Detailed logging**: Creates comprehensive logs for audit trails

## Integration with Analysis Tool

This organizer works perfectly with the DICOM analyzer tool:

1. First, analyze your data:
   ```bash
   python tools/dicom_analyzer.py
   ```

2. Then organize the files:
   ```bash
   python tools/dicom_organizer.py
   ```

3. Analyze the organized structure:
   ```bash
   python tools/dicom_analyzer.py data_organized
   ```

## Dependencies

- Python 3.6+
- pydicom (already included in requirements.txt)
- Standard library modules: os, sys, shutil, pathlib, collections, json

## Error Handling

The tool handles various error conditions:
- Invalid DICOM files (skipped with warning)
- Missing patient ID metadata (uses 'Unknown')
- File access permissions
- Disk space issues
- Directory creation failures

## Performance Considerations

- **Copying vs Moving**: Moving is faster but removes originals
- **Large datasets**: Progress is shown in real-time
- **Disk space**: Copying requires double the disk space temporarily
- **Network storage**: Works with network-mounted directories

## Use Cases

1. **Research datasets**: Organize multi-device studies by patient
2. **Clinical workflows**: Prepare data for patient-specific analysis
3. **Data migration**: Restructure existing DICOM archives
4. **Quality control**: Verify patient data completeness across devices
5. **Machine learning**: Prepare patient-grouped datasets for training
