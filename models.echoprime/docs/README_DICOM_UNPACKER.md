# DICOM Unpacker Scripts

This directory contains scripts to unpack and extract content from DICOM files in the EchoPrime raw_data directory.

## Scripts

### 1. `unpack_dicom_files.py`
The main unpacking script that extracts content from DICOM files.

**Features:**
- Extracts image data as PNG files
- Extracts video data as MP4 files (for multi-frame DICOM)
- Extracts comprehensive metadata as JSON files
- Extracts audio data if present
- Creates organized directory structure
- Handles various DICOM formats and pixel data types

**Usage:**
```bash
python unpack_dicom_files.py --raw-data-dir /path/to/raw_data --output-dir /path/to/output
```

**Options:**
- `--raw-data-dir`: Path to the directory containing DICOM files (default: raw_data)
- `--output-dir`: Output directory for extracted content (default: unpacked_dicom)
- `--save-log`: Save detailed log to JSON file

### 2. `unpack_echoprime_dicom.py`
A simplified script specifically for the EchoPrime project that automatically uses the correct paths.

**Usage:**
```bash
cd models.echoprime/scripts
python unpack_echoprime_dicom.py
```

This script will:
- Automatically find the `raw_data` directory
- Extract content to `unpacked_data` directory
- Save a detailed log file

## Requirements

Make sure you have the required Python packages installed:

```bash
pip install pydicom pillow opencv-python numpy
```

Or if you're using the project's requirements:
```bash
pip install -r ../requirements.txt
```

## Output Structure

The unpacker creates the following directory structure:

```
unpacked_data/
├── PatientID_StudyUID_SeriesUID/
│   ├── metadata.json          # Complete DICOM metadata
│   ├── image.png              # Single frame image (if applicable)
│   ├── video.mp4              # Multi-frame video (if applicable)
│   ├── audio.wav              # Audio data (if present)
│   └── frames/                # Individual frames (for multi-frame)
│       ├── frame_0000.png
│       ├── frame_0001.png
│       └── ...
└── echoprime_unpacking_log.json  # Detailed processing log
```

## What Gets Extracted

### Images
- Single-frame DICOM files are saved as PNG images
- Multi-frame DICOM files are saved as both MP4 videos and individual PNG frames
- Pixel data is normalized to 8-bit for compatibility
- Both grayscale and color images are supported

### Videos
- Multi-frame DICOM sequences are converted to MP4 format
- Default frame rate is 30 FPS (can be modified in the script)
- Individual frames are also extracted as PNG files

### Metadata
- Complete DICOM metadata is saved as JSON
- Includes patient information, study details, equipment info, and image parameters
- Metadata is organized into logical sections (patient, study, series, equipment, image)

### Audio
- Audio data embedded in DICOM files is extracted as WAV files
- Raw audio data is preserved

## Example Usage

### Quick Start (Recommended)
```bash
cd models.echoprime/scripts
python unpack_echoprime_dicom.py
```

### Custom Paths
```bash
python unpack_dicom_files.py --raw-data-dir ../raw_data --output-dir ../extracted_content --save-log
```

### From Project Root
```bash
cd models.echoprime
python scripts/unpack_echoprime_dicom.py
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install pydicom pillow opencv-python numpy
   ```

2. **Permission Errors**
   - Make sure you have write permissions to the output directory
   - Try running with appropriate permissions

3. **Invalid DICOM Files**
   - The script will skip invalid DICOM files and report them in the log
   - Check the log file for details about failed extractions

4. **Memory Issues with Large Files**
   - Large multi-frame DICOM files may require significant memory
   - Consider processing files individually if you encounter memory errors

### Output Verification

After running the script, check:
- The `unpacked_data` directory for extracted content
- The log file for processing statistics and any errors
- Individual patient directories for the extracted images/videos

## Technical Details

### Supported DICOM Types
- Single-frame images (2D pixel arrays)
- Multi-frame sequences (3D pixel arrays)
- Color and grayscale images
- Various bit depths (8-bit, 16-bit, etc.)
- Compressed and uncompressed DICOM files

### Image Processing
- Automatic normalization to 8-bit range
- Preservation of aspect ratio
- Support for different photometric interpretations
- Handling of various pixel spacing and orientations

### Video Creation
- Uses OpenCV for video encoding
- MP4 format with H.264 codec
- Configurable frame rate
- Automatic handling of grayscale and color sequences

## File Naming Convention

Extracted content is organized using a safe naming convention:
- Patient ID is sanitized (alphanumeric and hyphens/underscores only)
- Study and Series UIDs are truncated to last 20 characters
- Format: `PatientID_StudyUID_SeriesUID`

This ensures compatibility across different file systems and avoids issues with special characters.
