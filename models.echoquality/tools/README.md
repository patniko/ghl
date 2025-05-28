# DICOM Debug Tools

This directory contains debug tools for investigating image scaling issues in DICOM files. All tools have been updated to allow users to specify custom DICOM file paths instead of using hardcoded paths.

## Available Tools

### 1. debug_scaling.py
**Basic debugging with detailed logging and visualization**

- Provides detailed logging of each step in the crop_and_scale process
- Saves debug images showing the transformation at each stage
- Processes up to 5 frames for analysis
- Creates comprehensive log files

**Usage:**
```bash
# Interactive mode (prompts for file path)
python debug_scaling.py

# Command line mode
python debug_scaling.py /path/to/your/dicom/file

# From Python code
from debug_scaling import main
main("/path/to/your/dicom/file")
```

**Output:**
- Debug images saved to `./results/debug_images/`
- Log file: `debug_scaling.log`

### 2. debug_scaling_interactive.py
**Interactive debugging with step-by-step breakpoints**

- Provides interactive debugging using Python's pdb debugger
- Allows you to step through the scaling process manually
- Displays images at each stage for visual inspection
- Lets you choose which frames to process

**Usage:**
```bash
# Interactive mode (prompts for file path)
python debug_scaling_interactive.py

# Command line mode
python debug_scaling_interactive.py /path/to/your/dicom/file
```

**Features:**
- Interactive breakpoints at each processing stage
- Frame selection (choose which frames to analyze)
- Real-time image display using matplotlib
- Step-by-step debugging with pdb

**Output:**
- Debug images saved to `./results/debug_images_interactive/`
- Log file: `debug_scaling_interactive.log`

### 3. debug_scaling_specialized.py
**Specialized debugging for unusually shaped images**

- Focuses on problems with very narrow or wide images
- Provides alternative scaling approaches for extreme aspect ratios
- Includes detailed analysis of why standard scaling fails
- Demonstrates padding-to-square approach for unusual dimensions

**Usage:**
```bash
# Interactive mode (prompts for file path)
python debug_scaling_specialized.py

# Command line mode
python debug_scaling_specialized.py /path/to/your/dicom/file
```

**Features:**
- Alternative scaling algorithm for extreme aspect ratios
- Detailed analysis of scaling issues
- Pixel-level visualization for very small images
- Recommendations for handling unusual image dimensions

**Output:**
- Debug images saved to `./results/debug_images_specialized/`
- Log file: `debug_scaling_specialized.log`
- Summary of findings and recommendations

### 4. debug_scaling_visual.py
**Visual debugging with detailed comparisons**

- Creates comprehensive visual comparisons
- Shows before/after images at each processing stage
- Includes histograms, edge detection, and image statistics
- Provides detailed visualizations of cropping plans

**Usage:**
```bash
# Interactive mode (prompts for file path)
python debug_scaling_visual.py

# Command line mode
python debug_scaling_visual.py /path/to/your/dicom/file
```

**Features:**
- Side-by-side image comparisons
- Detailed image statistics and histograms
- Visual crop planning with overlay guides
- Frame-specific debug directories
- Edge detection analysis

**Output:**
- Debug images saved to `./results/debug_images_visual/`
- Individual frame directories with step-by-step visualizations
- Log file: `debug_scaling_visual.log`

## Common Usage Patterns

### Quick Analysis
For a quick analysis of scaling issues:
```bash
python debug_scaling.py /path/to/dicom/file
```

### Deep Investigation
For detailed investigation with visual feedback:
```bash
python debug_scaling_visual.py /path/to/dicom/file
```

### Interactive Debugging
For step-by-step debugging and experimentation:
```bash
python debug_scaling_interactive.py /path/to/dicom/file
```

### Unusual Image Shapes
For images with extreme aspect ratios or unusual dimensions:
```bash
python debug_scaling_specialized.py /path/to/dicom/file
```

## File Path Formats

All tools accept various file path formats:

- **Absolute paths:** `/full/path/to/dicom/file`
- **Relative paths:** `./relative/path/to/dicom/file`
- **Paths with spaces:** `"/path/with spaces/to/dicom/file"`

## Output Structure

Each tool creates its own output directory structure:

```
results/
├── debug_images/                    # Basic debug tool output
├── debug_images_interactive/        # Interactive debug tool output
├── debug_images_specialized/        # Specialized debug tool output
└── debug_images_visual/            # Visual debug tool output
    ├── frame_0/                    # Frame-specific visualizations
    │   ├── 01_original_details.png
    │   ├── 02_width_crop_plan.png
    │   └── ...
    └── frame_1/
        └── ...
```

## Log Files

Each tool generates detailed log files:

- `debug_scaling.log` - Basic debugging logs
- `debug_scaling_interactive.log` - Interactive session logs
- `debug_scaling_specialized.log` - Specialized analysis logs
- `debug_scaling_visual.log` - Visual debugging logs

## Requirements

All tools require the following Python packages:
- `numpy`
- `opencv-python` (cv2)
- `pydicom`
- `matplotlib`
- `pathlib`

## Tips

1. **Start with the basic tool** (`debug_scaling.py`) for initial analysis
2. **Use the visual tool** (`debug_scaling_visual.py`) for comprehensive investigation
3. **Try the specialized tool** if you're dealing with unusual image dimensions
4. **Use the interactive tool** when you need to experiment with different parameters
5. **Check the log files** for detailed information about processing steps and any errors

## Troubleshooting

- **File not found errors:** Ensure the DICOM file path is correct and the file exists
- **Permission errors:** Make sure you have read access to the DICOM file and write access to the results directory
- **Memory issues:** For large DICOM files, the tools process only a limited number of frames to avoid memory problems
- **Display issues:** The interactive and visual tools require a display environment (X11 forwarding if using SSH)
