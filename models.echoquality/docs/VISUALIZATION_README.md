# EchoQuality Visualization Tools

This document describes the visualization tools added to the EchoQuality project for analyzing inference results.

## Overview

The EchoQuality project now includes comprehensive visualization capabilities similar to the models.hubertecg project, allowing you to analyze and visualize quality assessment results from echo video inference.

## Quick Start

1. **Run inference first** (if you haven't already):
   ```bash
   make inference
   ```

2. **Generate static visualizations**:
   ```bash
   make visualize
   ```

3. **Launch interactive analysis**:
   ```bash
   make jupyter
   ```
   Then open the `notebooks/EchoQuality-Interactive-Analysis.ipynb` notebook.

## Features

### Static Visualizations (`make visualize`)

The `make visualize` command runs the `scripts/visualize_results.py` script which generates:

- **Overall Summary Dashboard**: Comprehensive overview with key statistics
- **Score Analysis Plots**: Detailed score distribution analysis with statistics
- **Folder Comparison Plots**: Performance comparison across different folders
- **Interactive HTML Plots**: Plotly-based interactive visualizations
- **Summary Report**: Text-based analysis report

**Output Location**: `./visualization_output/`

**Generated Files**:
- `echoquality_dashboard.png` - Main summary dashboard
- `score_analysis.png` - Detailed score analysis
- `folder_comparison.png` - Folder performance comparison
- `interactive_score_distribution.html` - Interactive score plots
- `interactive_folder_analysis.html` - Interactive folder comparison
- `interactive_folder_heatmap.html` - Performance heatmap
- `analysis_report.txt` - Comprehensive text report

### Interactive Analysis Notebook

The Jupyter notebook `notebooks/EchoQuality-Interactive-Analysis.ipynb` provides:

- **Interactive Data Explorer**: Filter and explore results with widgets
- **Folder Performance Analysis**: Detailed folder-by-folder analysis
- **Quality Assessment Breakdown**: Distribution of quality assessments
- **Error Analysis**: Analysis of processing errors and issues
- **Custom Analysis Functions**: Tools for comparing specific folders
- **Export Capabilities**: Save processed data and custom visualizations

## Data Structure

The visualization tools work with the inference results structure:

```
results/inference_output/
├── summary.json                 # Overall summary statistics
└── [folder-name]/
    ├── inference_results.json   # Detailed results for folder
    ├── folder_summary.json      # Folder-level summary
    ├── pass_fail_distribution.png
    ├── score_distribution.png
    └── mask_images/            # Generated mask visualizations
```

## Key Metrics Analyzed

- **Quality Scores**: Continuous scores from 0.0 to 1.0
- **Pass/Fail Status**: Binary classification (threshold: 0.5)
- **Quality Assessments**: Categorical quality descriptions
- **Processing Statistics**: File counts, success rates, error analysis
- **Folder Performance**: Comparative analysis across data sources

## Usage Examples

### Basic Visualization
```bash
# Generate all static visualizations
make visualize

# View results
open visualization_output/echoquality_dashboard.png
```

### Custom Analysis
```bash
# Launch Jupyter for interactive analysis
make jupyter

# Open the analysis notebook
# Navigate to: notebooks/EchoQuality-Interactive-Analysis.ipynb
```

### Advanced Usage
```bash
# Run visualization with custom parameters
poetry run python scripts/visualize_results.py \
    --results_dir ./results/inference_output \
    --output_dir ./custom_output
```

## Comparison with HuBERT-ECG

The EchoQuality visualization tools are modeled after the successful `models.hubertecg` visualization system but adapted for:

- **Quality Assessment Data**: Instead of feature vectors
- **Video Processing Results**: Instead of ECG signal analysis
- **Pass/Fail Metrics**: Instead of feature importance analysis
- **Folder-based Organization**: For different echo data sources

## Dependencies

The visualization tools require:
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualizations
- `plotly` - Interactive plots
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `ipywidgets` - Jupyter notebook widgets

These are included in the project's `pyproject.toml` dependencies.

## Troubleshooting

### Common Issues

1. **No results found**: Ensure you've run `make inference` first
2. **Missing dependencies**: Run `poetry install` to install all dependencies
3. **Jupyter not starting**: Check that port 8888 is available or specify a different port

### File Locations

- **Scripts**: `scripts/visualize_results.py`
- **Notebooks**: `notebooks/EchoQuality-Interactive-Analysis.ipynb`
- **Results**: `results/inference_output/`
- **Visualizations**: `visualization_output/`

## Integration with Workflow

The visualization tools integrate seamlessly with the EchoQuality workflow:

1. **Data Preparation**: Place echo videos in `raw_data/`
2. **Inference**: Run `make inference` to process videos
3. **Visualization**: Run `make visualize` for static analysis
4. **Interactive Analysis**: Use Jupyter notebook for detailed exploration
5. **Reporting**: Generated reports and visualizations for sharing results

This provides a complete pipeline from raw echo videos to comprehensive quality analysis and visualization.
