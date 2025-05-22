# GHL Tools

This directory contains utility tools for the GHL (Global Health Lab) project. These tools are designed to help with various tasks related to data generation, processing, and analysis.

## Current Tools

### samples.py

A utility for generating and processing sample data for the GHL project. This tool can generate various types of health-related data including:

- Questionnaire data
- Blood test results
- Mobile health measures
- Patient consent records
- Echo (DICOM) data processing
- ECG data processing

The tool can use existing samples in the sample folder or generate synthetic data dynamically.

#### Features

- Generate realistic sample data with appropriate correlations and distributions
- Process existing DICOM and ECG files
- Export data to CSV format
- Create zip archives of generated data
- Support for generating partial records to simulate real-world data collection scenarios

#### Dependencies

The tool has the following dependencies:
- Required: Python 3.11+, pandas, numpy
- Optional: pydicom (for Echo data), opencv-python (for image processing), torch (for advanced processing)

#### Usage Examples

Generate all types of sample data:
```bash
python samples.py --type all --num-samples 20
```

Generate only blood test results:
```bash
python samples.py --type blood --num-samples 10
```

Generate data with partial records (simulating real-world scenarios):
```bash
python samples.py --include-partials --partial-rate 0.3
```

Export to a specific directory and create a zip archive:
```bash
python samples.py --output-dir ./test_output --create-zip
```

## Poetry Setup and Usage

This directory uses [Poetry](https://python-poetry.org/) for dependency management. Poetry provides a robust way to manage Python dependencies, virtual environments, and package distribution.

### Installation

To set up the development environment:

1. Make sure you have Poetry installed (https://python-poetry.org/docs/#installation)
2. Run the following command in the tools directory:
   ```bash
   poetry install
   ```

This will create a virtual environment and install all the required dependencies specified in the pyproject.toml file.

### Running Tools with Poetry

To run any tool in this directory using the Poetry-managed environment:

```bash
poetry run python samples.py --type all
```

Alternatively, you can activate the Poetry shell and run commands directly:

```bash
poetry shell
python samples.py --type all
```

### Managing Dependencies

To add new dependencies to the project:

```bash
poetry add package-name
```

To add development-only dependencies:

```bash
poetry add --group dev package-name
```

To update dependencies:

```bash
poetry update
```

The dependencies are specified in the pyproject.toml file and locked versions are stored in poetry.lock.

## Future Tools

This directory will be expanded with additional tools for the GHL project. Potential future tools may include:

- Data validation and quality checking
- Data transformation and normalization
- Model training utilities
- Visualization tools
- Report generation

## Contributing

When adding new tools to this directory, please follow these guidelines:
1. Include comprehensive documentation in the tool itself
2. Update this README.md with information about the new tool
3. Ensure the tool follows the same command-line interface pattern as existing tools
4. Add any new dependencies using Poetry (`poetry add package-name`)
5. Make sure to commit both pyproject.toml and poetry.lock when dependencies change
6. Include usage examples with Poetry commands
