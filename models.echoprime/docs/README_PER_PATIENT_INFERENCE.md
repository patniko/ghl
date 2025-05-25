# EchoPrime Per-Patient Inference

This enhanced inference script processes each DICOM file as an individual patient and provides comprehensive device-specific analysis to help optimize ultrasound capture protocols.

## Quick Start

Run the per-patient inference with:
```bash
make inference-per-patient
```

Or directly:
```bash
poetry run python -m inference.inference_per_patient --data_dir ./data --output ./results/inference_output_per_patient
```

## What's Different from Original Inference

### Original Script (`make inference`)
- Processes entire device folders as single "studies"
- Combines all DICOM files from multiple patients
- Generates one report per device folder
- **Problem**: Creates meaningless composite results across different patients

### New Script (`make inference-per-patient`)
- Processes each DICOM file as a separate patient
- Generates individual clinical reports and metrics per patient
- Provides device-specific performance analysis
- **Benefit**: Clinically meaningful results for each patient

## Output Structure

```
results/inference_output_per_patient/
├── comprehensive_analysis.json          # Complete device analysis & recommendations
├── summary.json                         # Simple summary
├── affiniti50g/
│   ├── patient_1046703670967259063243803276037719201.json
│   ├── patient_4266369087561534034340696629413656908.json
│   ├── ...
│   └── device_summary.json             # Device-specific summary
├── vivide95/
│   ├── patient_xxx.json
│   └── device_summary.json
└── ...
```

## Individual Patient Results

Each patient file contains:
```json
{
  "patient_id": "1046703670967259063243803276037719201",
  "device": "affiniti50g",
  "status": "success",
  "report": "Left Ventricle: Normal left ventricular size...",
  "metrics": {
    "ejection_fraction": 58.2,
    "mitral_regurgitation": 0.15,
    "aortic_stenosis": 0.02,
    ...
  },
  "view_info": {
    "view_name": "apical_4chamber",
    "confidence": 0.92,
    "view_class": 2
  },
  "processing_info": {
    "original_frames": 45,
    "processed_frames": 16
  },
  "quality_indicators": {
    "visualization_rate": 0.85,
    "well_visualized_count": 8,
    "not_well_visualized_count": 2
  }
}
```

## Device Analysis & Recommendations

The `comprehensive_analysis.json` file provides:

### Device Performance Metrics
- Success rates per device
- Average clinical metrics (ejection fraction, etc.)
- View distribution analysis
- Quality indicators

### Actionable Recommendations
Examples of what you might see:
- **"High poor quality rate (25%). Review image acquisition technique."**
- **"Low visualization rate (65%). Optimize gain, depth, and focus settings."**
- **"High off-axis rate (18%). Focus on probe positioning and angulation."**
- **"Limited view diversity. Ensure comprehensive echocardiographic examination."**

## Key Features

### 1. View Classification Analysis
- Identifies which echocardiogram views each device captures
- Detects poor quality or off-axis acquisitions
- Recommends protocol improvements

### 2. Clinical Metrics Comparison
- Compares ejection fraction measurements across devices
- Identifies devices with high measurement variability
- Flags potential calibration issues

### 3. Quality Assessment
- Tracks "well visualized" vs "not well visualized" rates
- Analyzes report completeness
- Identifies devices needing protocol optimization

### 4. Processing Efficiency
- Monitors frame extraction success rates
- Identifies technical processing issues
- Recommends acquisition parameter adjustments

## Understanding the Metrics

### Clinical Metrics
- **ejection_fraction**: Left ventricular ejection fraction (normal ~55-70%)
- **mitral_regurgitation**: Probability of mitral valve regurgitation (0.0-1.0)
- **aortic_stenosis**: Probability of aortic valve stenosis (0.0-1.0)
- **left_atrium_dilation**: Probability of left atrial enlargement (0.0-1.0)

### Quality Indicators
- **visualization_rate**: Percentage of structures well visualized
- **view_confidence**: AI confidence in view classification
- **processing_success_rate**: Percentage of files processed successfully

### View Types
- **parasternal_long**: Parasternal long-axis view
- **parasternal_short**: Parasternal short-axis view
- **apical_4chamber**: Apical 4-chamber view
- **apical_2chamber**: Apical 2-chamber view
- **subcostal_4chamber**: Subcostal 4-chamber view
- **poor_quality**: Low quality acquisition
- **off_axis**: Incorrectly positioned probe

## Protocol Optimization Workflow

1. **Run Analysis**: `make inference-per-patient`
2. **Review Recommendations**: Check `comprehensive_analysis.json`
3. **Identify Problem Devices**: Look for low success rates or poor quality
4. **Adjust Protocols**: Modify acquisition settings based on recommendations
5. **Re-test**: Run analysis again to verify improvements

## Example Optimization Scenarios

### Scenario 1: High Poor Quality Rate
**Problem**: Device shows 30% poor quality acquisitions
**Recommendations**:
- Review operator training
- Check gain and depth settings
- Verify probe condition

### Scenario 2: Low Ejection Fraction Success
**Problem**: EF measurements failing in 40% of patients
**Recommendations**:
- Focus on optimal left ventricular visualization
- Ensure adequate apical views
- Check for adequate contrast

### Scenario 3: Limited View Diversity
**Problem**: Only capturing 2 of 4 standard views
**Recommendations**:
- Expand examination protocol
- Train operators on additional views
- Verify probe positioning techniques

## Command Line Options

```bash
python -m inference.inference_per_patient \
  --data_dir ./data \                    # Input directory with device folders
  --output ./results/output_per_patient \ # Output directory
  --weights_dir ./weights \              # Model weights directory
  --device auto                          # Device: auto, cpu, or cuda
```

## Troubleshooting

### Common Issues

1. **No DICOM files found**
   - Verify data directory structure
   - Check file extensions (.dcm, .dicom, or no extension)

2. **Processing failures**
   - Check DICOM file integrity
   - Verify sufficient memory/GPU resources

3. **Low success rates**
   - Review DICOM file quality
   - Check acquisition parameters

### Performance Tips

- Use GPU acceleration when available (`--device cuda`)
- Process smaller batches if memory limited
- Ensure adequate disk space for results

## Integration with Clinical Workflow

This analysis helps you:
1. **Standardize Quality**: Ensure consistent image quality across devices
2. **Optimize Protocols**: Adjust acquisition settings per device type
3. **Train Operators**: Identify areas needing additional training
4. **Equipment Decisions**: Make data-driven equipment purchases
5. **Quality Assurance**: Monitor ongoing performance trends

The goal is to maximize the diagnostic accuracy and clinical utility of your echocardiogram AI analysis by optimizing the input data quality for each specific ultrasound device.
