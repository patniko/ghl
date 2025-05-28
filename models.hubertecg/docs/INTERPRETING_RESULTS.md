# Interpreting HuBERT-ECG Inference Results

This guide explains how to understand and use the results from running `make inference` on your ECG data.

## Overview

After running `make inference`, you get:
- **Feature files**: 768-dimensional representations for each time step
- **Metadata**: Information about the inference process  
- **Analysis tools**: Scripts to interpret and visualize the results

## Understanding Your Results

### 1. Results Structure

```
results/
├── inference_results.csv          # Metadata about all processed files
├── PATIENTA/
│   └── 9C687DB1_features.npy     # Extracted features for Patient A
├── PATIENTB/
│   └── 75C6B3C1_features.npy     # Extracted features for Patient B
└── PATIENTC/
    └── 93C7C5AD_features.npy     # Extracted features for Patient C
```

### 2. Metadata File (inference_results.csv)

| Column | Description | Example |
|--------|-------------|---------|
| `patient` | Patient identifier | PATIENTA |
| `filename` | Original ECG file | 9C687DB1.npy |
| `original_shape` | Raw ECG dimensions | (12, 5000) |
| `preprocessed_shape` | Model input format | torch.Size([1, 12000]) |
| `features_shape` | Extracted features | (1, 187, 768) |
| `features_file` | Path to feature file | PATIENTA/9C687DB1_features.npy |
| `status` | Processing status | success |

### 3. Feature Files

Each `.npy` file contains a 3D array with shape `(1, 187, 768)`:
- **Batch dimension**: 1 (single ECG)
- **Time steps**: 187 (temporal sequence)
- **Feature dimensions**: 768 (learned cardiac representations)

## What the Features Represent

### Cardiac Representations
The 768-dimensional features at each time step encode:
- **Heart rhythm patterns**: Temporal cardiac dynamics
- **Morphological characteristics**: ECG waveform shapes
- **Pathological indicators**: Potential abnormalities
- **Patient-specific patterns**: Individual cardiac signatures

### Temporal Structure
The 187 time steps represent:
- **Sequential cardiac events**: How features evolve over time
- **Rhythm analysis**: Periodic patterns in cardiac cycles
- **Temporal dependencies**: Relationships between different time points

## Interpreting Your Specific Results

Based on your inference results:

### Patient Overview
- **PATIENTA**: Original ECG (12, 5000) → Features (187, 768)
- **PATIENTB**: Original ECG (12, 5000) → Features (187, 768)  
- **PATIENTC**: Original ECG (12, 5000) → Features (187, 768)

### Key Insights
1. **Consistent processing**: All patients successfully processed
2. **Standard format**: 12-lead ECGs with 5000 samples each
3. **Rich representations**: 143,616 features per patient (187 × 768)

## Analysis Approaches

### 1. Basic Feature Analysis

```python
import numpy as np
import pandas as pd

# Load a feature file
features = np.load('results/PATIENTA/9C687DB1_features.npy')
features = features.squeeze(0)  # Remove batch dimension: (187, 768)

# Basic statistics
print(f"Shape: {features.shape}")
print(f"Mean: {np.mean(features):.6f}")
print(f"Std: {np.std(features):.6f}")
print(f"Range: [{np.min(features):.6f}, {np.max(features):.6f}]")
```

### 2. Temporal Pattern Analysis

```python
# Analyze how features evolve over time
temporal_means = np.mean(features, axis=1)  # Mean across 768 features for each time step
temporal_stds = np.std(features, axis=1)    # Variability for each time step

# Find most/least active periods
most_active_time = np.argmax(temporal_means)
least_active_time = np.argmin(temporal_means)

print(f"Most active time step: {most_active_time}")
print(f"Least active time step: {least_active_time}")
```

### 3. Feature Importance

```python
# Identify most variable features (potential biomarkers)
feature_variances = np.var(features, axis=0)  # Variance across time for each feature
top_features = np.argsort(feature_variances)[-10:]  # Top 10 most variable

print(f"Most important features: {top_features}")
print(f"Their variances: {feature_variances[top_features]}")
```

### 4. Patient Comparison

```python
# Compare two patients
features_a = np.load('results/PATIENTA/9C687DB1_features.npy').squeeze(0)
features_b = np.load('results/PATIENTB/75C6B3C1_features.npy').squeeze(0)

# Flatten for comparison
flat_a = features_a.flatten()
flat_b = features_b.flatten()

# Calculate similarity
cosine_sim = np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
euclidean_dist = np.linalg.norm(flat_a - flat_b)

print(f"Cosine similarity: {cosine_sim:.4f}")
print(f"Euclidean distance: {euclidean_dist:.2f}")
```

## Downstream Applications

### 1. Classification Tasks

Use features for cardiac condition classification:

```python
# Aggregate temporal features for classification
def prepare_features_for_ml(features):
    """Convert temporal features to fixed-size vectors."""
    # Option 1: Mean pooling
    mean_features = np.mean(features, axis=0)  # (768,)
    
    # Option 2: Statistical aggregation
    stats_features = np.concatenate([
        np.mean(features, axis=0),
        np.std(features, axis=0),
        np.max(features, axis=0),
        np.min(features, axis=0)
    ])  # (3072,)
    
    return mean_features  # or stats_features
```

### 2. Clustering Analysis

Group patients by cardiac similarity:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare feature matrix
feature_matrix = []
patient_names = []

for patient in ['PATIENTA', 'PATIENTB', 'PATIENTC']:
    features = np.load(f'results/{patient}/...features.npy').squeeze(0)
    aggregated = np.mean(features, axis=0)  # Mean pooling
    feature_matrix.append(aggregated)
    patient_names.append(patient)

# Standardize and cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_matrix)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

for patient, cluster in zip(patient_names, clusters):
    print(f"{patient}: Cluster {cluster}")
```

### 3. Risk Stratification

Correlate features with clinical outcomes:

```python
# Example: Correlate with clinical labels
clinical_data = {
    'PATIENTA': {'diagnosis': 'Normal', 'risk_score': 0.2},
    'PATIENTB': {'diagnosis': 'Abnormal', 'risk_score': 0.8},
    'PATIENTC': {'diagnosis': 'Normal', 'risk_score': 0.3}
}

# Extract features and labels
X = []  # Features
y = []  # Labels

for patient, data in clinical_data.items():
    features = np.load(f'results/{patient}/...features.npy').squeeze(0)
    aggregated = np.mean(features, axis=0)
    
    X.append(aggregated)
    y.append(1 if data['diagnosis'] == 'Abnormal' else 0)

# Train classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Feature importance
importance = clf.feature_importances_
top_indices = np.argsort(importance)[-10:]
print(f"Most important features: {top_indices}")
```

## Quality Assessment

### 1. Feature Quality Indicators

- **Non-zero ratio**: Should be > 0.5 (features are active)
- **Value range**: Typically [-5, 5] for normalized features
- **Temporal consistency**: Smooth evolution across time steps
- **Cross-patient variability**: Different patients should have different patterns

### 2. Success Metrics

- **Processing success rate**: Should be 100% for clean data
- **Feature consistency**: All patients should have same shape (187, 768)
- **Reasonable statistics**: Mean ≈ 0, std ≈ 1 for normalized features

## Advanced Analysis Tools

### 1. Run Comprehensive Analysis

```bash
# Install additional dependencies
poetry add matplotlib seaborn scikit-learn plotly

# Run full analysis pipeline
poetry run python scripts/analyze_features.py
poetry run python scripts/visualize_results.py
poetry run python scripts/classification_pipeline.py
```

### 2. Interactive Analysis

Use the Jupyter notebook for interactive exploration:

```bash
poetry run jupyter notebook notebooks/HuBERT_Feature_Analysis.ipynb
```

### 3. Simple Analysis (No Dependencies)

```bash
poetry run python scripts/simple_analysis_demo.py
```

## Troubleshooting

### Common Issues

1. **Empty features**: Check if ECG data is valid
2. **Inconsistent shapes**: Verify preprocessing pipeline
3. **Poor similarity**: May indicate diverse patient population
4. **Low feature variance**: Could suggest limited cardiac diversity

### Validation Steps

1. **Check metadata**: Verify all files processed successfully
2. **Inspect features**: Look for reasonable value ranges
3. **Compare patients**: Ensure meaningful differences
4. **Temporal analysis**: Verify smooth temporal evolution

## Next Steps

1. **Correlate with clinical data**: Link features to diagnoses
2. **Build predictive models**: Use for classification/regression
3. **Identify biomarkers**: Find most discriminative features
4. **Temporal modeling**: Analyze rhythm patterns
5. **Population studies**: Scale to larger patient cohorts

## References

- [HuBERT-ECG Paper](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v1)
- [Feature Analysis Scripts](scripts/)
- [Interactive Notebook](notebooks/HuBERT_Feature_Analysis.ipynb)

---

For questions or issues, refer to the analysis scripts or create an issue in the repository.
