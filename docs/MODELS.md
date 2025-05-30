# Models Overview

This repository contains three specialized AI models for cardiovascular analysis, each designed for specific clinical applications. This document provides a high-level overview of each model's purpose, capabilities, and primary use cases.

## 🏗️ Model Architecture

```
ghl/
├── models.echoprime/      # Multi-view echocardiography analysis
├── models.echoquality/    # Echo video quality assessment  
└── models.hubertecg/      # ECG foundation model
```

---

## 🫀 EchoPrime: Multi-View Echocardiography AI

**Location**: `models.echoprime/`  
**Paper**: [arXiv:2410.09704](https://arxiv.org/abs/2410.09704)

### Purpose
EchoPrime is a state-of-the-art AI system that automatically analyzes echocardiogram videos to generate comprehensive clinical reports and predict cardiac metrics. Unlike traditional single-view models, EchoPrime intelligently combines information from multiple echocardiographic views.

### Key Capabilities
- **Multi-View Intelligence**: Automatically identifies and combines information from multiple echocardiographic views
- **Comprehensive Reports**: Generates detailed clinical reports covering 15 anatomical sections
- **Expert-Level Accuracy**: Performance comparable to experienced cardiologists
- **DICOM Processing**: Native support for medical DICOM file formats

### Primary Use Cases

#### 1. Clinical Cardiac Assessment
- **Automated Report Generation**: Generate comprehensive echocardiography reports from DICOM videos
- **Multi-View Analysis**: Combine apical, parasternal, and subcostal views for complete cardiac evaluation
- **Cardiac Function Assessment**: Evaluate left ventricular function, wall motion, and valve performance

#### 2. Research Applications
- **Large-Scale Studies**: Process thousands of echocardiograms for population health research
- **Standardized Analysis**: Ensure consistent interpretation across different operators and institutions
- **Longitudinal Tracking**: Monitor cardiac changes over time with standardized metrics

#### 3. Clinical Decision Support
- **Screening Programs**: Rapid assessment of cardiac function in screening scenarios
- **Quality Assurance**: Validate human interpretations with AI-powered second opinions
- **Training Support**: Educational tool for cardiology trainees

### Quick Start
```bash
cd models.echoprime/
make init
make download-weights
# Place DICOM files in raw_data/
make inference
```

---

## 📊 EchoQuality: Video Quality Assessment

**Location**: `models.echoquality/`

### Purpose
EchoQuality is a specialized model that assesses the quality of echocardiogram videos before clinical analysis. It helps filter out low-quality studies that might lead to inaccurate diagnoses and ensures only suitable videos proceed to clinical interpretation.

### Key Capabilities
- **Automatic Quality Scoring**: Predicts whether video quality is acceptable for clinical use
- **Ultrasound Masking**: Automatically identifies and masks non-ultrasound regions
- **GradCAM Visualization**: Provides visual explanations of quality decisions
- **Debug Tools**: Comprehensive debugging suite for image scaling and processing issues

### Primary Use Cases

#### 1. Quality Control Pipeline
- **Pre-Processing Filter**: Screen echocardiograms before clinical analysis
- **Batch Processing**: Assess quality of large datasets automatically
- **Study Validation**: Ensure minimum quality standards for research studies

#### 2. Clinical Workflow Optimization
- **Acquisition Feedback**: Real-time quality assessment during echo acquisition
- **Repeat Study Decisions**: Determine when studies need to be repeated
- **Technician Training**: Provide objective feedback on image acquisition quality

#### 3. Research Data Curation
- **Dataset Cleaning**: Remove low-quality videos from research datasets
- **Quality Stratification**: Group studies by quality levels for analysis
- **Bias Reduction**: Ensure quality doesn't confound research results

### Quick Start
```bash
cd models.echoquality/
make inference  # Processes all DICOM files in raw_data/
```

---

## 🔬 HuBERT-ECG: ECG Foundation Model

**Location**: `models.hubertecg/`  
**Paper**: [medRxiv:2024.11.14.24317328](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v1)

### Purpose
HuBERT-ECG is a self-supervised foundation model for electrocardiogram analysis, pre-trained on 9.1 million 12-lead ECGs encompassing 164 cardiovascular conditions. It serves as a versatile base model that can be fine-tuned for a wide array of downstream ECG analysis tasks.

### Key Capabilities
- **Foundation Model**: Pre-trained on massive ECG dataset (9.1M ECGs, 164 conditions)
- **Multi-Task Performance**: Achieves AUROCs from 84.3% to 99% across diverse scenarios
- **Broad Disease Detection**: Detects 164 overlapping cardiovascular conditions simultaneously
- **Prognostic Prediction**: Predicts death events within 2-year follow-up (AUROC 93.4%)

### Primary Use Cases

#### 1. Cardiovascular Disease Detection
- **Multi-Disease Screening**: Simultaneous detection of 164 cardiovascular conditions
- **Arrhythmia Detection**: Identify various rhythm abnormalities from 12-lead ECGs
- **Structural Heart Disease**: Detect signs of cardiomyopathy, valve disease, and other structural abnormalities

#### 2. Risk Stratification
- **Mortality Prediction**: 2-year mortality risk assessment from ECG patterns
- **Cardiovascular Events**: Predict future cardiac events and complications
- **Population Screening**: Large-scale cardiovascular risk assessment

#### 3. Research and Development
- **Transfer Learning**: Fine-tune for specific ECG analysis tasks
- **Feature Extraction**: Extract meaningful ECG representations for downstream analysis
- **Clinical Trial Support**: Standardized ECG analysis for pharmaceutical research

#### 4. Custom Applications
- **Fine-Tuning**: Adapt the model for specific clinical scenarios or populations
- **Low-Data Settings**: Leverage pre-trained features when labeled data is limited
- **Real-Time Analysis**: Deploy for continuous ECG monitoring applications

### Quick Start
```bash
cd models.hubertecg/
make setup
make ptbxl-setup  # Download and setup PTB-XL dataset
make jupyter      # Launch demo notebooks
```

---

## 🔄 Model Integration Workflows

### Complete Cardiac Assessment Pipeline
1. **Quality Assessment** (EchoQuality): Filter high-quality echocardiograms
2. **Echo Analysis** (EchoPrime): Generate comprehensive cardiac reports
3. **ECG Analysis** (HuBERT-ECG): Complement with electrical activity assessment

### Research Workflow
1. **Data Curation**: Use EchoQuality to ensure dataset quality
2. **Standardized Analysis**: Apply EchoPrime for consistent echo interpretation
3. **Risk Stratification**: Use HuBERT-ECG for prognostic assessment

### Clinical Decision Support
1. **Screening**: HuBERT-ECG for initial cardiovascular risk assessment
2. **Detailed Imaging**: EchoPrime for comprehensive structural evaluation
3. **Quality Assurance**: EchoQuality to validate image acquisition

---

## 📚 Getting Started

### Prerequisites
- Python 3.11+
- Poetry 2.1+
- Docker & Docker Compose (optional)
- CUDA Toolkit (for GPU acceleration)

### Quick Setup
```bash
# Choose your model of interest
cd models.echoprime/    # For echocardiography analysis
cd models.echoquality/  # For quality assessment
cd models.hubertecg/    # For ECG analysis

# Follow model-specific setup instructions
make init               # Initialize environment
make download-weights   # Download pre-trained models (where applicable)
```

### Documentation Links
- **EchoPrime**: [`models.echoprime/docs/`](../models.echoprime/docs/)
- **EchoQuality**: [`models.echoquality/docs/`](../models.echoquality/docs/)
- **HuBERT-ECG**: [`models.hubertecg/docs/`](../models.hubertecg/docs/)

---

## 🎯 Choosing the Right Model

| Use Case | Recommended Model | Key Benefits |
|----------|------------------|--------------|
| Comprehensive echo reports | **EchoPrime** | Multi-view analysis, clinical-grade reports |
| Echo quality control | **EchoQuality** | Automated filtering, quality scoring |
| ECG disease detection | **HuBERT-ECG** | 164 conditions, foundation model flexibility |
| Cardiovascular screening | **HuBERT-ECG** | Population-scale analysis, risk prediction |
| Research data curation | **EchoQuality** | Objective quality assessment |
| Clinical decision support | **EchoPrime + HuBERT-ECG** | Comprehensive cardiac evaluation |

---

## 📄 Licensing

All models are released under permissive licenses:
- **EchoPrime**: MIT License
- **EchoQuality**: MIT License  
- **HuBERT-ECG**: CC BY-NC 4.0 License

## 🆘 Support

For model-specific questions, refer to the individual documentation in each model's `docs/` directory. For general questions about the repository structure or integration workflows, please open an issue in the main repository.
