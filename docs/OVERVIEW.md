# Technical Requirements for Cardiovascular Disease Screening Platform

## 1. Introduction

### 1.1 Purpose

This document outlines the technical requirements for developing a comprehensive data collection and management platform to support a clinical study on cardiovascular disease screening in patient populations with diabetes or hypertension.

### Top 5 CVD Conditions of Interest

As part of this study, we will be focused on creating a data asset that will aid in the discovery of new screening models for top cardiovascular diseases.

| Disease | % of Total DALYs in India – IHME 2021 |
|---------|---------------------------------------|
| Ischemic Heart Disease | 7.8% |
| Rheumatic Heart Disease | 1.08% |
| Hypertensive Heart Disease | 0.60% |
| Cardiomyopathy and myocarditis | 0.31% |
| Hypertrophic cardiomyopathy | 1.1% |
| Atrial Fibrillation and Flutter | 0.16% |

*Source: Institute for Health Metrics and Evaluation. (n.d.). GBD Compare | India. % of total DALYs for both sexes across all age groups, from https://vizhub.healthdata.org/gbd-compare/india*

### Model Development

Our goal is to develop machine learning models for identifying cardiovascular diseases in patients with diabetes and hypertension, using our collected ECG dataset. We aim to explore:

- **Supervised machine learning methods**, particularly those based on Convolutional Neural Networks (CNNs), like the ResNet architecture that has seen success in detecting cardiovascular conditions based on single-lead ECG data¹
- **Transformer-based architectures** that can capture more complex temporal relationships that traditional CNN-based models might overlook²³⁴
- **Unsupervised representation learning techniques** to identify unknown disease markers by analyzing clusters in the data representations, as well as using the learned representations to improve the performance of supervised learning algorithms⁵

### Devices for Screening

The data asset will be used to develop AI models for low-cost sensors included in the study to help communities more readily screen for top 5 cardiovascular diseases:

1. **1-6 lead Mobile ECG Sensors** - AliveCor KardiaMobile 6L
2. **Eko Core 500**

### 1.2 Protocol

For detailed information on the approved protocol please refer to the IRB submission.

We hope to improve on a past population level study by increasing the quality of the asset created for model development and narrowing our scope to patients with additional hypertension or diabetes co-morbidities. Data will be collected with these high-level targets in mind:

| Metric | Target |
|--------|--------|
| Total Population Size | ~5,000 patients |
| Target Completeness Goal | 85% (30% improvement over TN study) |
| Target # of Sites | 3 Sites (CMC LCECU, CMC CHAD, Ranipet Cardiology Clinic) |

*The asset requirements are defined in the Data Dictionary and the Software Implementation Checklist in the Appendix.*

## Platform Components

The platform will support the collection, storage, and analysis of multi-modal health data to enable the development of machine learning models for cardiovascular disease screening. The system consists of four main components:

1. A data collection application for field use
2. A set of data collection integrations for labs and medical records systems
3. A secure cloud-based data hosting and model development platform
4. Reports to track collection progress

## Site Workflows

We plan to collect data from three sites:

- **Two primary and secondary-level healthcare facilities** (LCECU, CHAD), where the study population consists of individuals aged 30 years and older with a known diagnosis of diabetes and/or hypertension who present to the clinic with unknown cardiovascular disease status
- **One Cardiology clinic**, where we will target patients undergoing echocardiogram examinations who have a known diagnosis of diabetes and/or hypertension and a cardiac pathology relevant to this study

For each participant, a total of **13 tests** will be conducted—7 mobile-based and 6 gold-standard—along with a questionnaire.

### Data Collection Tests and Procedures

| Categories | # | Test | Tools | Data Collection Procedure | LCECU, CHAD | Cardiology Clinic |
|------------|---|------|-------|---------------------------|-------------|-------------------|
| **Mobile Test Measures** | | | | | | |
| | 1 | 1-6 lead Mobile ECG Sensors | AliveCor KardiaMobile 6L | Conduct mobile ECG sensor measurements, with sensor data streamed to the manufacturer's app and uploaded to the cloud. The data will then be retrieved from the cloud and integrated into the data platform via an API interface. | ✓ | ✓ |
| | 2 | Eko Core 500 | Eko Core 500 | | ✓ | ✓ |
| | 3 | Blood Pressure | Sphygmomanometer | Conduct the test and enter data into the study app. | ✓ | ✓ |
| | 4 | Pulse Rate, SpO2 | Pulse Oximeter | | ✓ | ✓ |
| | 5 | Respiratory Rate | Manual | | ✓ | ✓ |
| | 6 | Weight | Scale | | ✓ | ✓ |
| | 7 | Height | Measurement Tape | | ✓ | ✓ |
| **Gold Standard Measures** | | | | | | |
| | 8 | 12-lead ECG | 12 Lead ECG | Conduct the test. Transfer ECG trace data to CMC PACS system. | ✓ | Conduct the test if it has not already been performed at the cardiology clinic as part of the standard of care (within 3 days). Transfer ECG trace data to CMC PACS system. |
| | 9 | Echocardiogram | Echocardiogram | Conduct the test and transfer video and image data in DICOM format, along with the echo report, to the CMC PACS system. | ✓ | Conduct the test if it has not already been performed at the cardiology clinic as part of the standard of care (within 14 days). Transfer video and image data in DICOM format, along with the echo report, to the CMC PACS system. |
| | 10 | Hb | CMC Lab Equipment | Collect blood sample for testing (onsite or offsite). Transfer data from the CMC lab system to CMC EMR, then CMC Research Hub. | ✓ | Collect blood sample for testing if it has not already been performed at the cardiology clinic as part of the standard of care (within 3 months). Transfer data from the CMC lab system to CMC EMR, then CMC Research Hub. |
| | 11 | HbA1c | CMC Lab Equipment | | ✓ | ✓ |
| | 12 | Lipid panel (non-fasting) | CMC Lab Equipment | | ✓ | ✓ |
| | 13 | Serum Creatinine | CMC Lab Equipment | | ✓ | ✓ |
| **Questionnaires** | | | | | | |
| | N/A | Patient demographic, history, family history, risk factors and symptoms | Questionnaire with questions from the WHO STEPS questionnaire and the Rose Angina questionnaire | Enter the questionnaire directly in the study app | ✓ | ✓ |

### 1.3 Timeline

| Phase | Duration |
|-------|----------|
| Platform Development & Setup | May - Aug 2025 |
| Integration Testing | Sep - Oct 2025 |
| Data Collection Phase | Nov 2025 - Aug 2026 |
| Ongoing Maintenance & Study Close | Q3 2026 and beyond |