# Understanding EchoPrime Model Weights

This document explains the purpose and usage of two critical files in the EchoPrime model: `MIL_weights.csv` and `roc_thresholds.csv`. These files contain parameters that are essential for the model's ability to analyze echocardiogram videos and make accurate clinical predictions.

## Table of Contents

1. [MIL_weights.csv](#mil_weightscsv)
   - [Overview](#mil-overview)
   - [Structure](#mil-structure)
   - [How the Weights Were Generated](#mil-generation)
   - [How the Weights Are Used](#mil-usage)
   - [Clinical Relevance](#mil-clinical-relevance)

2. [roc_thresholds.csv](#roc_thresholdscsv)
   - [Overview](#roc-overview)
   - [Structure](#roc-structure)
   - [How the Thresholds Were Generated](#roc-generation)
   - [How the Thresholds Are Used](#roc-usage)
   - [Clinical Relevance](#roc-clinical-relevance)

3. [How They Work Together](#how-they-work-together)

---

## MIL_weights.csv

<a id="mil-overview"></a>
### Overview

`MIL_weights.csv` contains **Multiple Instance Learning (MIL) weights** that are crucial for the EchoPrime model's ability to focus on relevant anatomical sections when analyzing different echocardiogram views. These weights tell the model which echocardiographic views are most informative for analyzing specific cardiac structures.

<a id="mil-structure"></a>
### Structure

The file is structured as a matrix with:

- **Rows**: 15 anatomical sections (Left Ventricle, Right Ventricle, Left Atrium, Right Atrium, Mitral Valve, Aortic Valve, etc.)
- **Columns**: 11 different echocardiogram views (A2C, A3C, A4C, A5C, Apical_Doppler, etc.)
- **Values**: Weights between 0.0 and 1.0 indicating how relevant each view is for analyzing each anatomical section

Example from the file:
```
Section,A2C,A3C,A4C,A5C,Apical_Doppler,Doppler_Parasternal_Long,Doppler_Parasternal_Short,Parasternal_Long,Parasternal_Short,SSN,Subcostal
Left Ventricle,1.0,0.3220147,0.6511186,0.0,0.8786839,0.42251515,0.5776004,0.48847628,0.7149652,0.80168504,0.6545674
```

<a id="mil-generation"></a>
### How the Weights Were Generated

The MIL weights were **learned automatically** through a sophisticated training process:

1. **View Classification Training**
   - First, a view classifier was trained on 77,426 echocardiogram videos labeled by cardiac sonographers
   - This classifier learned to distinguish between 58 different standard echocardiographic views
   - Used a ConvNext-Base architecture trained on 224x224 images

2. **Multiple Instance Learning (MIL) Training**
   - Used attention-based deep multiple instance learning to automatically learn the importance weights
   - For each anatomical section, the model learned which views are most informative
   - The training process:
     - Combined video embeddings with view classifications
     - Used ground truth labels from structured report sections
     - Backpropagated loss to update the MIL attention weights

3. **Clinical Validation**
   - The learned weights were compared against expert cardiologist opinions
   - Three independent cardiologists manually assigned importance to each view for each anatomical structure
   - The model's learned weights closely matched clinical expert consensus (as shown in Figure 3D of the research paper)

<a id="mil-usage"></a>
### How the Weights Are Used

In the inference pipeline, the MIL weights serve as **attention weights** that tell the model how much to focus on each echocardiogram view when analyzing specific anatomical structures:

1. **View Classification**: Each video gets classified into one of 11 views (A2C, A3C, A4C, etc.)

2. **Weight Lookup**: For each anatomical section (like "Left Ventricle"), the model looks up the corresponding weights:
   ```
   Left Ventricle weights: [1.0, 0.32, 0.65, 0.0, 0.88, 0.42, 0.58, 0.49, 0.71, 0.80, 0.65]
                           A2C  A3C   A4C  A5C  Apical_Doppler  etc...
   ```

3. **Weighted Video Embeddings**: The model multiplies each video's embedding by its corresponding weight:
   ```python
   # From the code:
   cur_weights = [self.section_weights[s_dx][torch.where(ten == 1)[0]] 
                  for ten in study_embedding[:, 512:]]
   no_view_study_embedding = study_embedding[:, :512] * torch.tensor(cur_weights, dtype=torch.float).unsqueeze(1)
   ```

4. **Weighted Average**: All weighted embeddings are averaged to create a section-specific representation

<a id="mil-clinical-relevance"></a>
### Clinical Relevance

These weights encode **clinical expertise**:

- **1.0**: "Pay maximum attention to this view for this anatomy"
- **0.0**: "Completely ignore this view for this anatomy" 
- **0.5-0.8**: "This view provides some useful information"
- **0.1-0.3**: "This view provides minimal information"

Examples:
- **Left Ventricle**: Apical views (A2C, A4C) get high weights because they show LV function best
- **Right Ventricle**: Subcostal and RV-focused views get high weights
- **Mitral Valve**: Apical views get high weights because they show mitral valve clearly
- **Aortic Valve**: A5C gets maximum weight (1.0) because it's specifically designed to show aortic valve

---

## roc_thresholds.csv

<a id="roc-overview"></a>
### Overview

`roc_thresholds.csv` contains **ROC (Receiver Operating Characteristic) curve-derived thresholds** for binary classification of various cardiac conditions and findings. These thresholds are used to convert the model's continuous probability outputs into binary predictions (present/absent) for each cardiac finding.

<a id="roc-structure"></a>
### Structure

The file is structured as a simple table with:

- **feature**: 16 different cardiac conditions/findings (pacemaker, impella, tavr, aortic_stenosis, etc.)
- **threshold**: Optimal decision thresholds (ranging from 0.04 to 1.06) for classifying each condition

Example from the file:
```
,feature,threshold
0,pacemaker,0.1
1,impella,0.16
2,tavr,0.76
```

<a id="roc-generation"></a>
### How the Thresholds Were Generated

The ROC thresholds were derived through **empirical optimization** on validation data:

1. **Model Prediction Generation**
   - EchoPrime generated continuous probability scores (0.0 to 1.0) for each cardiac condition
   - These probabilities came from the retrieval-augmented interpretation process

2. **ROC Curve Analysis**
   - For each of the 16 cardiac conditions, they:
     - Plotted ROC curves using validation data with known ground truth labels
     - Calculated sensitivity and specificity at different threshold values
     - Found the optimal threshold that maximized the Youden Index (sensitivity + specificity - 1)

3. **Threshold Selection Strategy**
   - **Low thresholds** (like mitral regurgitation = 0.06): High sensitivity needed - don't want to miss cases
   - **High thresholds** (like atrial septum hypertrophy = 1.06): High specificity needed - avoid false positives
   - The specific values reflect the clinical importance and prevalence of each condition

4. **Validation Process**
   - Thresholds were optimized on internal validation data from Cedars-Sinai
   - Then tested on external validation data from Stanford Healthcare
   - This ensures the thresholds generalize across different healthcare systems

<a id="roc-usage"></a>
### How the Thresholds Are Used

The ROC thresholds serve as **decision boundaries** that convert EchoPrime's continuous probability outputs into binary clinical decisions:

1. **EchoPrime generates probabilities**: For each cardiac condition, the model outputs a continuous probability score between 0.0 and 1.0
   - Example: For a patient, the model might predict:
     - `aortic_stenosis`: 0.85 probability
     - `mitral_regurgitation`: 0.03 probability
     - `pacemaker`: 0.95 probability

2. **Thresholds convert to binary decisions**: Each probability is compared against its specific threshold:
   - `aortic_stenosis`: 0.85 > 0.78 threshold → **POSITIVE** (stenosis present)
   - `mitral_regurgitation`: 0.03 < 0.06 threshold → **NEGATIVE** (no significant regurgitation)
   - `pacemaker`: 0.95 > 0.1 threshold → **POSITIVE** (pacemaker present)

<a id="roc-clinical-relevance"></a>
### Clinical Relevance

These thresholds essentially encode **clinical decision-making preferences**:

- **Conservative conditions** (low threshold): "When in doubt, flag it for review"
- **Intervention-requiring conditions** (high threshold): "Only flag if very confident"

Examples:
- **Low Thresholds (High Sensitivity)**:
  - `mitral_regurgitation`: 0.06 - Don't want to miss valve problems
  - `rv_systolic_function_depressed`: 0.04 - Critical for heart failure management

- **High Thresholds (High Specificity)**:
  - `atrial_septum_hypertrophy`: 1.06 - Avoid false positives for this specific finding
  - `aortic_stenosis`: 0.78 - Need high confidence before suggesting valve intervention

- **Medium Thresholds (Balanced)**:
  - `pacemaker`: 0.1 - Usually obvious on imaging, low threshold is fine
  - `tricuspid_valve_regurgitation`: 0.26 - Balanced approach for common finding

---

## How They Work Together

In the EchoPrime inference pipeline:

1. **MIL weights** help the model focus on the most relevant views when analyzing each anatomical section
   - This creates a weighted representation of each anatomical structure based on all available videos

2. The model generates probability scores for various cardiac conditions using this weighted representation
   - This is done through retrieval-augmented interpretation

3. **ROC thresholds** convert these probabilities into final binary predictions for clinical findings
   - This provides clear yes/no decisions for clinical use

This two-stage approach allows EchoPrime to provide both detailed anatomical analysis (using MIL weights) and reliable clinical predictions (using ROC thresholds) from multi-view echocardiogram data.

Instead of treating all videos equally, EchoPrime **intelligently focuses** on the most relevant views for each anatomical assessment - just like an expert cardiologist would. The MIL weights represent learned clinical knowledge about which views are most informative for each cardiac structure, while the ROC thresholds encode clinical decision-making preferences for different conditions.

This is why EchoPrime outperforms single-view models - it knows which views to trust for which anatomical assessments and how to make appropriate clinical decisions based on the evidence!
