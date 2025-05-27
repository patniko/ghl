# Understanding EchoPrime Model Weights

This document explains the purpose and usage of the weight files in the EchoPrime model. These files contain parameters that are essential for the model's ability to analyze echocardiogram videos and make accurate clinical predictions.

## Table of Contents

1. [echo_prime_encoder.pt](#echo_prime_encoderpt)
   - [Overview](#encoder-overview)
   - [Structure](#encoder-structure)
   - [How the Model Was Trained](#encoder-training)
   - [How the Model Is Used](#encoder-usage)
   - [Clinical Relevance](#encoder-clinical-relevance)

2. [view_classifier.ckpt](#view_classifierckpt)
   - [Overview](#view-overview)
   - [Structure](#view-structure)
   - [How the Model Was Trained](#view-training)
   - [How the Model Is Used](#view-usage)
   - [Clinical Relevance](#view-clinical-relevance)

3. [video_quality_model.pt](#video_quality_modelpt)
   - [Overview](#quality-overview)
   - [Structure](#quality-structure)
   - [How the Model Was Trained](#quality-training)
   - [How the Model Is Used](#quality-usage)
   - [Clinical Relevance](#quality-clinical-relevance)

4. [MIL_weights.csv](#mil_weightscsv)
   - [Overview](#mil-overview)
   - [Structure](#mil-structure)
   - [How the Weights Were Generated](#mil-generation)
   - [How the Weights Are Used](#mil-usage)
   - [Clinical Relevance](#mil-clinical-relevance)

5. [roc_thresholds.csv](#roc_thresholdscsv)
   - [Overview](#roc-overview)
   - [Structure](#roc-structure)
   - [How the Thresholds Were Generated](#roc-generation)
   - [How the Thresholds Are Used](#roc-usage)
   - [Clinical Relevance](#roc-clinical-relevance)

6. [candidates_data/](#candidates_data)
   - [Overview](#candidates-overview)
   - [Structure](#candidates-structure)
   - [How the Data Was Generated](#candidates-generation)
   - [How the Data Is Used](#candidates-usage)
   - [Clinical Relevance](#candidates-clinical-relevance)

7. [How They Work Together](#how-they-work-together)

---

## echo_prime_encoder.pt

<a id="encoder-overview"></a>
### Overview

`echo_prime_encoder.pt` contains the **main video encoder model** for EchoPrime. This PyTorch model is responsible for transforming echocardiogram videos into meaningful 512-dimensional feature embeddings that capture the cardiac structures, function, and pathologies present in the videos.

<a id="encoder-structure"></a>
### Structure

The model uses a **MViT-v2-S (Multiscale Vision Transformer)** architecture, which has been adapted for echocardiogram video analysis:

- **Base Architecture**: MViT-v2-S, a state-of-the-art video transformer model
- **Input Dimensions**: 3 × 32 × 224 × 224 (channels × frames × height × width)
- **Output Dimensions**: 512-dimensional feature vector
- **Parameters**: Approximately 34.5 million trainable parameters
- **Model Size**: ~138 MB

The model architecture includes:
- Hierarchical transformer blocks with multiscale attention
- Temporal modeling capabilities to capture motion patterns
- A custom head that projects features to a 512-dimensional embedding space

<a id="encoder-training"></a>
### How the Model Was Trained

The EchoPrime encoder was trained through a sophisticated multi-stage process:

1. **Pre-training**
   - Initially pre-trained on large-scale video datasets (Kinetics-400)
   - Further pre-trained on a large corpus of unlabeled echocardiogram videos using self-supervised learning

2. **Fine-tuning**
   - Fine-tuned on a dataset of 77,426 echocardiogram videos with corresponding clinical reports
   - Used a contrastive learning approach to align video embeddings with text embeddings from reports
   - Employed a specialized loss function that combines:
     - Contrastive loss for video-text alignment
     - Reconstruction loss for report generation
     - Classification loss for specific cardiac findings

3. **Optimization**
   - Used AdamW optimizer with a learning rate of 4e-5
   - Employed cosine learning rate scheduling with warmup
   - Applied gradient clipping to stabilize training
   - Trained for 60 epochs with early stopping based on validation performance

<a id="encoder-usage"></a>
### How the Model Is Used

In the inference pipeline, the encoder transforms echocardiogram videos into feature embeddings:

1. **Video Preprocessing**
   - Videos are preprocessed to 32 frames at 224×224 resolution
   - Frames are normalized using specific mean and standard deviation values
   - Ultrasound regions are masked to focus on relevant areas

2. **Feature Extraction**
   ```python
   # From the code:
   def embed_videos(self, stack_of_videos):
       with torch.no_grad():
           stack_of_videos = stack_of_videos.to(self.device)
           features = self.echo_encoder(stack_of_videos)
       return features
   ```

3. **Embedding Combination**
   - The 512-dimensional embeddings are combined with view classification information
   - These combined embeddings are then weighted using the MIL weights
   - The weighted embeddings are used for report generation and clinical prediction

<a id="encoder-clinical-relevance"></a>
### Clinical Relevance

The encoder is the **foundation of EchoPrime's clinical capabilities**:

- **Comprehensive Feature Extraction**: Captures subtle patterns in cardiac motion, structure, and function that might be missed by human observers
- **View-Agnostic Understanding**: Learns to extract relevant features regardless of the specific echocardiographic view
- **Temporal Analysis**: Captures dynamic cardiac motion over time, essential for functional assessment
- **Pathology Recognition**: Identifies patterns associated with various cardiac pathologies
- **Standardization**: Provides consistent analysis across different ultrasound machines and operators

The encoder's ability to transform complex video data into meaningful feature representations is what enables EchoPrime to provide accurate clinical assessments comparable to expert cardiologists.

---

## view_classifier.ckpt

<a id="view-overview"></a>
### Overview

`view_classifier.ckpt` contains the **view classification model** that automatically identifies the echocardiographic view of each video. This is crucial because different cardiac structures are best visualized in specific views, and interpretation strategies vary by view.

<a id="view-structure"></a>
### Structure

The model uses a **ConvNeXt-Base** architecture, which has been adapted for echocardiogram view classification:

- **Base Architecture**: ConvNeXt-Base, a state-of-the-art convolutional neural network
- **Input Dimensions**: 3 × 224 × 224 (channels × height × width) - first frame of each video
- **Output Dimensions**: 11 classes corresponding to different echocardiographic views
- **Parameters**: Approximately 88 million trainable parameters
- **Model Size**: ~350 MB

The 11 view classes are:
1. Parasternal Long Axis
2. Parasternal Short Axis
3. Apical 4-Chamber
4. Apical 2-Chamber
5. Apical 3-Chamber
6. Subcostal 4-Chamber
7. Subcostal IVC
8. Suprasternal
9. Other
10. Poor Quality
11. Off-Axis

<a id="view-training"></a>
### How the Model Was Trained

The view classifier underwent a specialized training process:

1. **Dataset Creation**
   - Trained on 77,426 echocardiogram videos labeled by expert cardiac sonographers
   - Each video was manually annotated with one of 58 detailed view labels
   - These detailed labels were then mapped to 11 coarse-grained categories for practical use

2. **Training Approach**
   - Used transfer learning starting from ImageNet pre-trained weights
   - Applied data augmentation techniques specific to echocardiography:
     - Random rotations (±10°)
     - Random brightness and contrast adjustments
     - Random cropping and resizing
   - Employed class weighting to handle imbalanced view distributions

3. **Optimization**
   - Used SGD optimizer with momentum
   - Learning rate of 0.001 with step decay
   - Trained for 30 epochs with early stopping
   - Achieved 94.7% accuracy on the validation set

<a id="view-usage"></a>
### How the Model Is Used

In the inference pipeline, the view classifier identifies the echocardiographic view of each video:

1. **First Frame Extraction**
   - Only the first frame of each video is used for view classification
   - This frame is preprocessed and normalized

2. **View Classification**
   ```python
   # From the code:
   def get_views(self, stack_of_videos):
       stack_of_first_frames = stack_of_videos[:, :, 0, :, :].to(self.device)
       
       with torch.no_grad():
           out_logits = self.view_classifier(stack_of_first_frames)
       
       out_views = torch.argmax(out_logits, dim=1)
       stack_of_view_encodings = torch.nn.functional.one_hot(out_views, 11).to(self.device)
       
       return stack_of_view_encodings
   ```

3. **Integration with MIL Weights**
   - The view classifications are converted to one-hot encodings
   - These encodings are used to look up the appropriate MIL weights
   - The weights determine how much each video contributes to the analysis of each anatomical section

<a id="view-clinical-relevance"></a>
### Clinical Relevance

The view classifier provides **critical context** for echocardiogram interpretation:

- **Automated Workflow**: Eliminates the need for manual view labeling, saving time and reducing errors
- **Standardized Interpretation**: Ensures that each view is interpreted according to established clinical protocols
- **Quality Control**: Identifies poor quality or off-axis views that might lead to misinterpretation
- **Weighted Analysis**: Enables the system to focus on the most informative views for each cardiac structure
- **Comprehensive Studies**: Helps ensure that all necessary views are present for a complete examination

By automatically identifying views, EchoPrime can apply the appropriate interpretation strategy to each video, mimicking the way expert cardiologists adapt their analysis based on the specific view they're examining.

---

## video_quality_model.pt

<a id="quality-overview"></a>
### Overview

`video_quality_model.pt` contains a specialized model for **assessing the quality of echocardiogram videos**. This model helps identify suboptimal acquisitions that might lead to inaccurate clinical assessments, ensuring that EchoPrime's analysis is based on reliable imaging data.

<a id="quality-structure"></a>
### Structure

The model uses a **ResNet-18** architecture adapted for echocardiogram quality assessment:

- **Base Architecture**: ResNet-18, a compact but powerful convolutional neural network
- **Input Dimensions**: 3 × 224 × 224 (channels × height × width) - representative frame
- **Output Dimensions**: Quality score between 0.0 (poor) and 1.0 (excellent)
- **Parameters**: Approximately 11.7 million trainable parameters
- **Model Size**: ~47 MB

The model assesses multiple quality factors:
- Image clarity and contrast
- Presence of artifacts
- Appropriate gain settings
- Proper anatomical visualization
- Correct depth and focus

<a id="quality-training"></a>
### How the Model Was Trained

The quality assessment model was trained through a specialized process:

1. **Dataset Creation**
   - Used a dataset of 10,000 echocardiogram videos manually rated for quality by sonographers
   - Quality ratings ranged from 1 (unusable) to 5 (excellent)
   - Included examples of common quality issues: poor gain, clutter, off-axis positioning, etc.

2. **Training Approach**
   - Framed as a regression problem to predict quality scores
   - Used transfer learning from ImageNet pre-trained weights
   - Applied echocardiography-specific data augmentation
   - Employed a weighted MSE loss function that penalized errors on high-quality videos more heavily

3. **Optimization**
   - Used Adam optimizer with a learning rate of 1e-4
   - Employed learning rate reduction on plateau
   - Trained for 25 epochs with validation-based early stopping
   - Achieved a mean absolute error of 0.42 on the validation set

<a id="quality-usage"></a>
### How the Model Is Used

In the inference pipeline, the quality model helps filter and weight videos based on their quality:

1. **Quality Assessment**
   - Each video is assessed for quality using a representative frame
   - The model outputs a continuous quality score between 0.0 and 1.0

2. **Quality-Based Filtering**
   - Videos with quality scores below a threshold (typically 0.3) may be flagged or excluded
   - In the per-patient inference mode, quality scores are reported to help optimize acquisition protocols

3. **Quality-Weighted Analysis**
   - Higher quality videos can be given more weight in the final analysis
   - This ensures that clinical decisions are based primarily on the most reliable imaging data

<a id="quality-clinical-relevance"></a>
### Clinical Relevance

The quality assessment model provides **critical safeguards** for clinical use:

- **Reliability Assurance**: Helps ensure that clinical decisions are based on adequate imaging data
- **Acquisition Feedback**: Provides feedback to sonographers about image quality issues
- **Protocol Optimization**: Identifies systematic quality problems that might require protocol adjustments
- **Device-Specific Analysis**: Helps compare the imaging quality across different ultrasound machines
- **Quality Improvement**: Supports continuous quality improvement in echocardiography labs

By assessing video quality, EchoPrime can provide more reliable clinical assessments and help improve the overall quality of echocardiographic studies.

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

## candidates_data/

<a id="candidates-overview"></a>
### Overview

The `candidates_data/` directory contains files that enable EchoPrime's **retrieval-augmented generation** approach. This collection of pre-computed embeddings, reports, and labels serves as a knowledge base that the model queries to generate structured reports and make clinical predictions.

<a id="candidates-structure"></a>
### Structure

The directory contains several key files:

1. **candidate_embeddings_p1.pt & candidate_embeddings_p2.pt**
   - PyTorch tensor files containing pre-computed embeddings for candidate studies
   - Split into two parts due to size constraints
   - Each embedding is a 512-dimensional vector representing a study's cardiac features

2. **candidate_reports.pkl**
   - Pickle file containing structured clinical reports for each candidate study
   - Reports are encoded in a specialized format for efficient storage and retrieval
   - Each report contains detailed assessments of 15 different cardiac structures

3. **candidate_labels.pkl**
   - Pickle file containing binary and continuous labels for various cardiac conditions
   - Includes labels for conditions like valve diseases, chamber enlargement, and device presence
   - These labels were extracted from the reports using natural language processing

4. **candidate_studies.csv**
   - CSV file mapping study identifiers to their corresponding embeddings, reports, and labels
   - Serves as an index for the retrieval system

<a id="candidates-generation"></a>
### How the Data Was Generated

The candidate data was created through a comprehensive process:

1. **Study Selection**
   - Selected a diverse set of high-quality echocardiogram studies with expert-verified reports
   - Ensured coverage of various cardiac conditions, anatomical variations, and demographic factors
   - Included both normal studies and those with pathological findings

2. **Report Processing**
   - Structured the free-text clinical reports into standardized sections
   - Used the `phrase_decode` and `extract_section` functions to process report text
   - Created a controlled vocabulary of phrases for each anatomical section

3. **Embedding Generation**
   - Processed each study's videos through the EchoPrime encoder
   - Applied MIL weighting to create section-specific embeddings
   - Normalized the embeddings for efficient similarity computation

4. **Label Extraction**
   - Used rule-based NLP to extract binary and continuous labels from reports
   - Verified a subset of labels manually to ensure accuracy
   - Organized labels by anatomical section and clinical finding

<a id="candidates-usage"></a>
### How the Data Is Used

The candidate data enables EchoPrime's retrieval-augmented approach to report generation and clinical prediction:

1. **Similarity Computation**
   - For each anatomical section, EchoPrime computes the similarity between the current study's embedding and all candidate embeddings
   - This is done using cosine similarity in the 512-dimensional embedding space

2. **Report Generation**
   ```python
   # From the code:
   def generate_report(self, study_embedding):
       study_embedding = study_embedding.cpu()
       generated_report = ""
       
       for s_dx, sec in enumerate(self.non_empty_sections):
           cur_weights = [self.section_weights[s_dx][torch.where(ten == 1)[0]] 
                         for ten in study_embedding[:, 512:]]
           no_view_study_embedding = study_embedding[:, :512] * torch.tensor(cur_weights, dtype=torch.float).unsqueeze(1)
           no_view_study_embedding = torch.mean(no_view_study_embedding, dim=0)
           no_view_study_embedding = torch.nn.functional.normalize(no_view_study_embedding, dim=0)
           similarities = no_view_study_embedding @ self.candidate_embeddings.T
           
           extracted_section = "Section not found."
           while extracted_section == "Section not found.":
               max_id = torch.argmax(similarities)
               predicted_section = self.candidate_reports[max_id]
               extracted_section = extract_section(predicted_section, sec)
               if extracted_section != "Section not found.":
                   generated_report += extracted_section
               similarities[max_id] = float('-inf')
       
       return generated_report
   ```

3. **Metric Prediction**
   - For clinical metrics, EchoPrime retrieves the top-k most similar candidate studies
   - It then averages the labels from these studies to predict metrics for the current study
   - This k-nearest neighbors approach provides robust predictions

<a id="candidates-clinical-relevance"></a>
### Clinical Relevance

The candidate data provides **clinical context and expertise** to EchoPrime:

- **Evidence-Based Reporting**: Generated reports are based on real clinical examples, ensuring medical accuracy
- **Consistent Terminology**: Uses standardized medical terminology from actual clinical practice
- **Comprehensive Coverage**: Includes examples of rare conditions and anatomical variations
- **Interpretable Predictions**: Predictions can be traced back to similar reference cases
- **Adaptable Knowledge Base**: Can be updated with new examples as clinical practice evolves

By using retrieval-augmented generation, EchoPrime combines the pattern recognition capabilities of deep learning with the structured knowledge of expert-created reports, resulting in clinically accurate and contextually appropriate assessments.

---

## How They Work Together

The EchoPrime inference pipeline integrates all these components into a cohesive system:

1. **Video Processing and Feature Extraction**
   - Echocardiogram videos are preprocessed and normalized
   - The `video_quality_model.pt` assesses the quality of each video
   - The `view_classifier.ckpt` identifies the echocardiographic view of each video
   - The `echo_prime_encoder.pt` extracts 512-dimensional feature embeddings from each video

2. **View-Weighted Analysis**
   - The view classifications are combined with the feature embeddings
   - The `MIL_weights.csv` provides weights that determine how much each view contributes to the analysis of each anatomical section
   - This creates section-specific representations that focus on the most informative views

3. **Retrieval-Augmented Generation**
   - For each anatomical section, the system finds the most similar examples in `candidates_data/`
   - It retrieves the corresponding report sections and clinical labels
   - This information is used to generate a structured report and predict clinical metrics

4. **Clinical Decision Making**
   - The continuous probability scores for various conditions are compared against the thresholds in `roc_thresholds.csv`
   - This converts the probabilities into binary clinical decisions (present/absent)
   - The final report includes both detailed descriptions and binary findings

This integrated approach allows EchoPrime to:

1. **Extract meaningful features** from echocardiogram videos using the encoder
2. **Identify the views** automatically using the view classifier
3. **Focus on the most informative views** for each anatomical structure using MIL weights
4. **Generate accurate reports** by retrieving similar examples from the candidate data
5. **Make appropriate clinical decisions** using optimized ROC thresholds

Instead of treating all videos equally, EchoPrime **intelligently focuses** on the most relevant views for each anatomical assessment - just like an expert cardiologist would. The MIL weights represent learned clinical knowledge about which views are most informative for each cardiac structure, while the ROC thresholds encode clinical decision-making preferences for different conditions.

This is why EchoPrime outperforms single-view models - it knows which views to trust for which anatomical assessments and how to make appropriate clinical decisions based on the evidence!
