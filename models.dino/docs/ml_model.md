# ML Model

## Introduction

The convergence of artificial intelligence (AI) and healthcare has enabled sophisticated diagnostic systems capable of operating in low-resource environments. In cardiovascular care, 12-lead ECGs, echocardiograms, and blood work provide complementary insights into cardiac health, yet only limited modalities—such as 1–6 lead mobile ECGs—are available at the point of care. This disparity necessitates an architecture that distills the diagnostic power of multimodal data into a compact, edge-deployable model.

While there are self-supervised learning (SSL) frameworks for multimodal alignment, their success hinges on vast datasets (e.g., millions of samples). Given constrained data resources, we propose a three-stage SSL tailored to our ultimate task of triaging patients at point of care using 1–6 lead mobile ECG devices.

## Key Hypothesis

The key hypothesis that we test in this study is whether tracings from a mobile ECG device with limited leads (1 to 6-leads) can approximate the diagnostic power of a 12-lead ECG and echocardiogram when fused with other data. As a part of the study, we collect a comprehensive set of data modalities:

- ECG data from 1-6 lead mobile ECG devices
- ECG data from standard 12-lead ECG devices
- Echocardiogram data
- Blood work reports
- Accompanying cardiologist notes

The main challenge in utilizing this rich data is to develop an inference model that is small enough to be deployable on edge devices while also possessing high-level diagnostic reasoning typically only possible with larger models. A deep learning framework that is particularly suitable for such a task is that of teacher-student distillation models.

## Teacher-Student Distillation in Deep Learning

Knowledge distillation is a training paradigm in deep learning where a smaller, more efficient model (the student) learns to mimic the behavior of a larger, more powerful model (the teacher). This approach enables the deployment of lightweight, high-performance models on resource-constrained devices while retaining much of the performance of the larger model.

Instead of training the student model directly on hard labels (e.g., class IDs), it learns from the soft targets or intermediate representations produced by the teacher. While handcrafted features (e.g., QRS duration) or simple labels (e.g., "normal/abnormal") discard signal richness, these soft targets contain richer information, including class probabilities and inter-class relationships, which help the student generalize better. The paradigm also scales to new diseases without re-engineering features.

## Overview of Proposed Pipeline

To bridge the gap between rich multimodal training data (12-lead ECG, echocardiograms, blood work) and sparse inference-time inputs (1–6 lead mobile ECG), we propose a three-stage training pipeline:

1. **Modality-Specific Pretraining**: Leverage large public datasets to train robust ECG and echocardiogram encoders without requiring labeled data.

2. **Cross-Modal Self-Supervised Learning (SSL)**: Align pretrained ECG and echo representations using 20k unlabeled patient records from our internal dataset (CMI dataset + TN-JPAL dataset).

3. **Supervised Fine-Tuning**: Train a triage classifier on 5k high-quality labeled examples to specialize the model for risk stratification.

### Modality-Specific Pretraining

This approach begins with modality-specific pretraining on large-scale public repositories such as PTB-XL or Chapman-Shaoxing for ECG data and EchoNet-Dynamic for echocardiograms, where self-supervised learning techniques like contrastive learning and masked autoencoding extract robust, generalizable features without requiring labeled data. Public datasets provide orders of magnitude more samples than the internal dataset, improving feature robustness.

### Cross-Modal Alignment

The next step is to align the ECG and echocardiogram embeddings to capture clinically meaningful relationships. To align the embeddings, the pretrained encoders then undergo cross-modal alignment using our internal dataset of 20,000 unlabeled patient records (12-lead ECG + echo videos + blood work), where advanced techniques including:

- **Temporal contrastive learning**: Force ECG R-peak embeddings to align with echo keyframes (e.g., end-systole)
- **Joint masked modeling**: Randomly mask ECG leads and echo frames → predict missing data using cross-attention

These techniques exploit the natural physiological synchrony between electrical cardiac activity (ECG) and mechanical function (echocardiogram). This step also requires no manual labeling and requires only synchronized ECG-echo pairs to exploit the natural physiological alignments (e.g., ECG electrical events ↔ echo mechanical events).

### Supervised Fine-Tuning and Knowledge Distillation

The final stage is that of supervised fine-tuning on the very high-quality 5k labeled dataset as well as distilling the knowledge into a lightweight student model that takes 1–6 lead ECG as input.

## Data and Compute Requirements

Raw echocardiograms (DICOM) dominate storage needs. Assuming 20k patient records, each record contains:

- 12-lead ECG: ~5 MB (high-resolution waveforms with headers)
- Echocardiogram: ~200-500 MB
- Bloodwork reports: Negligible (text/structured data)

This brings the total storage requirement for 20k patients to 4-10TB. Using lossless compression (e.g., JPEG2000), could reduce this by 30-50%.

The storage for the model embeddings (assuming 1024 dimensional embeddings of float 32) would require about 8 GB of storage, but storing intermediate feature caches during training, and multiple embedding versions during experimentation will necessitate allocating up to 2 TB.

Computational requirements are substantial but justified, with an estimated 1,000 A100 GPU hours (approximately $10,000 in cloud costs) distributed across the various training phases. Strategic optimizations like mixed precision training and gradient checkpointing can significantly reduce these resource demands without compromising model performance.

## Performance Targets

The proposed system is designed to achieve:

- Target AUC of 0.88-0.92 for the comprehensive teacher model
- Target AUC of 0.83-0.87 for the deployable student model

For reference:
- ECG-only models (e.g., CardioGPT) achieve an AUC of 0.82-0.87
- Echo-only models (e.g., EchoPrime) achieve an AUC of 0.85-0.92
- Multimodal fusion typically adds +0.03-0.05 AUC
- Distillation could drop 3-5% AUC when going from 12-lead → 1-6 lead ECG

Inference latency is kept under 50 milliseconds for edge device deployment.

## Conclusion and Recommendations

This approach offers several compelling advantages over conventional end-to-end supervised training:

- By decoupling the feature learning process from task-specific fine-tuning, we reduce dependence on expensive labeled data by 75% while simultaneously improving generalization to rare conditions through the rich representations learned during self-supervised pretraining.
- The pipeline's modular design also facilitates ongoing improvement, as newer or larger pretraining datasets can be incorporated without disrupting the overall architecture.

Critical to success will be:
- Ensuring precise temporal alignment between ECG and echocardiogram data
- Maintaining rigorous label quality standards for the 5,000 fine-tuning examples
