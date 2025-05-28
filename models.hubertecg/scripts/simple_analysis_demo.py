#!/usr/bin/env python3
"""
Simple demonstration of how to interpret HuBERT-ECG inference results.

This script provides a basic analysis without requiring additional dependencies,
showing you how to load and understand your extracted features.

Usage:
    python scripts/simple_analysis_demo.py
"""

import os
import sys
import numpy as np
import pandas as pd


def load_and_analyze_results(results_dir='results'):
    """Load and analyze HuBERT-ECG inference results."""
    print("HuBERT-ECG Results Analysis")
    print("=" * 50)
    
    # Load metadata
    metadata_path = os.path.join(results_dir, 'inference_results.csv')
    if not os.path.exists(metadata_path):
        print(f"❌ No results found at {metadata_path}")
        print("Please run 'make inference' first to generate results.")
        return
    
    df = pd.read_csv(metadata_path)
    successful_results = df[df['status'] == 'success']
    
    print(f"📊 Results Overview:")
    print(f"  • Total files processed: {len(df)}")
    print(f"  • Successful inferences: {len(successful_results)}")
    print(f"  • Success rate: {len(successful_results)/len(df)*100:.1f}%")
    print()
    
    if len(successful_results) == 0:
        print("❌ No successful results to analyze.")
        return
    
    # Analyze each patient's features
    print(f"🔍 Feature Analysis:")
    print()
    
    for _, row in successful_results.iterrows():
        patient = row['patient']
        features_file = row['features_file']
        features_path = os.path.join(results_dir, features_file)
        
        if os.path.exists(features_path):
            # Load features
            features = np.load(features_path)
            
            # Remove batch dimension if present
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)  # Shape: (187, 768)
            
            print(f"Patient: {patient}")
            print(f"  📁 File: {row['filename']}")
            print(f"  📏 Original ECG shape: {row['original_shape']}")
            print(f"  🔄 Preprocessed shape: {row['preprocessed_shape']}")
            print(f"  🧠 Features shape: {features.shape}")
            
            # Calculate basic statistics
            mean_val = np.mean(features)
            std_val = np.std(features)
            min_val = np.min(features)
            max_val = np.max(features)
            zero_ratio = np.mean(features == 0)
            
            print(f"  📈 Statistics:")
            print(f"    • Mean: {mean_val:.6f}")
            print(f"    • Std: {std_val:.6f}")
            print(f"    • Range: [{min_val:.6f}, {max_val:.6f}]")
            print(f"    • Sparsity (zero ratio): {zero_ratio:.4f}")
            
            # Temporal analysis
            if len(features.shape) == 2:  # (time_steps, features)
                temporal_means = np.mean(features, axis=1)  # Mean across features for each time step
                temporal_stds = np.std(features, axis=1)    # Std across features for each time step
                
                print(f"  ⏱️  Temporal patterns:")
                print(f"    • Time steps: {features.shape[0]}")
                print(f"    • Feature dimensions: {features.shape[1]}")
                print(f"    • Temporal mean range: [{np.min(temporal_means):.6f}, {np.max(temporal_means):.6f}]")
                print(f"    • Temporal variability: {np.mean(temporal_stds):.6f} ± {np.std(temporal_stds):.6f}")
                
                # Find most active time periods
                most_active_time = np.argmax(temporal_means)
                least_active_time = np.argmin(temporal_means)
                print(f"    • Most active time step: {most_active_time} (value: {temporal_means[most_active_time]:.6f})")
                print(f"    • Least active time step: {least_active_time} (value: {temporal_means[least_active_time]:.6f})")
                
                # Feature importance (simple variance-based)
                feature_variances = np.var(features, axis=0)
                top_features = np.argsort(feature_variances)[-5:]  # Top 5 most variable features
                print(f"    • Top 5 most variable features: {top_features.tolist()}")
                print(f"    • Their variances: {feature_variances[top_features]}")
            
            print()
    
    # Cross-patient comparison if multiple patients
    if len(successful_results) > 1:
        print(f"🔗 Cross-Patient Analysis:")
        print()
        
        # Load all features for comparison
        all_features = {}
        for _, row in successful_results.iterrows():
            patient = row['patient']
            features_file = row['features_file']
            features_path = os.path.join(results_dir, features_file)
            
            if os.path.exists(features_path):
                features = np.load(features_path)
                if features.ndim == 3 and features.shape[0] == 1:
                    features = features.squeeze(0)
                all_features[patient] = features
        
        # Calculate pairwise similarities
        patients = list(all_features.keys())
        print(f"Comparing {len(patients)} patients:")
        
        for i, patient1 in enumerate(patients):
            for j, patient2 in enumerate(patients):
                if i < j:  # Only upper triangle
                    feat1 = all_features[patient1].flatten()
                    feat2 = all_features[patient2].flatten()
                    
                    # Cosine similarity
                    dot_product = np.dot(feat1, feat2)
                    norm1 = np.linalg.norm(feat1)
                    norm2 = np.linalg.norm(feat2)
                    cosine_sim = dot_product / (norm1 * norm2)
                    
                    # Euclidean distance (normalized)
                    euclidean_dist = np.linalg.norm(feat1 - feat2)
                    
                    print(f"  {patient1} ↔ {patient2}:")
                    print(f"    • Cosine similarity: {cosine_sim:.4f}")
                    print(f"    • Euclidean distance: {euclidean_dist:.2f}")
                    
                    # Interpretation
                    if cosine_sim > 0.9:
                        similarity_desc = "Very similar"
                    elif cosine_sim > 0.7:
                        similarity_desc = "Similar"
                    elif cosine_sim > 0.5:
                        similarity_desc = "Moderately similar"
                    else:
                        similarity_desc = "Different"
                    
                    print(f"    • Interpretation: {similarity_desc} cardiac patterns")
                    print()
    
    # Summary and recommendations
    print(f"💡 Key Insights:")
    print(f"  • Each patient has 187 time steps × 768 features = 143,616 total features")
    print(f"  • Features represent learned cardiac representations from HuBERT-ECG")
    print(f"  • Temporal patterns show how cardiac features evolve over the ECG sequence")
    print(f"  • Feature variability indicates discriminative cardiac patterns")
    print()
    
    print(f"🚀 Next Steps:")
    print(f"  1. Use these features for classification tasks (e.g., arrhythmia detection)")
    print(f"  2. Apply clustering to identify patient subgroups")
    print(f"  3. Correlate features with clinical outcomes")
    print(f"  4. Analyze temporal patterns for rhythm analysis")
    print(f"  5. Use feature importance for biomarker discovery")
    print()
    
    print(f"📁 Your results are saved in:")
    print(f"  • Feature files: {results_dir}/PATIENT*/")
    print(f"  • Metadata: {results_dir}/inference_results.csv")
    print()
    
    print(f"🔧 For advanced analysis, install additional dependencies:")
    print(f"  poetry add matplotlib seaborn scikit-learn plotly")
    print(f"  Then run: python scripts/analyze_features.py")


def main():
    """Main function."""
    results_dir = 'results'
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    
    load_and_analyze_results(results_dir)


if __name__ == "__main__":
    main()
