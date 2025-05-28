#!/usr/bin/env python3
"""
Comprehensive analysis script for HuBERT-ECG inference results.

This script analyzes the extracted features from HuBERT-ECG inference,
providing detailed statistics, comparisons, and insights.

Usage:
    python scripts/analyze_features.py [--results_dir RESULTS_DIR] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class HuBERTFeatureAnalyzer:
    """Comprehensive analyzer for HuBERT-ECG features."""
    
    def __init__(self, results_dir='results', output_dir='analysis_output'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.features_data = {}
        self.metadata = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load all feature files and metadata."""
        print("Loading HuBERT-ECG inference results...")
        
        # Load metadata
        metadata_path = os.path.join(self.results_dir, 'inference_results.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)
        successful_results = self.metadata[self.metadata['status'] == 'success']
        
        print(f"Found {len(successful_results)} successful inference results")
        
        # Load feature files
        for _, row in successful_results.iterrows():
            patient = row['patient']
            features_file = row['features_file']
            features_path = os.path.join(self.results_dir, features_file)
            
            if os.path.exists(features_path):
                features = np.load(features_path)
                self.features_data[patient] = {
                    'features': features,
                    'filename': row['filename'],
                    'original_shape': row['original_shape'],
                    'features_shape': row['features_shape']
                }
                print(f"  Loaded {patient}: {features.shape}")
            else:
                print(f"  Warning: Features file not found: {features_path}")
        
        if not self.features_data:
            raise ValueError("No feature files could be loaded")
            
        print(f"Successfully loaded features for {len(self.features_data)} patients")
        
    def analyze_basic_statistics(self):
        """Analyze basic statistics of the features."""
        print("\nAnalyzing basic feature statistics...")
        
        stats_data = []
        
        for patient, data in self.features_data.items():
            features = data['features']
            
            # Remove batch dimension if present
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)  # Shape: (187, 768)
            
            stats_data.append({
                'patient': patient,
                'shape': features.shape,
                'mean': np.mean(features),
                'std': np.std(features),
                'min': np.min(features),
                'max': np.max(features),
                'median': np.median(features),
                'q25': np.percentile(features, 25),
                'q75': np.percentile(features, 75),
                'zero_ratio': np.mean(features == 0),
                'nan_count': np.sum(np.isnan(features)),
                'inf_count': np.sum(np.isinf(features))
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, 'feature_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"Basic statistics saved to: {stats_path}")
        
        # Print summary
        print("\nFeature Statistics Summary:")
        print("=" * 50)
        for _, row in stats_df.iterrows():
            print(f"{row['patient']}:")
            print(f"  Shape: {row['shape']}")
            print(f"  Mean: {row['mean']:.6f}, Std: {row['std']:.6f}")
            print(f"  Range: [{row['min']:.6f}, {row['max']:.6f}]")
            print(f"  Zero ratio: {row['zero_ratio']:.4f}")
            if row['nan_count'] > 0 or row['inf_count'] > 0:
                print(f"  Issues: {row['nan_count']} NaNs, {row['inf_count']} Infs")
            print()
        
        return stats_df
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns across the 187 time steps."""
        print("Analyzing temporal patterns...")
        
        # Prepare data for temporal analysis
        temporal_data = {}
        
        for patient, data in self.features_data.items():
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)  # Shape: (187, 768)
            
            # Calculate statistics across feature dimensions for each time step
            temporal_data[patient] = {
                'mean_per_timestep': np.mean(features, axis=1),  # (187,)
                'std_per_timestep': np.std(features, axis=1),    # (187,)
                'max_per_timestep': np.max(features, axis=1),    # (187,)
                'min_per_timestep': np.min(features, axis=1),    # (187,)
            }
        
        # Create temporal analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Temporal Feature Analysis Across 187 Time Steps', fontsize=16)
        
        metrics = ['mean_per_timestep', 'std_per_timestep', 'max_per_timestep', 'min_per_timestep']
        titles = ['Mean Features', 'Std Features', 'Max Features', 'Min Features']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            for patient, data in temporal_data.items():
                ax.plot(data[metric], label=patient, alpha=0.8, linewidth=2)
            
            ax.set_title(title)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Feature Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        temporal_plot_path = os.path.join(self.output_dir, 'temporal_analysis.png')
        plt.savefig(temporal_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Temporal analysis plot saved to: {temporal_plot_path}")
        
        # Save temporal data
        temporal_df_data = []
        for patient, data in temporal_data.items():
            for t in range(len(data['mean_per_timestep'])):
                temporal_df_data.append({
                    'patient': patient,
                    'timestep': t,
                    'mean': data['mean_per_timestep'][t],
                    'std': data['std_per_timestep'][t],
                    'max': data['max_per_timestep'][t],
                    'min': data['min_per_timestep'][t]
                })
        
        temporal_df = pd.DataFrame(temporal_df_data)
        temporal_csv_path = os.path.join(self.output_dir, 'temporal_patterns.csv')
        temporal_df.to_csv(temporal_csv_path, index=False)
        print(f"Temporal patterns data saved to: {temporal_csv_path}")
        
        return temporal_data
    
    def analyze_feature_distributions(self):
        """Analyze the distribution of feature values."""
        print("Analyzing feature distributions...")
        
        # Create distribution plots
        n_patients = len(self.features_data)
        fig, axes = plt.subplots(n_patients, 2, figsize=(12, 4*n_patients))
        if n_patients == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Feature Value Distributions', fontsize=16)
        
        for i, (patient, data) in enumerate(self.features_data.items()):
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            
            # Flatten features for distribution analysis
            flat_features = features.flatten()
            
            # Histogram
            axes[i, 0].hist(flat_features, bins=50, alpha=0.7, density=True)
            axes[i, 0].set_title(f'{patient} - Feature Distribution')
            axes[i, 0].set_xlabel('Feature Value')
            axes[i, 0].set_ylabel('Density')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Box plot
            axes[i, 1].boxplot(flat_features, vert=True)
            axes[i, 1].set_title(f'{patient} - Feature Box Plot')
            axes[i, 1].set_ylabel('Feature Value')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        dist_plot_path = os.path.join(self.output_dir, 'feature_distributions.png')
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Distribution plots saved to: {dist_plot_path}")
    
    def compare_patients(self):
        """Compare features across patients."""
        print("Comparing patients...")
        
        # Prepare data for comparison
        patient_names = list(self.features_data.keys())
        n_patients = len(patient_names)
        
        # Calculate pairwise similarities
        similarities = {}
        distances = {}
        
        # Get flattened features for each patient
        patient_features = {}
        for patient, data in self.features_data.items():
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            patient_features[patient] = features.flatten()
        
        # Calculate similarities and distances
        for i, patient1 in enumerate(patient_names):
            for j, patient2 in enumerate(patient_names):
                if i <= j:  # Only calculate upper triangle
                    feat1 = patient_features[patient1].reshape(1, -1)
                    feat2 = patient_features[patient2].reshape(1, -1)
                    
                    # Cosine similarity
                    cos_sim = cosine_similarity(feat1, feat2)[0, 0]
                    similarities[f"{patient1}_vs_{patient2}"] = cos_sim
                    
                    # Euclidean distance
                    eucl_dist = euclidean_distances(feat1, feat2)[0, 0]
                    distances[f"{patient1}_vs_{patient2}"] = eucl_dist
        
        # Create similarity matrix
        similarity_matrix = np.eye(n_patients)
        distance_matrix = np.zeros((n_patients, n_patients))
        
        for i, patient1 in enumerate(patient_names):
            for j, patient2 in enumerate(patient_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    distance_matrix[i, j] = 0.0
                else:
                    key1 = f"{patient1}_vs_{patient2}"
                    key2 = f"{patient2}_vs_{patient1}"
                    
                    if key1 in similarities:
                        similarity_matrix[i, j] = similarities[key1]
                        distance_matrix[i, j] = distances[key1]
                    elif key2 in similarities:
                        similarity_matrix[i, j] = similarities[key2]
                        distance_matrix[i, j] = distances[key2]
        
        # Plot similarity and distance matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Similarity heatmap
        sns.heatmap(similarity_matrix, annot=True, fmt='.4f', 
                   xticklabels=patient_names, yticklabels=patient_names,
                   cmap='viridis', ax=axes[0])
        axes[0].set_title('Patient Similarity (Cosine)')
        
        # Distance heatmap
        sns.heatmap(distance_matrix, annot=True, fmt='.2f',
                   xticklabels=patient_names, yticklabels=patient_names,
                   cmap='viridis_r', ax=axes[1])
        axes[1].set_title('Patient Distance (Euclidean)')
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(self.output_dir, 'patient_comparison.png')
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Patient comparison plot saved to: {comparison_plot_path}")
        
        # Save comparison data
        comparison_data = []
        for key, sim in similarities.items():
            dist = distances[key]
            patients = key.split('_vs_')
            comparison_data.append({
                'patient1': patients[0],
                'patient2': patients[1],
                'cosine_similarity': sim,
                'euclidean_distance': dist
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv_path = os.path.join(self.output_dir, 'patient_similarities.csv')
        comparison_df.to_csv(comparison_csv_path, index=False)
        print(f"Patient comparison data saved to: {comparison_csv_path}")
        
        return comparison_df
    
    def dimensionality_reduction_analysis(self):
        """Perform dimensionality reduction analysis."""
        print("Performing dimensionality reduction analysis...")
        
        # Prepare data
        all_features = []
        patient_labels = []
        
        for patient, data in self.features_data.items():
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)  # Shape: (187, 768)
            
            # Use mean pooling across time steps to get one vector per patient
            pooled_features = np.mean(features, axis=0)  # Shape: (768,)
            all_features.append(pooled_features)
            patient_labels.append(patient)
        
        all_features = np.array(all_features)  # Shape: (n_patients, 768)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(all_features)
        
        # PCA
        pca = PCA(n_components=min(len(patient_labels), 10))
        features_pca = pca.fit_transform(features_scaled)
        
        # t-SNE (if we have enough samples)
        features_tsne = None
        if len(patient_labels) >= 3:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(len(patient_labels)-1, 30))
            features_tsne = tsne.fit_transform(features_scaled)
        
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA plot
        if features_pca.shape[1] >= 2:
            for i, patient in enumerate(patient_labels):
                axes[0].scatter(features_pca[i, 0], features_pca[i, 1], 
                              label=patient, s=100, alpha=0.8)
                axes[0].annotate(patient, (features_pca[i, 0], features_pca[i, 1]),
                               xytext=(5, 5), textcoords='offset points')
            
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[0].set_title('PCA of HuBERT-ECG Features')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # t-SNE plot
        if features_tsne is not None:
            for i, patient in enumerate(patient_labels):
                axes[1].scatter(features_tsne[i, 0], features_tsne[i, 1],
                              label=patient, s=100, alpha=0.8)
                axes[1].annotate(patient, (features_tsne[i, 0], features_tsne[i, 1]),
                               xytext=(5, 5), textcoords='offset points')
            
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            axes[1].set_title('t-SNE of HuBERT-ECG Features')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Not enough samples\nfor t-SNE', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('t-SNE (Not Available)')
        
        plt.tight_layout()
        dimred_plot_path = os.path.join(self.output_dir, 'dimensionality_reduction.png')
        plt.savefig(dimred_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Dimensionality reduction plot saved to: {dimred_plot_path}")
        
        # Save PCA results
        pca_data = []
        for i, patient in enumerate(patient_labels):
            pca_row = {'patient': patient}
            for j in range(min(features_pca.shape[1], 10)):
                pca_row[f'PC{j+1}'] = features_pca[i, j]
            pca_data.append(pca_row)
        
        pca_df = pd.DataFrame(pca_data)
        pca_csv_path = os.path.join(self.output_dir, 'pca_results.csv')
        pca_df.to_csv(pca_csv_path, index=False)
        print(f"PCA results saved to: {pca_csv_path}")
        
        # Print PCA explained variance
        print(f"PCA Explained Variance Ratios:")
        for i, ratio in enumerate(pca.explained_variance_ratio_[:5]):
            print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
        
        return pca, features_pca, features_tsne
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("Generating summary report...")
        
        report_lines = []
        report_lines.append("HuBERT-ECG Feature Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Basic info
        report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Results Directory: {self.results_dir}")
        report_lines.append(f"Number of Patients: {len(self.features_data)}")
        report_lines.append("")
        
        # Patient info
        report_lines.append("Patients Analyzed:")
        for patient, data in self.features_data.items():
            features = data['features']
            report_lines.append(f"  - {patient}: {features.shape} ({data['filename']})")
        report_lines.append("")
        
        # Feature characteristics
        report_lines.append("Feature Characteristics:")
        report_lines.append("  - Feature dimensions: 768 (learned cardiac representations)")
        report_lines.append("  - Temporal steps: 187 (time sequence)")
        report_lines.append("  - Total parameters per patient: 187 × 768 = 143,616")
        report_lines.append("")
        
        # Analysis outputs
        report_lines.append("Generated Analysis Files:")
        analysis_files = [
            "feature_statistics.csv - Basic statistical measures",
            "temporal_patterns.csv - Time-series analysis data", 
            "patient_similarities.csv - Cross-patient comparisons",
            "pca_results.csv - Principal component analysis",
            "temporal_analysis.png - Temporal pattern visualizations",
            "feature_distributions.png - Feature value distributions",
            "patient_comparison.png - Similarity/distance heatmaps",
            "dimensionality_reduction.png - PCA and t-SNE plots"
        ]
        
        for file_desc in analysis_files:
            report_lines.append(f"  - {file_desc}")
        report_lines.append("")
        
        # Interpretation guide
        report_lines.append("Interpretation Guide:")
        report_lines.append("  1. Feature Statistics: Check for data quality and range")
        report_lines.append("  2. Temporal Patterns: Identify cardiac rhythm characteristics")
        report_lines.append("  3. Patient Similarities: Find cardiac phenotype relationships")
        report_lines.append("  4. Dimensionality Reduction: Visualize feature space structure")
        report_lines.append("")
        
        report_lines.append("Next Steps:")
        report_lines.append("  - Use features for downstream classification tasks")
        report_lines.append("  - Correlate with clinical outcomes or diagnoses")
        report_lines.append("  - Expand analysis with additional patients")
        report_lines.append("  - Apply clustering for patient stratification")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to: {report_path}")
        
        # Print report to console
        print("\n" + '\n'.join(report_lines))
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive HuBERT-ECG feature analysis...")
        print("=" * 60)
        
        try:
            # Load data
            self.load_data()
            
            # Run all analyses
            stats_df = self.analyze_basic_statistics()
            temporal_data = self.analyze_temporal_patterns()
            self.analyze_feature_distributions()
            comparison_df = self.compare_patients()
            pca, features_pca, features_tsne = self.dimensionality_reduction_analysis()
            
            # Generate summary
            self.generate_summary_report()
            
            print("\n" + "=" * 60)
            print("Analysis complete! Check the output directory for results:")
            print(f"  {os.path.abspath(self.output_dir)}")
            
            return {
                'stats': stats_df,
                'temporal': temporal_data,
                'comparison': comparison_df,
                'pca': pca,
                'features_pca': features_pca,
                'features_tsne': features_tsne
            }
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Analyze HuBERT-ECG inference results')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing inference results (default: results)')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                        help='Directory to save analysis results (default: analysis_output)')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = HuBERTFeatureAnalyzer(args.results_dir, args.output_dir)
    results = analyzer.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    main()
