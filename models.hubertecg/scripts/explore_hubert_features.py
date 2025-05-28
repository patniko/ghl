#!/usr/bin/env python3
"""
Script to explore HuBERT-ECG feature values extracted from DICOM files.

This script provides comprehensive analysis of the (1, 187, 768) feature tensors
including temporal patterns, feature dimensions, cross-ECG comparisons, and visualizations.

Usage:
    python scripts/explore_hubert_features.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')


class HuBERTFeatureExplorer:
    """Class for comprehensive HuBERT-ECG feature analysis."""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.features = {}
        self.feature_names = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_features(self):
        """Load all HuBERT feature files."""
        print("Loading HuBERT-ECG features...")
        
        feature_files = []
        for file in os.listdir(self.input_dir):
            if file.endswith('_features.npy'):
                feature_files.append(file)
        
        if not feature_files:
            print(f"No feature files found in {self.input_dir}")
            return False
        
        print(f"Found {len(feature_files)} feature files")
        
        for file in sorted(feature_files):
            name = file.replace('_features.npy', '')
            path = os.path.join(self.input_dir, file)
            
            try:
                features = np.load(path)
                self.features[name] = features
                self.feature_names.append(name)
                print(f"  Loaded {name}: {features.shape}")
            except Exception as e:
                print(f"  Error loading {file}: {e}")
        
        return len(self.features) > 0
    
    def basic_statistics(self):
        """Generate basic statistics for all features."""
        print("\n" + "="*60)
        print("BASIC FEATURE STATISTICS")
        print("="*60)
        
        stats_data = []
        
        for name, features in self.features.items():
            # Remove batch dimension: (1, 187, 768) -> (187, 768)
            feat = features.squeeze(0)
            
            stats = {
                'ECG_Name': name,
                'Shape': f"{feat.shape}",
                'Data_Type': str(feat.dtype),
                'Min_Value': feat.min(),
                'Max_Value': feat.max(),
                'Mean_Value': feat.mean(),
                'Std_Value': feat.std(),
                'Zero_Values': (feat == 0).sum(),
                'Total_Values': feat.size
            }
            stats_data.append(stats)
            
            print(f"\n{name}:")
            print(f"  Shape: {feat.shape}")
            print(f"  Data type: {feat.dtype}")
            print(f"  Value range: [{feat.min():.6f}, {feat.max():.6f}]")
            print(f"  Mean: {feat.mean():.6f}")
            print(f"  Std: {feat.std():.6f}")
            print(f"  Zero values: {(feat == 0).sum()} / {feat.size} ({(feat == 0).sum()/feat.size*100:.2f}%)")
        
        # Save statistics to CSV
        stats_df = pd.DataFrame(stats_data)
        stats_path = os.path.join(self.output_dir, 'feature_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"\nStatistics saved to: {stats_path}")
        
        return stats_df
    
    def temporal_analysis(self):
        """Analyze temporal patterns across the 187 time steps."""
        print("\n" + "="*60)
        print("TEMPORAL ANALYSIS (187 time steps)")
        print("="*60)
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Prepare data
        temporal_data = {}
        for name, features in self.features.items():
            feat = features.squeeze(0)  # (187, 768)
            temporal_data[name] = feat
        
        # Plot 1: Feature magnitude over time
        ax1 = fig.add_subplot(gs[0, :])
        colors = plt.cm.tab10(np.linspace(0, 1, len(temporal_data)))
        
        for i, (name, feat) in enumerate(temporal_data.items()):
            # Calculate L2 norm across feature dimensions for each time step
            magnitude = np.linalg.norm(feat, axis=1)
            ax1.plot(magnitude, label=name, linewidth=2, color=colors[i])
        
        ax1.set_title('Feature Magnitude Over Time (L2 norm across 768 dimensions)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Step (0-186)')
        ax1.set_ylabel('Feature Magnitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean feature activation over time
        ax2 = fig.add_subplot(gs[1, 0])
        for i, (name, feat) in enumerate(temporal_data.items()):
            mean_activation = np.mean(feat, axis=1)
            ax2.plot(mean_activation, label=name, linewidth=2, color=colors[i])
        
        ax2.set_title('Mean Feature Activation Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Mean Activation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature variance over time
        ax3 = fig.add_subplot(gs[1, 1])
        for i, (name, feat) in enumerate(temporal_data.items()):
            variance = np.var(feat, axis=1)
            ax3.plot(variance, label=name, linewidth=2, color=colors[i])
        
        ax3.set_title('Feature Variance Over Time', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Variance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Temporal correlation heatmap (first ECG)
        ax4 = fig.add_subplot(gs[2, 0])
        first_name = list(temporal_data.keys())[0]
        first_feat = temporal_data[first_name]
        
        # Sample every 10th time step for visualization
        sample_indices = range(0, 187, 10)
        sampled_feat = first_feat[sample_indices, :]
        
        # Calculate correlation between time steps
        temporal_corr = np.corrcoef(sampled_feat)
        im1 = ax4.imshow(temporal_corr, cmap='coolwarm', aspect='auto')
        ax4.set_title(f'Temporal Correlation Matrix\n({first_name})', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time Step (sampled)')
        ax4.set_ylabel('Time Step (sampled)')
        plt.colorbar(im1, ax=ax4)
        
        # Plot 5: Feature activation heatmap over time
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Show first 50 feature dimensions over time for first ECG
        feat_subset = first_feat[:, :50].T  # (50, 187)
        im2 = ax5.imshow(feat_subset, cmap='viridis', aspect='auto')
        ax5.set_title(f'Feature Activation Heatmap\n({first_name}, first 50 dims)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Feature Dimension')
        plt.colorbar(im2, ax=ax5)
        
        plt.suptitle('Temporal Analysis of HuBERT-ECG Features', fontsize=16, fontweight='bold')
        
        # Save plot
        temporal_path = os.path.join(self.output_dir, 'temporal_analysis.png')
        plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal analysis saved to: {temporal_path}")
        
        return temporal_path
    
    def cross_ecg_comparison(self):
        """Compare features between different ECG recordings."""
        print("\n" + "="*60)
        print("CROSS-ECG COMPARISON")
        print("="*60)
        
        if len(self.features) < 2:
            print("Need at least 2 ECG recordings for comparison")
            return None
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Prepare data
        ecg_names = list(self.features.keys())
        n_ecgs = len(ecg_names)
        
        # Flatten features for comparison: (187*768,) for each ECG
        flattened_features = {}
        for name, features in self.features.items():
            feat = features.squeeze(0)  # (187, 768)
            flattened_features[name] = feat.flatten()
        
        # Calculate similarity matrices
        feature_vectors = np.array(list(flattened_features.values()))  # (n_ecgs, 187*768)
        
        # Cosine similarity
        cosine_sim = cosine_similarity(feature_vectors)
        
        # Euclidean distance
        euclidean_dist = euclidean_distances(feature_vectors)
        
        # Plot 1: Cosine similarity heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(cosine_sim, cmap='viridis', vmin=0, vmax=1)
        ax1.set_title('Cosine Similarity Between ECGs', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(n_ecgs))
        ax1.set_yticks(range(n_ecgs))
        ax1.set_xticklabels(ecg_names, rotation=45)
        ax1.set_yticklabels(ecg_names)
        
        # Add similarity values as text
        for i in range(n_ecgs):
            for j in range(n_ecgs):
                ax1.text(j, i, f'{cosine_sim[i, j]:.3f}', 
                        ha='center', va='center', color='white' if cosine_sim[i, j] < 0.5 else 'black')
        
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Euclidean distance heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(euclidean_dist, cmap='plasma')
        ax2.set_title('Euclidean Distance Between ECGs', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(n_ecgs))
        ax2.set_yticks(range(n_ecgs))
        ax2.set_xticklabels(ecg_names, rotation=45)
        ax2.set_yticklabels(ecg_names)
        
        # Add distance values as text
        for i in range(n_ecgs):
            for j in range(n_ecgs):
                ax2.text(j, i, f'{euclidean_dist[i, j]:.1f}', 
                        ha='center', va='center', color='white' if euclidean_dist[i, j] > euclidean_dist.max()/2 else 'black')
        
        plt.colorbar(im2, ax=ax2)
        
        # Plot 3: Hierarchical clustering dendrogram
        ax3 = fig.add_subplot(gs[1, :])
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(feature_vectors, method='ward')
        dendrogram(linkage_matrix, labels=ecg_names, ax=ax3)
        ax3.set_title('Hierarchical Clustering of ECGs (Ward linkage)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Distance')
        
        plt.suptitle('Cross-ECG Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Save plot
        comparison_path = os.path.join(self.output_dir, 'cross_ecg_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cross-ECG comparison saved to: {comparison_path}")
        
        # Print similarity analysis
        print("\nSimilarity Analysis:")
        for i in range(n_ecgs):
            for j in range(i+1, n_ecgs):
                print(f"  {ecg_names[i]} vs {ecg_names[j]}:")
                print(f"    Cosine similarity: {cosine_sim[i, j]:.4f}")
                print(f"    Euclidean distance: {euclidean_dist[i, j]:.2f}")
        
        return comparison_path
    
    def run_complete_analysis(self):
        """Run the complete feature exploration analysis."""
        print("Starting comprehensive HuBERT-ECG feature exploration...")
        
        # Load features
        if not self.load_features():
            print("Failed to load features. Exiting.")
            return
        
        # Run all analyses
        print("\n1. Basic Statistics Analysis")
        self.basic_statistics()
        
        print("\n2. Temporal Analysis")
        self.temporal_analysis()
        
        print("\n3. Cross-ECG Comparison")
        self.cross_ecg_comparison()
        
        # Generate summary report
        print("\n4. Generating Summary Report")
        report_path = os.path.join(self.output_dir, 'feature_exploration_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("HuBERT-ECG Feature Exploration Summary\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Directory: {self.input_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write(f"ECG Recordings Analyzed: {len(self.features)}\n")
            for name in self.feature_names:
                f.write(f"  - {name}\n")
            f.write("\n")
            
            f.write("Feature Shape Information:\n")
            for name, features in self.features.items():
                f.write(f"  {name}: {features.shape}\n")
            f.write("\n")
            
            f.write("Generated Analysis Files:\n")
            f.write("  - feature_statistics.csv: Basic feature statistics\n")
            f.write("  - temporal_analysis.png: Temporal pattern visualizations\n")
            f.write("  - cross_ecg_comparison.png: ECG comparison visualizations\n\n")
            
            f.write("Key Findings:\n")
            f.write("  - Each ECG produces features with shape (1, 187, 768)\n")
            f.write("  - 187 time steps represent temporal evolution of ECG patterns\n")
            f.write("  - 768 feature dimensions capture rich cardiac representations\n")
            f.write("  - Features show distinct temporal and dimensional patterns\n")
            f.write("  - Cross-ECG comparisons reveal similarity/difference patterns\n\n")
            
            f.write("Analysis Complete!\n")
        
        print(f"\nSummary report saved to: {report_path}")
        print(f"\nFeature exploration complete! Check {self.output_dir} for all results.")


def main():
    parser = argparse.ArgumentParser(description='Explore HuBERT-ECG feature values.')
    parser.add_argument('--input_dir', type=str, default='results/12L/inference_results',
                        help='Directory containing feature files (default: results/12L/inference_results)')
    parser.add_argument('--output_dir', type=str, default='results/12L/feature_exploration',
                        help='Directory to save analysis results (default: results/12L/feature_exploration)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        print("Please run 'make dicom-inference' first to generate HuBERT features")
        sys.exit(1)
    
    # Create explorer and run analysis
    explorer = HuBERTFeatureExplorer(args.input_dir, args.output_dir)
    explorer.run_complete_analysis()


if __name__ == "__main__":
    main()
