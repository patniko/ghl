#!/usr/bin/env python3
"""
Comprehensive visualization script for HuBERT-ECG inference results.

This script creates detailed visualizations of the extracted features,
including heatmaps, temporal patterns, and comparative analyses.

Usage:
    python scripts/visualize_results.py [--results_dir RESULTS_DIR] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class HuBERTFeatureVisualizer:
    """Comprehensive visualizer for HuBERT-ECG features."""
    
    def __init__(self, results_dir='results', output_dir='visualization_output'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.features_data = {}
        self.metadata = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load all feature files and metadata."""
        print("Loading HuBERT-ECG inference results for visualization...")
        
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
    
    def create_feature_heatmaps(self):
        """Create detailed heatmaps of the features."""
        print("Creating feature heatmaps...")
        
        n_patients = len(self.features_data)
        fig, axes = plt.subplots(n_patients, 1, figsize=(20, 6*n_patients))
        if n_patients == 1:
            axes = [axes]
        
        fig.suptitle('HuBERT-ECG Feature Heatmaps (Variable Time Steps × 768 Features)', fontsize=16)
        
        for i, (patient, data) in enumerate(self.features_data.items()):
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)  # Shape: (187, 768)
            
            # Create heatmap
            im = axes[i].imshow(features.T, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i].set_title(f'{patient} - Feature Heatmap\n{data["filename"]} | Shape: {features.shape}')
            axes[i].set_xlabel(f'Time Steps ({features.shape[0]})')
            axes[i].set_ylabel('Feature Dimensions (768)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label('Feature Value')
            
            # Add grid lines for better readability
            time_step_interval = max(1, features.shape[0] // 10)
            axes[i].set_xticks(np.arange(0, features.shape[0], time_step_interval))
            axes[i].set_yticks(np.arange(0, 768, 100))
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        heatmap_path = os.path.join(self.output_dir, 'feature_heatmaps.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature heatmaps saved to: {heatmap_path}")
    
    def create_temporal_evolution_plots(self):
        """Create plots showing how features evolve over time."""
        print("Creating temporal evolution plots...")
        
        # Calculate temporal statistics and determine max time steps
        temporal_stats = {}
        max_time_steps = 0
        
        for patient, data in self.features_data.items():
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            
            time_steps = features.shape[0]
            max_time_steps = max(max_time_steps, time_steps)
            
            temporal_stats[patient] = {
                'mean': np.mean(features, axis=1),
                'std': np.std(features, axis=1),
                'max': np.max(features, axis=1),
                'min': np.min(features, axis=1),
                'median': np.median(features, axis=1),
                'q75': np.percentile(features, 75, axis=1),
                'q25': np.percentile(features, 25, axis=1),
                'time_steps': time_steps
            }
        
        # Create comprehensive temporal plots
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Temporal Evolution of HuBERT-ECG Features', fontsize=16)
        
        # Plot 1: Mean evolution
        for patient, stats in temporal_stats.items():
            patient_time_steps = np.arange(stats['time_steps'])
            axes[0, 0].plot(patient_time_steps, stats['mean'], label=f"{patient} ({stats['time_steps']} steps)", linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Mean Feature Values Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Mean Feature Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation evolution
        for patient, stats in temporal_stats.items():
            patient_time_steps = np.arange(stats['time_steps'])
            axes[0, 1].plot(patient_time_steps, stats['std'], label=f"{patient} ({stats['time_steps']} steps)", linewidth=2, alpha=0.8)
        axes[0, 1].set_title('Feature Variability Over Time')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Range evolution (max-min)
        for patient, stats in temporal_stats.items():
            patient_time_steps = np.arange(stats['time_steps'])
            range_vals = stats['max'] - stats['min']
            axes[1, 0].plot(patient_time_steps, range_vals, label=f"{patient} ({stats['time_steps']} steps)", linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Feature Range Over Time')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Range (Max - Min)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Median evolution with quartiles
        for patient, stats in temporal_stats.items():
            patient_time_steps = np.arange(stats['time_steps'])
            axes[1, 1].plot(patient_time_steps, stats['median'], label=f"{patient} (median, {stats['time_steps']} steps)", linewidth=2)
            axes[1, 1].fill_between(patient_time_steps, stats['q25'], stats['q75'], alpha=0.2)
        axes[1, 1].set_title('Median Features with Quartiles')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Feature Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Feature activation patterns (top 10% of features)
        for patient, data in self.features_data.items():
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            
            patient_time_steps = np.arange(features.shape[0])
            
            # Find highly active features (top 10% by variance)
            feature_vars = np.var(features, axis=0)
            top_features_idx = np.argsort(feature_vars)[-int(0.1 * len(feature_vars)):]
            
            # Plot mean of top features over time
            top_features_mean = np.mean(features[:, top_features_idx], axis=1)
            axes[2, 0].plot(patient_time_steps, top_features_mean, label=f"{patient} ({features.shape[0]} steps)", linewidth=2, alpha=0.8)
        
        axes[2, 0].set_title('High-Variance Features Over Time')
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_ylabel('Mean of Top 10% Variable Features')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Cumulative feature energy
        for patient, data in self.features_data.items():
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            
            patient_time_steps = np.arange(features.shape[0])
            
            # Calculate cumulative energy (sum of squared features)
            energy_per_timestep = np.sum(features**2, axis=1)
            cumulative_energy = np.cumsum(energy_per_timestep)
            axes[2, 1].plot(patient_time_steps, cumulative_energy, label=f"{patient} ({features.shape[0]} steps)", linewidth=2, alpha=0.8)
        
        axes[2, 1].set_title('Cumulative Feature Energy')
        axes[2, 1].set_xlabel('Time Step')
        axes[2, 1].set_ylabel('Cumulative Energy')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        temporal_path = os.path.join(self.output_dir, 'temporal_evolution.png')
        plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Temporal evolution plots saved to: {temporal_path}")
    
    def create_feature_importance_plots(self):
        """Create plots showing which features are most important."""
        print("Creating feature importance plots...")
        
        # Calculate feature importance metrics
        feature_importance = {}
        
        for patient, data in self.features_data.items():
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            
            # Calculate various importance metrics
            feature_importance[patient] = {
                'variance': np.var(features, axis=0),
                'mean_abs': np.mean(np.abs(features), axis=0),
                'max_abs': np.max(np.abs(features), axis=0),
                'energy': np.sum(features**2, axis=0)
            }
        
        # Create feature importance plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Feature Importance Analysis (768 Features)', fontsize=16)
        
        feature_indices = np.arange(768)
        
        metrics = ['variance', 'mean_abs', 'max_abs', 'energy']
        titles = ['Feature Variance', 'Mean Absolute Value', 'Maximum Absolute Value', 'Feature Energy']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            for patient, importance in feature_importance.items():
                ax.plot(feature_indices, importance[metric], label=patient, alpha=0.7, linewidth=1)
            
            ax.set_title(title)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel(title.split()[-1])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Highlight top features
            if len(self.features_data) == 1:  # If only one patient, highlight top features
                patient = list(self.features_data.keys())[0]
                top_indices = np.argsort(feature_importance[patient][metric])[-10:]
                ax.scatter(top_indices, feature_importance[patient][metric][top_indices], 
                          color='red', s=30, alpha=0.8, zorder=5)
        
        plt.tight_layout()
        importance_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plots saved to: {importance_path}")
        
        # Create top features summary
        self.create_top_features_summary(feature_importance)
    
    def create_top_features_summary(self, feature_importance):
        """Create a summary of the most important features."""
        print("Creating top features summary...")
        
        # Find top features for each patient and metric
        top_features_data = []
        
        for patient, importance in feature_importance.items():
            for metric, values in importance.items():
                top_indices = np.argsort(values)[-20:]  # Top 20 features
                
                for rank, idx in enumerate(reversed(top_indices)):
                    top_features_data.append({
                        'patient': patient,
                        'metric': metric,
                        'rank': rank + 1,
                        'feature_index': idx,
                        'value': values[idx]
                    })
        
        top_features_df = pd.DataFrame(top_features_data)
        
        # Save to CSV
        top_features_path = os.path.join(self.output_dir, 'top_features_summary.csv')
        top_features_df.to_csv(top_features_path, index=False)
        print(f"Top features summary saved to: {top_features_path}")
        
        # Create visualization of top features
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Top 20 Most Important Features by Metric', fontsize=16)
        
        metrics = ['variance', 'mean_abs', 'max_abs', 'energy']
        titles = ['Highest Variance', 'Highest Mean Absolute', 'Highest Max Absolute', 'Highest Energy']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            for patient in self.features_data.keys():
                patient_data = top_features_df[(top_features_df['patient'] == patient) & 
                                             (top_features_df['metric'] == metric) & 
                                             (top_features_df['rank'] <= 20)]
                
                ax.barh(patient_data['rank'], patient_data['value'], 
                       label=patient, alpha=0.7)
            
            ax.set_title(title)
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Rank (1 = highest)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()  # Highest rank at top
        
        plt.tight_layout()
        top_features_viz_path = os.path.join(self.output_dir, 'top_features_visualization.png')
        plt.savefig(top_features_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top features visualization saved to: {top_features_viz_path}")
    
    def create_patient_comparison_detailed(self):
        """Create detailed patient comparison visualizations."""
        print("Creating detailed patient comparisons...")
        
        if len(self.features_data) < 2:
            print("Need at least 2 patients for comparison. Skipping detailed comparison.")
            return
        
        # Prepare data for comparison
        patients = list(self.features_data.keys())
        n_patients = len(patients)
        
        # Create pairwise comparison plots
        n_pairs = n_patients * (n_patients - 1) // 2
        fig, axes = plt.subplots(n_pairs, 3, figsize=(18, 6*n_pairs))
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Detailed Patient Comparisons', fontsize=16)
        
        pair_idx = 0
        for i in range(n_patients):
            for j in range(i+1, n_patients):
                patient1, patient2 = patients[i], patients[j]
                
                features1 = self.features_data[patient1]['features']
                features2 = self.features_data[patient2]['features']
                
                if features1.ndim == 3 and features1.shape[0] == 1:
                    features1 = features1.squeeze(0)
                if features2.ndim == 3 and features2.shape[0] == 1:
                    features2 = features2.squeeze(0)
                
                # Plot 1: Feature difference heatmap (only if same shape)
                if features1.shape == features2.shape:
                    diff = features1 - features2
                    im1 = axes[pair_idx, 0].imshow(diff.T, aspect='auto', cmap='RdBu_r', 
                                                  vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
                    axes[pair_idx, 0].set_title(f'{patient1} - {patient2}\nFeature Differences')
                    plt.colorbar(im1, ax=axes[pair_idx, 0], fraction=0.046, pad=0.04)
                else:
                    # Show both heatmaps side by side when shapes differ
                    min_time = min(features1.shape[0], features2.shape[0])
                    combined = np.concatenate([features1[:min_time, :], features2[:min_time, :]], axis=0)
                    im1 = axes[pair_idx, 0].imshow(combined.T, aspect='auto', cmap='viridis')
                    axes[pair_idx, 0].axvline(x=min_time-0.5, color='red', linestyle='--', linewidth=2)
                    axes[pair_idx, 0].set_title(f'{patient1} | {patient2}\nConcatenated Features (Different Shapes)')
                    plt.colorbar(im1, ax=axes[pair_idx, 0], fraction=0.046, pad=0.04)
                
                axes[pair_idx, 0].set_xlabel('Time Steps')
                axes[pair_idx, 0].set_ylabel('Features')
                
                # Plot 2: Temporal mean comparison
                mean1 = np.mean(features1, axis=1)
                mean2 = np.mean(features2, axis=1)
                
                # Handle different time dimensions
                if features1.shape[0] == features2.shape[0]:
                    time_steps = np.arange(features1.shape[0])
                    axes[pair_idx, 1].plot(time_steps, mean1, label=patient1, linewidth=2)
                    axes[pair_idx, 1].plot(time_steps, mean2, label=patient2, linewidth=2)
                    axes[pair_idx, 1].fill_between(time_steps, mean1, mean2, alpha=0.3)
                else:
                    time_steps1 = np.arange(features1.shape[0])
                    time_steps2 = np.arange(features2.shape[0])
                    axes[pair_idx, 1].plot(time_steps1, mean1, label=f'{patient1} ({features1.shape[0]} steps)', linewidth=2)
                    axes[pair_idx, 1].plot(time_steps2, mean2, label=f'{patient2} ({features2.shape[0]} steps)', linewidth=2)
                axes[pair_idx, 1].set_title('Temporal Mean Comparison')
                axes[pair_idx, 1].set_xlabel('Time Steps')
                axes[pair_idx, 1].set_ylabel('Mean Feature Value')
                axes[pair_idx, 1].legend()
                axes[pair_idx, 1].grid(True, alpha=0.3)
                
                # Plot 3: Feature correlation
                # Handle different shapes by using minimum dimensions
                min_time_steps = min(features1.shape[0], features2.shape[0])
                min_features = min(features1.shape[1], features2.shape[1])
                
                # Truncate to common dimensions
                features1_trunc = features1[:min_time_steps, :min_features]
                features2_trunc = features2[:min_time_steps, :min_features]
                
                # Flatten features for correlation
                flat1 = features1_trunc.flatten()
                flat2 = features2_trunc.flatten()
                
                # Sample points for scatter plot (too many points otherwise)
                n_sample = min(5000, len(flat1))
                sample_idx = np.random.choice(len(flat1), n_sample, replace=False)
                
                axes[pair_idx, 2].scatter(flat1[sample_idx], flat2[sample_idx], 
                                        alpha=0.5, s=1)
                
                # Add correlation line
                correlation = np.corrcoef(flat1, flat2)[0, 1]
                min_val = min(np.min(flat1), np.min(flat2))
                max_val = max(np.max(flat1), np.max(flat2))
                axes[pair_idx, 2].plot([min_val, max_val], [min_val, max_val], 
                                     'r--', alpha=0.8, linewidth=2)
                
                axes[pair_idx, 2].set_title(f'Feature Correlation\nr = {correlation:.4f}')
                axes[pair_idx, 2].set_xlabel(f'{patient1} Features')
                axes[pair_idx, 2].set_ylabel(f'{patient2} Features')
                axes[pair_idx, 2].grid(True, alpha=0.3)
                
                pair_idx += 1
        
        plt.tight_layout()
        comparison_path = os.path.join(self.output_dir, 'detailed_patient_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Detailed patient comparison saved to: {comparison_path}")
    
    def create_interactive_plots(self):
        """Create interactive Plotly visualizations."""
        print("Creating interactive visualizations...")
        
        try:
            # Interactive heatmap for each patient
            for patient, data in self.features_data.items():
                features = data['features']
                if features.ndim == 3 and features.shape[0] == 1:
                    features = features.squeeze(0)
                
                fig = go.Figure(data=go.Heatmap(
                    z=features.T,
                    x=list(range(features.shape[0])),
                    y=list(range(768)),
                    colorscale='Viridis',
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title=f'{patient} - Interactive Feature Heatmap<br>{data["filename"]} | Shape: {features.shape}',
                    xaxis_title=f'Time Steps ({features.shape[0]})',
                    yaxis_title='Feature Dimensions (768)',
                    width=1000,
                    height=600
                )
                
                interactive_path = os.path.join(self.output_dir, f'{patient}_interactive_heatmap.html')
                fig.write_html(interactive_path)
                print(f"Interactive heatmap for {patient} saved to: {interactive_path}")
            
            # Interactive temporal comparison
            if len(self.features_data) > 1:
                fig = go.Figure()
                
                for patient, data in self.features_data.items():
                    features = data['features']
                    if features.ndim == 3 and features.shape[0] == 1:
                        features = features.squeeze(0)
                    
                    mean_features = np.mean(features, axis=1)
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(features.shape[0])),
                        y=mean_features,
                        mode='lines',
                        name=f'{patient} ({features.shape[0]} steps)',
                        line=dict(width=3)
                    ))
                
                fig.update_layout(
                    title='Interactive Temporal Comparison of Mean Features',
                    xaxis_title='Time Steps',
                    yaxis_title='Mean Feature Value',
                    width=1000,
                    height=500,
                    hovermode='x unified'
                )
                
                temporal_interactive_path = os.path.join(self.output_dir, 'interactive_temporal_comparison.html')
                fig.write_html(temporal_interactive_path)
                print(f"Interactive temporal comparison saved to: {temporal_interactive_path}")
                
        except ImportError:
            print("Plotly not available. Skipping interactive visualizations.")
            print("Install plotly with: pip install plotly")
    
    def create_summary_dashboard(self):
        """Create a summary dashboard with key visualizations."""
        print("Creating summary dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('HuBERT-ECG Feature Analysis Dashboard', fontsize=20, y=0.98)
        
        # 1. Feature statistics summary (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        stats_data = []
        for patient, data in self.features_data.items():
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            
            stats_data.append([
                patient,
                f"{features.shape}",
                f"{np.mean(features):.4f}",
                f"{np.std(features):.4f}",
                f"[{np.min(features):.3f}, {np.max(features):.3f}]"
            ])
        
        table = ax1.table(cellText=stats_data,
                         colLabels=['Patient', 'Shape', 'Mean', 'Std', 'Range'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax1.axis('off')
        ax1.set_title('Feature Statistics Summary', fontsize=14, pad=20)
        
        # 2. Mini heatmaps (top-right)
        for i, (patient, data) in enumerate(self.features_data.items()):
            ax = fig.add_subplot(gs[0, 2+i] if i < 2 else gs[1, i-2])
            
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            
            # Downsample for mini heatmap
            downsampled = features[::4, ::16]  # Every 4th time step, every 16th feature
            
            im = ax.imshow(downsampled.T, aspect='auto', cmap='viridis')
            ax.set_title(f'{patient}\n(Downsampled)', fontsize=10)
            ax.set_xlabel('Time')
            ax.set_ylabel('Features')
        
        # 3. Temporal evolution (middle)
        ax3 = fig.add_subplot(gs[1, :])
        for patient, data in self.features_data.items():
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            
            mean_temporal = np.mean(features, axis=1)
            ax3.plot(mean_temporal, label=patient, linewidth=2, alpha=0.8)
        
        ax3.set_title('Temporal Evolution of Mean Features', fontsize=14)
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Mean Feature Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance (bottom-left)
        ax4 = fig.add_subplot(gs[2, :2])
        for patient, data in self.features_data.items():
            features = data['features']
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            
            feature_vars = np.var(features, axis=0)
            # Show only every 10th feature for readability
            ax4.plot(feature_vars[::10], label=patient, alpha=0.7)
        
        ax4.set_title('Feature Variance (Every 10th Feature)', fontsize=14)
        ax4.set_xlabel('Feature Index (Subsampled)')
        ax4.set_ylabel('Variance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Patient similarity (bottom-right)
        if len(self.features_data) > 1:
            ax5 = fig.add_subplot(gs[2, 2:])
            
            # Calculate similarity matrix
            patients = list(self.features_data.keys())
            n_patients = len(patients)
            similarity_matrix = np.eye(n_patients)
            
            for i, patient1 in enumerate(patients):
                for j, patient2 in enumerate(patients):
                    if i != j:
                        features1 = self.features_data[patient1]['features']
                        features2 = self.features_data[patient2]['features']
                        
                        if features1.ndim == 3 and features1.shape[0] == 1:
                            features1 = features1.squeeze(0)
                        if features2.ndim == 3 and features2.shape[0] == 1:
                            features2 = features2.squeeze(0)
                        
                        # Calculate cosine similarity using common dimensions
                        min_time_steps = min(features1.shape[0], features2.shape[0])
                        min_features = min(features1.shape[1], features2.shape[1])
                        
                        # Truncate to common dimensions
                        features1_trunc = features1[:min_time_steps, :min_features]
                        features2_trunc = features2[:min_time_steps, :min_features]
                        
                        flat1 = features1_trunc.flatten()
                        flat2 = features2_trunc.flatten()
                        similarity = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))
                        similarity_matrix[i, j] = similarity
            
            im = ax5.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
            ax5.set_title('Patient Similarity Matrix', fontsize=14)
            ax5.set_xticks(range(n_patients))
            ax5.set_yticks(range(n_patients))
            ax5.set_xticklabels(patients)
            ax5.set_yticklabels(patients)
            
            # Add text annotations
            for i in range(n_patients):
                for j in range(n_patients):
                    ax5.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                           ha='center', va='center', color='white' if similarity_matrix[i, j] < 0.5 else 'black')
            
            plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        
        # 6. Analysis summary (bottom)
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        summary_text = f"""
        Analysis Summary:
        • Processed {len(self.features_data)} patients with HuBERT-ECG features
        • Each patient has 187 time steps × 768 feature dimensions = 143,616 total features
        • Features represent learned cardiac representations from the HuBERT-ECG foundation model
        • Temporal patterns show how cardiac features evolve over the ECG sequence
        • Feature importance analysis identifies the most discriminative cardiac patterns
        • Patient similarities reveal relationships between cardiac phenotypes
        """
        
        ax6.text(0.05, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, 'analysis_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary dashboard saved to: {dashboard_path}")
    
    def run_complete_visualization(self):
        """Run the complete visualization pipeline."""
        print("Starting comprehensive HuBERT-ECG feature visualization...")
        print("="*60)
        
        try:
            # Load data
            self.load_data()
            
            # Create all visualizations
            self.create_feature_heatmaps()
            self.create_temporal_evolution_plots()
            self.create_feature_importance_plots()
            self.create_patient_comparison_detailed()
            self.create_interactive_plots()
            self.create_summary_dashboard()
            
            print("\n" + "="*60)
            print("Visualization complete! Check the output directory for results:")
            print(f"  {os.path.abspath(self.output_dir)}")
            
            # List generated files
            generated_files = []
            for file in os.listdir(self.output_dir):
                if file.endswith(('.png', '.html', '.csv')):
                    generated_files.append(file)
            
            if generated_files:
                print("\nGenerated files:")
                for file in sorted(generated_files):
                    print(f"  - {file}")
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Visualize HuBERT-ECG inference results')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing inference results (default: results)')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                        help='Directory to save visualization results (default: visualization_output)')
    
    args = parser.parse_args()
    
    # Create visualizer and run visualization
    visualizer = HuBERTFeatureVisualizer(args.results_dir, args.output_dir)
    visualizer.run_complete_visualization()


if __name__ == "__main__":
    main()
