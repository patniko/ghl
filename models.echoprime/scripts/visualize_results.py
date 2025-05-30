#!/usr/bin/env python3
"""
Comprehensive visualization script for EchoPrime inference results.

This script creates detailed visualizations of the cardiac phenotype predictions,
including metric distributions, success rates, and comparative analyses across folders.

Usage:
    python scripts/visualize_results.py [--results_dir RESULTS_DIR] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import json
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


class EchoPrimeVisualizer:
    """Comprehensive visualizer for EchoPrime inference results with device analytics."""
    
    def __init__(self, results_dir='results/inference_output', output_dir='results/visualization_output'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.summary_data = None
        self.folder_data = []
        self.all_metrics = []
        self.device_data = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define cardiac metrics for analysis
        self.cardiac_metrics = [
            'ejection_fraction', 'impella', 'pacemaker', 'rv_systolic_function_depressed',
            'right_ventricle_dilation', 'left_atrium_dilation', 'right_atrium_dilation',
            'mitraclip', 'mitral_annular_calcification', 'mitral_stenosis', 'mitral_regurgitation',
            'tavr', 'bicuspid_aov_morphology', 'aortic_stenosis', 'aortic_regurgitation',
            'tricuspid_stenosis', 'tricuspid_valve_regurgitation', 'pericardial_effusion',
            'aortic_root_dilation', 'dilated_ivc', 'pulmonary_artery_pressure_continuous'
        ]
        
    def extract_device_info(self, folder_name):
        """Extract device information from folder name."""
        # Try to extract device info from folder name (format: device-model-id)
        parts = folder_name.split('-')
        if len(parts) >= 2:
            return parts[0]  # Return device name
        return folder_name  # Return full folder name if no clear device pattern
        
    def load_data(self):
        """Load summary and individual folder results."""
        print("Loading EchoPrime inference results for visualization...")
        
        # Load summary data
        summary_path = os.path.join(self.results_dir, 'summary.json')
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        
        with open(summary_path, 'r') as f:
            self.summary_data = json.load(f)
        
        print(f"Found {self.summary_data['total_folders']} folders with {self.summary_data['total_videos']} total videos")
        print(f"Successful folders: {self.summary_data['successful']}")
        print(f"Failed folders: {self.summary_data['failed']}")
        
        # Process folder results with device information
        for folder_result in self.summary_data['results']:
            if folder_result['status'] == 'success' and 'metrics' in folder_result:
                # Extract device information from folder name
                device_name = self.extract_device_info(folder_result['folder'])
                folder_result['device_name'] = device_name
                
                self.folder_data.append(folder_result)
                
                # Add folder info to metrics for analysis
                metrics_with_folder = folder_result['metrics'].copy()
                metrics_with_folder['folder'] = folder_result['folder']
                metrics_with_folder['device_name'] = device_name
                metrics_with_folder['num_videos'] = folder_result['num_videos']
                metrics_with_folder['num_processed'] = folder_result['num_processed']
                metrics_with_folder['num_files'] = folder_result['num_files']
                
                # Calculate failure metrics
                if 'error_stats' in folder_result:
                    total_errors = sum(folder_result['error_stats']['error_counts'].values())
                    metrics_with_folder['total_errors'] = total_errors
                    metrics_with_folder['error_rate'] = total_errors / folder_result['num_files'] if folder_result['num_files'] > 0 else 0
                    metrics_with_folder['processing_success_rate'] = folder_result['num_processed'] / folder_result['num_files'] if folder_result['num_files'] > 0 else 0
                else:
                    metrics_with_folder['total_errors'] = 0
                    metrics_with_folder['error_rate'] = 0
                    metrics_with_folder['processing_success_rate'] = 1.0
                
                self.all_metrics.append(metrics_with_folder)
                
                # Group by device
                if device_name not in self.device_data:
                    self.device_data[device_name] = []
                self.device_data[device_name].append(metrics_with_folder)
        
        print(f"Loaded metrics for {len(self.folder_data)} successful folders")
        print(f"Total metric records: {len(self.all_metrics)}")
    
    def create_overall_summary_dashboard(self):
        """Create a comprehensive summary dashboard."""
        print("Creating overall summary dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('EchoPrime Cardiac Phenotype Analysis Dashboard', fontsize=20, y=0.98)
        
        # 1. Overall statistics (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        stats_data = [
            ['Total Folders', f"{self.summary_data['total_folders']}"],
            ['Successful Folders', f"{self.summary_data['successful']}"],
            ['Failed Folders', f"{self.summary_data['failed']}"],
            ['Total Videos', f"{self.summary_data['total_videos']}"],
            ['Success Rate', f"{(self.summary_data['successful']/self.summary_data['total_folders']*100):.1f}%"],
            ['Avg Videos/Folder', f"{np.mean([f['num_videos'] for f in self.folder_data]):.1f}"]
        ]
        
        table = ax1.table(cellText=stats_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        ax1.axis('off')
        ax1.set_title('Overall Statistics', fontsize=14, pad=20)
        
        # 2. Ejection fraction distribution (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        ef_values = [m['ejection_fraction'] for m in self.all_metrics if m['ejection_fraction'] > 0]
        if ef_values:
            ax2.hist(ef_values, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Normal Threshold (50%)')
            ax2.axvline(x=np.mean(ef_values), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ef_values):.1f}%')
            ax2.set_title('Ejection Fraction Distribution', fontsize=14)
            ax2.set_xlabel('Ejection Fraction (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No EF data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Ejection Fraction Distribution', fontsize=14)
        
        # 3. Processing success by folder (middle-left)
        ax3 = fig.add_subplot(gs[1, :2])
        folder_names = [f['folder'][:30] + '...' if len(f['folder']) > 30 else f['folder'] for f in self.folder_data[:20]]
        processing_rates = [(f['num_processed']/f['num_files']*100) if f['num_files'] > 0 else 0 for f in self.folder_data[:20]]
        
        bars = ax3.barh(range(len(folder_names)), processing_rates, color='lightgreen', alpha=0.7)
        ax3.set_yticks(range(len(folder_names)))
        ax3.set_yticklabels(folder_names, fontsize=8)
        ax3.set_xlabel('Processing Success Rate (%)')
        ax3.set_title('Processing Success Rate by Folder (Top 20)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add rate labels on bars
        for i, (bar, rate) in enumerate(zip(bars, processing_rates)):
            ax3.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=8)
        
        # 4. Cardiac condition prevalence (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # Calculate prevalence of binary conditions (excluding continuous metrics)
        binary_metrics = [m for m in self.cardiac_metrics if m not in ['ejection_fraction', 'pulmonary_artery_pressure_continuous']]
        prevalence_data = {}
        
        for metric in binary_metrics:
            values = [m[metric] for m in self.all_metrics if metric in m and m[metric] > 0]
            prevalence_data[metric] = len(values)
        
        # Show top 10 most prevalent conditions
        sorted_conditions = sorted(prevalence_data.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if sorted_conditions:
            conditions = [c[0].replace('_', ' ').title() for c, _ in sorted_conditions]
            counts = [count for _, count in sorted_conditions]
            
            bars = ax4.barh(range(len(conditions)), counts, color='skyblue', alpha=0.7)
            ax4.set_yticks(range(len(conditions)))
            ax4.set_yticklabels([c[:25] + '...' if len(c) > 25 else c for c in conditions], fontsize=8)
            ax4.set_xlabel('Number of Cases')
            ax4.set_title('Top 10 Cardiac Conditions (Prevalence)', fontsize=14)
            ax4.grid(True, alpha=0.3)
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax4.text(count + max(counts)*0.01, i, str(count), va='center', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No condition data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cardiac Conditions Prevalence', fontsize=14)
        
        # 5. Error analysis (bottom-left)
        ax5 = fig.add_subplot(gs[2, :2])
        
        # Aggregate error statistics
        total_errors = {}
        for folder in self.folder_data:
            if 'error_stats' in folder:
                for error_type, count in folder['error_stats']['error_counts'].items():
                    total_errors[error_type] = total_errors.get(error_type, 0) + count
        
        if total_errors and sum(total_errors.values()) > 0:
            error_types = list(total_errors.keys())
            error_counts = list(total_errors.values())
            
            bars = ax5.bar(error_types, error_counts, color='orange', alpha=0.7)
            ax5.set_title('Error Type Distribution', fontsize=14)
            ax5.set_xlabel('Error Type')
            ax5.set_ylabel('Count')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add count labels
            for bar, count in zip(bars, error_counts):
                if count > 0:
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(error_counts)*0.01,
                           str(count), ha='center', va='bottom', fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'No errors found', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Error Type Distribution', fontsize=14)
        
        # 6. Video processing statistics (bottom-right)
        ax6 = fig.add_subplot(gs[2, 2:])
        
        total_files = sum(f['num_files'] for f in self.folder_data)
        total_processed = sum(f['num_processed'] for f in self.folder_data)
        total_videos = sum(f['num_videos'] for f in self.folder_data)
        
        categories = ['Total Files', 'Processed Files', 'Video Files']
        counts = [total_files, total_processed, total_videos]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        bars = ax6.bar(categories, counts, color=colors, alpha=0.7)
        ax6.set_title('File Processing Summary', fontsize=14)
        ax6.set_ylabel('Count')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                   f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        # 7. Folder performance comparison (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        
        # Create scatter plot of folder performance
        x_data = [f['num_videos'] for f in self.folder_data]
        y_data = [(f['num_processed']/f['num_files']*100) if f['num_files'] > 0 else 0 for f in self.folder_data]
        
        scatter = ax7.scatter(x_data, y_data, alpha=0.6, s=50, c='purple')
        ax7.set_xlabel('Number of Videos')
        ax7.set_ylabel('Processing Success Rate (%)')
        ax7.set_title('Folder Performance: Videos vs Processing Success Rate', fontsize=14)
        ax7.grid(True, alpha=0.3)
        
        # Add correlation info
        if len(x_data) > 1:
            correlation = np.corrcoef(x_data, y_data)[0, 1]
            ax7.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, 'echoprime_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary dashboard saved to: {dashboard_path}")
    
    def create_cardiac_metrics_analysis(self):
        """Create detailed cardiac metrics analysis."""
        print("Creating cardiac metrics analysis...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('EchoPrime Cardiac Metrics Analysis', fontsize=16)
        
        # 1. Ejection fraction detailed analysis
        ef_values = [m['ejection_fraction'] for m in self.all_metrics if m['ejection_fraction'] > 0]
        if ef_values:
            axes[0, 0].hist(ef_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 0].axvline(x=50, color='red', linestyle='--', label='Normal Threshold')
            axes[0, 0].axvline(x=np.mean(ef_values), color='blue', linestyle='--', label=f'Mean: {np.mean(ef_values):.1f}%')
            axes[0, 0].axvline(x=np.median(ef_values), color='green', linestyle='--', label=f'Median: {np.median(ef_values):.1f}%')
            axes[0, 0].set_title('Ejection Fraction Distribution')
            axes[0, 0].set_xlabel('Ejection Fraction (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No EF data', ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # 2. Pulmonary artery pressure analysis
        pa_values = [m['pulmonary_artery_pressure_continuous'] for m in self.all_metrics 
                    if m['pulmonary_artery_pressure_continuous'] > 0]
        if pa_values:
            axes[0, 1].hist(pa_values, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0, 1].axvline(x=25, color='red', linestyle='--', label='Elevated Threshold (25 mmHg)')
            axes[0, 1].axvline(x=np.mean(pa_values), color='blue', linestyle='--', label=f'Mean: {np.mean(pa_values):.1f}')
            axes[0, 1].set_title('Pulmonary Artery Pressure Distribution')
            axes[0, 1].set_xlabel('PA Pressure (mmHg)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No PA pressure data', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. Valve conditions prevalence
        valve_conditions = ['mitral_stenosis', 'mitral_regurgitation', 'aortic_stenosis', 
                           'aortic_regurgitation', 'tricuspid_stenosis', 'tricuspid_valve_regurgitation']
        valve_counts = []
        valve_labels = []
        
        for condition in valve_conditions:
            count = sum(1 for m in self.all_metrics if m.get(condition, 0) > 0)
            if count > 0:
                valve_counts.append(count)
                valve_labels.append(condition.replace('_', ' ').title())
        
        if valve_counts:
            axes[1, 0].bar(valve_labels, valve_counts, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Valve Conditions Prevalence')
            axes[1, 0].set_ylabel('Number of Cases')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add count labels
            for i, count in enumerate(valve_counts):
                axes[1, 0].text(i, count + max(valve_counts)*0.01, str(count), 
                               ha='center', va='bottom', fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'No valve conditions detected', ha='center', va='center', 
                           transform=axes[1, 0].transAxes)
        
        # 4. Chamber dilation analysis
        chamber_conditions = ['left_atrium_dilation', 'right_atrium_dilation', 'right_ventricle_dilation']
        chamber_counts = []
        chamber_labels = []
        
        for condition in chamber_conditions:
            count = sum(1 for m in self.all_metrics if m.get(condition, 0) > 0)
            if count > 0:
                chamber_counts.append(count)
                chamber_labels.append(condition.replace('_', ' ').title())
        
        if chamber_counts:
            axes[1, 1].bar(chamber_labels, chamber_counts, color='lightyellow', alpha=0.7)
            axes[1, 1].set_title('Chamber Dilation Prevalence')
            axes[1, 1].set_ylabel('Number of Cases')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add count labels
            for i, count in enumerate(chamber_counts):
                axes[1, 1].text(i, count + max(chamber_counts)*0.01, str(count), 
                               ha='center', va='bottom', fontsize=8)
        else:
            axes[1, 1].text(0.5, 0.5, 'No chamber dilation detected', ha='center', va='center', 
                           transform=axes[1, 1].transAxes)
        
        # 5. Device/intervention prevalence
        device_conditions = ['impella', 'pacemaker', 'mitraclip', 'tavr']
        device_counts = []
        device_labels = []
        
        for condition in device_conditions:
            count = sum(1 for m in self.all_metrics if m.get(condition, 0) > 0)
            if count > 0:
                device_counts.append(count)
                device_labels.append(condition.upper())
        
        if device_counts:
            axes[2, 0].bar(device_labels, device_counts, color='lightpink', alpha=0.7)
            axes[2, 0].set_title('Cardiac Devices/Interventions')
            axes[2, 0].set_ylabel('Number of Cases')
            axes[2, 0].grid(True, alpha=0.3)
            
            # Add count labels
            for i, count in enumerate(device_counts):
                axes[2, 0].text(i, count + max(device_counts)*0.01, str(count), 
                               ha='center', va='bottom', fontsize=8)
        else:
            axes[2, 0].text(0.5, 0.5, 'No devices/interventions detected', ha='center', va='center', 
                           transform=axes[2, 0].transAxes)
        
        # 6. EF vs other conditions correlation
        if ef_values:
            # Create correlation matrix for continuous variables
            continuous_metrics = ['ejection_fraction', 'pulmonary_artery_pressure_continuous']
            correlation_data = []
            
            for metric in continuous_metrics:
                values = [m[metric] for m in self.all_metrics if m[metric] > 0]
                if len(values) > 1:
                    correlation_data.append(values)
            
            if len(correlation_data) >= 2:
                # Pad shorter arrays to same length for correlation
                min_len = min(len(arr) for arr in correlation_data)
                correlation_data = [arr[:min_len] for arr in correlation_data]
                
                axes[2, 1].scatter(correlation_data[0], correlation_data[1], alpha=0.6)
                axes[2, 1].set_xlabel('Ejection Fraction (%)')
                axes[2, 1].set_ylabel('PA Pressure (mmHg)')
                axes[2, 1].set_title('EF vs PA Pressure Correlation')
                axes[2, 1].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                if len(correlation_data[0]) > 1:
                    corr = np.corrcoef(correlation_data[0], correlation_data[1])[0, 1]
                    axes[2, 1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[2, 1].transAxes,
                                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            else:
                axes[2, 1].text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center', 
                               transform=axes[2, 1].transAxes)
        else:
            axes[2, 1].text(0.5, 0.5, 'No EF data for correlation', ha='center', va='center', 
                           transform=axes[2, 1].transAxes)
        
        plt.tight_layout()
        metrics_path = os.path.join(self.output_dir, 'cardiac_metrics_analysis.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cardiac metrics analysis saved to: {metrics_path}")
    
    def create_folder_comparison_plots(self):
        """Create folder comparison visualizations."""
        print("Creating folder comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Folder Performance Comparison', fontsize=16)
        
        # 1. Processing success rate by folder
        folder_names = [f['folder'][:25] + '...' if len(f['folder']) > 25 else f['folder'] for f in self.folder_data[:15]]
        success_rates = [(f['num_processed']/f['num_files']*100) if f['num_files'] > 0 else 0 for f in self.folder_data[:15]]
        
        bars = axes[0, 0].barh(range(len(folder_names)), success_rates, color='lightgreen', alpha=0.7)
        axes[0, 0].set_yticks(range(len(folder_names)))
        axes[0, 0].set_yticklabels(folder_names, fontsize=8)
        axes[0, 0].set_xlabel('Processing Success Rate (%)')
        axes[0, 0].set_title('Processing Success Rate (Top 15 Folders)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Videos per folder distribution
        video_counts = [f['num_videos'] for f in self.folder_data]
        axes[0, 1].hist(video_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(video_counts), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(video_counts):.1f}')
        axes[0, 1].axvline(x=np.median(video_counts), color='green', linestyle='--', 
                          label=f'Median: {np.median(video_counts):.1f}')
        axes[0, 1].set_xlabel('Number of Videos')
        axes[0, 1].set_ylabel('Number of Folders')
        axes[0, 1].set_title('Videos per Folder Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Success rate vs video count scatter
        axes[1, 0].scatter(video_counts, [(f['num_processed']/f['num_files']*100) if f['num_files'] > 0 else 0 for f in self.folder_data], 
                          alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Number of Videos')
        axes[1, 0].set_ylabel('Processing Success Rate (%)')
        axes[1, 0].set_title('Success Rate vs Video Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add correlation
        success_rates_all = [(f['num_processed']/f['num_files']*100) if f['num_files'] > 0 else 0 for f in self.folder_data]
        if len(video_counts) > 1:
            correlation = np.corrcoef(video_counts, success_rates_all)[0, 1]
            axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                           transform=axes[1, 0].transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # 4. Error distribution across folders
        error_data = []
        folder_labels = []
        
        for folder in self.folder_data[:10]:  # Top 10 folders
            if 'error_stats' in folder:
                total_errors = sum(folder['error_stats']['error_counts'].values())
                error_data.append(total_errors)
                folder_labels.append(folder['folder'][:20] + '...' if len(folder['folder']) > 20 else folder['folder'])
        
        if error_data:
            axes[1, 1].bar(range(len(folder_labels)), error_data, color='orange', alpha=0.7)
            axes[1, 1].set_xticks(range(len(folder_labels)))
            axes[1, 1].set_xticklabels(folder_labels, rotation=45, ha='right', fontsize=8)
            axes[1, 1].set_ylabel('Total Errors')
            axes[1, 1].set_title('Error Count by Folder (Top 10)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No error data available', ha='center', va='center', 
                           transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        folder_comparison_path = os.path.join(self.output_dir, 'folder_comparison.png')
        plt.savefig(folder_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Folder comparison plots saved to: {folder_comparison_path}")
    
    def create_interactive_plots(self):
        """Create interactive Plotly visualizations."""
        print("Creating interactive visualizations...")
        
        try:
            # 1. Interactive cardiac metrics dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Ejection Fraction Distribution', 'Cardiac Conditions Prevalence',
                               'Folder Performance', 'Processing Success Rates'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # EF distribution
            ef_values = [m['ejection_fraction'] for m in self.all_metrics if m['ejection_fraction'] > 0]
            if ef_values:
                fig.add_trace(
                    go.Histogram(x=ef_values, name='EF Distribution', nbinsx=20),
                    row=1, col=1
                )
            
            # Cardiac conditions
            binary_metrics = [m for m in self.cardiac_metrics if m not in ['ejection_fraction', 'pulmonary_artery_pressure_continuous']]
            condition_counts = []
            condition_names = []
            
            for metric in binary_metrics[:10]:  # Top 10
                count = sum(1 for m in self.all_metrics if m.get(metric, 0) > 0)
                if count > 0:
                    condition_counts.append(count)
                    condition_names.append(metric.replace('_', ' ').title())
            
            if condition_counts:
                fig.add_trace(
                    go.Bar(x=condition_names, y=condition_counts, name='Conditions'),
                    row=1, col=2
                )
            
            # Folder performance
            video_counts = [f['num_videos'] for f in self.folder_data]
            success_rates = [(f['num_processed']/f['num_files']*100) if f['num_files'] > 0 else 0 for f in self.folder_data]
            folder_names = [f['folder'] for f in self.folder_data]
            
            if video_counts and success_rates:
                fig.add_trace(
                    go.Scatter(x=video_counts, y=success_rates, mode='markers',
                             text=folder_names, name='Folder Performance'),
                    row=2, col=1
                )
            
            # Processing success rates
            if success_rates:
                fig.add_trace(
                    go.Histogram(x=success_rates, name='Success Rate Distribution', nbinsx=20),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='Interactive EchoPrime Analysis Dashboard',
                height=800,
                showlegend=False
            )
            
            interactive_dashboard_path = os.path.join(self.output_dir, 'interactive_dashboard.html')
            fig.write_html(interactive_dashboard_path)
            print(f"Interactive dashboard saved to: {interactive_dashboard_path}")
            
        except ImportError:
            print("Plotly not available. Skipping interactive visualizations.")
            print("Install plotly with: pip install plotly")
    
    def create_summary_report(self):
        """Create a text summary report."""
        print("Creating summary report...")
        
        report_lines = []
        report_lines.append("EchoPrime Cardiac Phenotype Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 20)
        report_lines.append(f"Total Folders: {self.summary_data['total_folders']}")
        report_lines.append(f"Successful Folders: {self.summary_data['successful']}")
        report_lines.append(f"Failed Folders: {self.summary_data['failed']}")
        report_lines.append(f"Total Videos: {self.summary_data['total_videos']}")
        report_lines.append(f"Success Rate: {(self.summary_data['successful']/self.summary_data['total_folders']*100):.1f}%")
        report_lines.append("")
        
        # Cardiac metrics summary
        ef_values = [m['ejection_fraction'] for m in self.all_metrics if m['ejection_fraction'] > 0]
        pa_values = [m['pulmonary_artery_pressure_continuous'] for m in self.all_metrics 
                    if m['pulmonary_artery_pressure_continuous'] > 0]
        
        report_lines.append("CARDIAC METRICS SUMMARY")
        report_lines.append("-" * 25)
        
        if ef_values:
            report_lines.append(f"Ejection Fraction - Cases: {len(ef_values)}")
            report_lines.append(f"  Mean EF: {np.mean(ef_values):.1f}%")
            report_lines.append(f"  Median EF: {np.median(ef_values):.1f}%")
            report_lines.append(f"  Range: {np.min(ef_values):.1f}% - {np.max(ef_values):.1f}%")
            report_lines.append(f"  Normal EF (≥50%): {sum(1 for ef in ef_values if ef >= 50)} ({sum(1 for ef in ef_values if ef >= 50)/len(ef_values)*100:.1f}%)")
        else:
            report_lines.append("Ejection Fraction - No data available")
        
        if pa_values:
            report_lines.append(f"PA Pressure - Cases: {len(pa_values)}")
            report_lines.append(f"  Mean PA Pressure: {np.mean(pa_values):.1f} mmHg")
            report_lines.append(f"  Median PA Pressure: {np.median(pa_values):.1f} mmHg")
            report_lines.append(f"  Range: {np.min(pa_values):.1f} - {np.max(pa_values):.1f} mmHg")
            report_lines.append(f"  Elevated PA (≥25 mmHg): {sum(1 for pa in pa_values if pa >= 25)} ({sum(1 for pa in pa_values if pa >= 25)/len(pa_values)*100:.1f}%)")
        else:
            report_lines.append("PA Pressure - No data available")
        
        report_lines.append("")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to: {report_path}")
    
    def create_device_analytics(self):
        """Create comprehensive per-device analytics visualizations."""
        print("Creating device analytics...")
        
        if not self.device_data:
            print("No device data available for analytics")
            return
        
        # Create comprehensive device analytics with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Comprehensive Per-Device Analytics Dashboard', fontsize=16)
        
        device_names = list(self.device_data.keys())
        device_success_rates = []
        device_error_rates = []
        device_video_counts = []
        device_mean_ef = []
        device_file_counts = []
        device_processing_rates = []
        
        # Calculate comprehensive device statistics
        for device in device_names:
            device_metrics = self.device_data[device]
            
            # Processing metrics
            success_rates = [m['processing_success_rate'] * 100 for m in device_metrics]
            error_rates = [m['error_rate'] * 100 for m in device_metrics]
            video_counts = [m['num_videos'] for m in device_metrics]
            file_counts = [m['num_files'] for m in device_metrics]
            
            # Cardiac metrics
            ef_values = [m['ejection_fraction'] for m in device_metrics if m['ejection_fraction'] > 0]
            
            device_success_rates.append(np.mean(success_rates) if success_rates else 0)
            device_error_rates.append(np.mean(error_rates) if error_rates else 0)
            device_video_counts.append(np.sum(video_counts) if video_counts else 0)
            device_file_counts.append(np.sum(file_counts) if file_counts else 0)
            device_mean_ef.append(np.mean(ef_values) if ef_values else 0)
            
            # Processing efficiency
            total_processed = sum(m['num_processed'] for m in device_metrics)
            total_files = sum(m['num_files'] for m in device_metrics)
            processing_rate = (total_processed / total_files * 100) if total_files > 0 else 0
            device_processing_rates.append(processing_rate)
        
        # Plot 1: Processing success rates by device
        bars = axes[0, 0].bar(device_names, device_success_rates, color='lightgreen', alpha=0.7)
        axes[0, 0].set_title('Processing Success Rate by Device')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        for bar, rate in zip(bars, device_success_rates):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Error rates by device
        bars = axes[0, 1].bar(device_names, device_error_rates, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Error Rate by Device')
        axes[0, 1].set_ylabel('Error Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, rate in zip(bars, device_error_rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(device_error_rates)*0.01,
                           f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Total videos processed by device
        bars = axes[1, 0].bar(device_names, device_video_counts, color='skyblue', alpha=0.7)
        axes[1, 0].set_title('Total Videos Processed by Device')
        axes[1, 0].set_ylabel('Number of Videos')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, count in zip(bars, device_video_counts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(device_video_counts)*0.01,
                           str(count), ha='center', va='bottom', fontsize=10)
        
        # Plot 4: Total files processed by device
        bars = axes[1, 1].bar(device_names, device_file_counts, color='orange', alpha=0.7)
        axes[1, 1].set_title('Total Files Processed by Device')
        axes[1, 1].set_ylabel('Number of Files')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, count in zip(bars, device_file_counts):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(device_file_counts)*0.01,
                           str(count), ha='center', va='bottom', fontsize=10)
        
        # Plot 5: Mean Ejection Fraction by device
        bars = axes[2, 0].bar(device_names, device_mean_ef, color='lightpink', alpha=0.7)
        axes[2, 0].set_title('Mean Ejection Fraction by Device')
        axes[2, 0].set_ylabel('Mean EF (%)')
        axes[2, 0].tick_params(axis='x', rotation=45)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Normal Threshold')
        axes[2, 0].legend()
        
        for bar, ef in zip(bars, device_mean_ef):
            if ef > 0:
                axes[2, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{ef:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 6: EF distribution by device (box plot)
        ef_by_device = {}
        for device in device_names:
            device_metrics = self.device_data[device]
            ef_values = [m['ejection_fraction'] for m in device_metrics if m['ejection_fraction'] > 0]
            if ef_values:
                ef_by_device[device] = ef_values
        
        if ef_by_device:
            ef_data = [ef_by_device[device] for device in ef_by_device.keys()]
            ef_labels = list(ef_by_device.keys())
            
            axes[2, 1].boxplot(ef_data, labels=ef_labels)
            axes[2, 1].set_title('EF Distribution by Device')
            axes[2, 1].set_ylabel('Ejection Fraction (%)')
            axes[2, 1].tick_params(axis='x', rotation=45)
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Normal Threshold')
            axes[2, 1].legend()
        else:
            axes[2, 1].text(0.5, 0.5, 'No EF data available by device', ha='center', va='center',
                           transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('EF Distribution by Device')
        
        plt.tight_layout()
        device_analytics_path = os.path.join(self.output_dir, 'device_analytics.png')
        plt.savefig(device_analytics_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Device analytics saved to: {device_analytics_path}")
        
        # Create additional device-specific cardiac metrics analysis
        self.create_device_cardiac_metrics_analysis()
    
    def create_device_cardiac_metrics_analysis(self):
        """Create detailed device-specific cardiac metrics analysis."""
        print("Creating device-specific cardiac metrics analysis...")
        
        if not self.device_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Device-Specific Cardiac Metrics Analysis', fontsize=16)
        
        device_names = list(self.device_data.keys())
        
        # 1. Cardiac conditions prevalence by device
        condition_by_device = {}
        key_conditions = ['mitral_regurgitation', 'aortic_stenosis', 'left_atrium_dilation', 'pacemaker']
        
        for condition in key_conditions:
            condition_by_device[condition] = []
            for device in device_names:
                device_metrics = self.device_data[device]
                count = sum(1 for m in device_metrics if m.get(condition, 0) > 0)
                total = len(device_metrics)
                prevalence = (count / total * 100) if total > 0 else 0
                condition_by_device[condition].append(prevalence)
        
        # Plot condition prevalence
        x = np.arange(len(device_names))
        width = 0.2
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (condition, prevalences) in enumerate(condition_by_device.items()):
            axes[0, 0].bar(x + i*width, prevalences, width, 
                          label=condition.replace('_', ' ').title(), 
                          color=colors[i], alpha=0.7)
        
        axes[0, 0].set_title('Cardiac Conditions Prevalence by Device')
        axes[0, 0].set_ylabel('Prevalence (%)')
        axes[0, 0].set_xlabel('Device')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(device_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. EF statistics by device
        device_ef_stats = []
        for device in device_names:
            device_metrics = self.device_data[device]
            ef_values = [m['ejection_fraction'] for m in device_metrics if m['ejection_fraction'] > 0]
            if ef_values:
                device_ef_stats.append({
                    'device': device,
                    'mean': np.mean(ef_values),
                    'median': np.median(ef_values),
                    'std': np.std(ef_values),
                    'count': len(ef_values)
                })
        
        if device_ef_stats:
            devices = [stat['device'] for stat in device_ef_stats]
            means = [stat['mean'] for stat in device_ef_stats]
            stds = [stat['std'] for stat in device_ef_stats]
            
            axes[0, 1].bar(devices, means, yerr=stds, capsize=5, 
                          color='lightcoral', alpha=0.7, error_kw={'elinewidth': 2})
            axes[0, 1].set_title('Mean EF by Device (with Std Dev)')
            axes[0, 1].set_ylabel('Ejection Fraction (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Normal Threshold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Processing efficiency vs cardiac metrics
        device_processing_eff = []
        device_normal_ef_rate = []
        
        for device in device_names:
            device_metrics = self.device_data[device]
            
            # Processing efficiency
            total_processed = sum(m['num_processed'] for m in device_metrics)
            total_files = sum(m['num_files'] for m in device_metrics)
            eff = (total_processed / total_files * 100) if total_files > 0 else 0
            device_processing_eff.append(eff)
            
            # Normal EF rate
            ef_values = [m['ejection_fraction'] for m in device_metrics if m['ejection_fraction'] > 0]
            normal_ef_count = sum(1 for ef in ef_values if ef >= 50)
            normal_ef_rate = (normal_ef_count / len(ef_values) * 100) if ef_values else 0
            device_normal_ef_rate.append(normal_ef_rate)
        
        if device_processing_eff and device_normal_ef_rate:
            axes[1, 0].scatter(device_processing_eff, device_normal_ef_rate, 
                             s=100, alpha=0.7, c='purple')
            
            # Add device labels
            for i, device in enumerate(device_names):
                axes[1, 0].annotate(device, 
                                   (device_processing_eff[i], device_normal_ef_rate[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            axes[1, 0].set_xlabel('Processing Efficiency (%)')
            axes[1, 0].set_ylabel('Normal EF Rate (%)')
            axes[1, 0].set_title('Processing Efficiency vs Normal EF Rate')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add correlation
            if len(device_processing_eff) > 1:
                corr = np.corrcoef(device_processing_eff, device_normal_ef_rate)[0, 1]
                axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                               transform=axes[1, 0].transAxes, 
                               bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # 4. Device performance summary table
        axes[1, 1].axis('off')
        
        if device_ef_stats:
            table_data = []
            for stat in device_ef_stats:
                device_metrics = self.device_data[stat['device']]
                total_videos = sum(m['num_videos'] for m in device_metrics)
                
                table_data.append([
                    stat['device'][:10],  # Truncate device name
                    f"{stat['mean']:.1f}%",
                    f"{stat['count']}",
                    f"{total_videos}"
                ])
            
            table = axes[1, 1].table(cellText=table_data,
                                   colLabels=['Device', 'Mean EF', 'EF Cases', 'Videos'],
                                   cellLoc='center',
                                   loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            axes[1, 1].set_title('Device Performance Summary', fontsize=14, pad=20)
        
        plt.tight_layout()
        device_cardiac_path = os.path.join(self.output_dir, 'device_cardiac_metrics.png')
        plt.savefig(device_cardiac_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Device cardiac metrics analysis saved to: {device_cardiac_path}")
    
    def create_batch_quality_assessment(self):
        """Create enhanced batch quality assessment visualizations."""
        print("Creating batch quality assessment...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Batch Quality Assessment', fontsize=16)
        
        # 1. Median failures per files analysis
        error_rates = [m['error_rate'] for m in self.all_metrics]
        if error_rates:
            median_error_rate = np.median(error_rates)
            axes[0, 0].hist(error_rates, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[0, 0].axvline(x=median_error_rate, color='red', linestyle='--', 
                             label=f'Median: {median_error_rate:.3f}')
            axes[0, 0].axvline(x=np.mean(error_rates), color='blue', linestyle='--', 
                             label=f'Mean: {np.mean(error_rates):.3f}')
            axes[0, 0].set_title('Error Rate Distribution\n(Failures per Files)')
            axes[0, 0].set_xlabel('Error Rate')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No error data available', ha='center', va='center',
                           transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Error Rate Distribution')
        
        # 2. Processing efficiency distribution
        processing_rates = [m['processing_success_rate'] for m in self.all_metrics]
        if processing_rates:
            axes[0, 1].hist(processing_rates, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].axvline(x=np.median(processing_rates), color='red', linestyle='--', 
                             label=f'Median: {np.median(processing_rates):.3f}')
            axes[0, 1].axvline(x=np.mean(processing_rates), color='blue', linestyle='--', 
                             label=f'Mean: {np.mean(processing_rates):.3f}')
            axes[0, 1].set_title('Processing Success Rate Distribution')
            axes[0, 1].set_xlabel('Processing Success Rate')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Batch quality score (composite metric)
        quality_scores = []
        for m in self.all_metrics:
            # Composite quality score: processing success rate - error rate + EF normalization
            ef_score = 1.0 if m['ejection_fraction'] >= 50 else 0.5 if m['ejection_fraction'] > 0 else 0.0
            quality_score = (m['processing_success_rate'] * 0.5) + ((1 - m['error_rate']) * 0.3) + (ef_score * 0.2)
            quality_scores.append(quality_score)
        
        if quality_scores:
            axes[0, 2].hist(quality_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[0, 2].axvline(x=np.median(quality_scores), color='red', linestyle='--', 
                             label=f'Median: {np.median(quality_scores):.3f}')
            axes[0, 2].axvline(x=np.mean(quality_scores), color='blue', linestyle='--', 
                             label=f'Mean: {np.mean(quality_scores):.3f}')
            axes[0, 2].set_title('Composite Batch Quality Score')
            axes[0, 2].set_xlabel('Quality Score (0-1)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Error type breakdown with percentages
        total_errors = {}
        for folder in self.folder_data:
            if 'error_stats' in folder:
                for error_type, count in folder['error_stats']['error_counts'].items():
                    total_errors[error_type] = total_errors.get(error_type, 0) + count
        
        if total_errors and sum(total_errors.values()) > 0:
            error_types = list(total_errors.keys())
            error_counts = list(total_errors.values())
            total_error_count = sum(error_counts)
            
            # Create pie chart
            axes[1, 0].pie(error_counts, labels=error_types, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title(f'Error Type Breakdown\n(Total: {total_error_count} errors)')
        else:
            axes[1, 0].text(0.5, 0.5, 'No errors found', ha='center', va='center',
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Error Type Breakdown')
        
        # 5. Quality trends across folders
        folder_quality_scores = []
        folder_names_short = []
        for i, m in enumerate(self.all_metrics):
            if i < 20:  # Limit to first 20 for readability
                ef_score = 1.0 if m['ejection_fraction'] >= 50 else 0.5 if m['ejection_fraction'] > 0 else 0.0
                quality_score = (m['processing_success_rate'] * 0.5) + ((1 - m['error_rate']) * 0.3) + (ef_score * 0.2)
                folder_quality_scores.append(quality_score)
                folder_name = m['folder'][:15] + '...' if len(m['folder']) > 15 else m['folder']
                folder_names_short.append(folder_name)
        
        if folder_quality_scores:
            bars = axes[1, 1].bar(range(len(folder_quality_scores)), folder_quality_scores, 
                                color='lightblue', alpha=0.7)
            axes[1, 1].set_title('Quality Score by Folder (First 20)')
            axes[1, 1].set_xlabel('Folders')
            axes[1, 1].set_ylabel('Quality Score')
            axes[1, 1].set_xticks(range(len(folder_names_short)))
            axes[1, 1].set_xticklabels(folder_names_short, rotation=45, ha='right', fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add horizontal line for median quality
            median_quality = np.median(quality_scores)
            axes[1, 1].axhline(y=median_quality, color='red', linestyle='--', alpha=0.7,
                             label=f'Batch Median: {median_quality:.3f}')
            axes[1, 1].legend()
        
        # 6. Correlation matrix of quality metrics
        if len(self.all_metrics) > 1:
            # Prepare data for correlation
            correlation_data = []
            metric_names = ['Processing Success', 'Error Rate', 'EF Available', 'Video Count']
            
            for m in self.all_metrics:
                correlation_data.append([
                    m['processing_success_rate'],
                    m['error_rate'],
                    1.0 if m['ejection_fraction'] > 0 else 0.0,
                    m['num_videos'] / 100.0  # Normalize video count
                ])
            
            correlation_matrix = np.corrcoef(np.array(correlation_data).T)
            
            im = axes[1, 2].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 2].set_title('Quality Metrics Correlation')
            axes[1, 2].set_xticks(range(len(metric_names)))
            axes[1, 2].set_yticks(range(len(metric_names)))
            axes[1, 2].set_xticklabels(metric_names, rotation=45, ha='right')
            axes[1, 2].set_yticklabels(metric_names)
            
            # Add correlation values to the plot
            for i in range(len(metric_names)):
                for j in range(len(metric_names)):
                    text = axes[1, 2].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                          ha="center", va="center", color="black", fontsize=8)
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center',
                           transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Quality Metrics Correlation')
        
        plt.tight_layout()
        batch_quality_path = os.path.join(self.output_dir, 'batch_quality_assessment.png')
        plt.savefig(batch_quality_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Batch quality assessment saved to: {batch_quality_path}")
    
    def run_complete_visualization(self):
        """Run the complete visualization pipeline."""
        print("Starting comprehensive EchoPrime visualization...")
        print("=" * 60)
        
        try:
            # Load data
            self.load_data()
            
            # Create all visualizations
            self.create_overall_summary_dashboard()
            self.create_cardiac_metrics_analysis()
            self.create_folder_comparison_plots()
            self.create_device_analytics()
            self.create_batch_quality_assessment()
            self.create_interactive_plots()
            self.create_summary_report()
            
            print("\n" + "=" * 60)
            print("Visualization complete! Check the output directory for results:")
            print(f"  {os.path.abspath(self.output_dir)}")
            
            # List generated files
            generated_files = []
            for file in os.listdir(self.output_dir):
                if file.endswith(('.png', '.html', '.txt', '.csv')):
                    generated_files.append(file)
            
            if generated_files:
                print("\nGenerated files:")
                for file in sorted(generated_files):
                    print(f"  - {file}")
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Visualize EchoPrime inference results')
    parser.add_argument('--results_dir', type=str, default='results/inference_output',
                        help='Directory containing inference results (default: results/inference_output)')
    parser.add_argument('--output_dir', type=str, default='results/visualization_output',
                        help='Directory to save visualization results (default: results/visualization_output)')
    
    args = parser.parse_args()
    
    # Create visualizer and run visualization
    visualizer = EchoPrimeVisualizer(args.results_dir, args.output_dir)
    visualizer.run_complete_visualization()


if __name__ == "__main__":
    main()
