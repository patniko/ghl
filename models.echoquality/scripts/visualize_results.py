#!/usr/bin/env python3
"""
Comprehensive visualization script for EchoQuality inference results.

This script creates detailed visualizations of the quality assessment results,
including score distributions, pass/fail rates, and comparative analyses across folders.

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


class EchoQualityVisualizer:
    """Comprehensive visualizer for EchoQuality inference results."""
    
    def __init__(self, results_dir='results/inference_output', output_dir='visualization_output'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.summary_data = None
        self.folder_data = []
        self.all_results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load summary and individual folder results."""
        print("Loading EchoQuality inference results for visualization...")
        
        # Load summary data
        summary_path = os.path.join(self.results_dir, 'summary.json')
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        
        with open(summary_path, 'r') as f:
            self.summary_data = json.load(f)
        
        print(f"Found {self.summary_data['total_folders']} folders with {self.summary_data['total_files']} total files")
        print(f"Overall pass rate: {self.summary_data['overall_pass_rate']:.2f}%")
        
        # Load individual folder data
        for folder_result in self.summary_data['folder_results']:
            folder_name = folder_result['folder']
            folder_path = os.path.join(self.results_dir, folder_name)
            
            # Load detailed results if available
            results_file = os.path.join(folder_path, 'inference_results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    detailed_results = json.load(f)
                    folder_result['detailed_results'] = detailed_results
            
            self.folder_data.append(folder_result)
            
            # Collect all individual results for global analysis
            if 'results' in folder_result:
                for file_id, result in folder_result['results'].items():
                    result_entry = {
                        'folder': folder_name,
                        'file_id': file_id,
                        'score': result['score'],
                        'status': result['status'],
                        'assessment': result['assessment'],
                        'path': result['path']
                    }
                    self.all_results.append(result_entry)
        
        print(f"Loaded detailed results for {len(self.folder_data)} folders")
        print(f"Total individual results: {len(self.all_results)}")
    
    def create_overall_summary_dashboard(self):
        """Create a comprehensive summary dashboard."""
        print("Creating overall summary dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('EchoQuality Analysis Dashboard', fontsize=20, y=0.98)
        
        # 1. Overall statistics (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        stats_data = [
            ['Total Folders', f"{self.summary_data['total_folders']}"],
            ['Total Files', f"{self.summary_data['total_files']}"],
            ['Processed Files', f"{self.summary_data['total_processed']}"],
            ['Pass Count', f"{self.summary_data['total_pass']}"],
            ['Fail Count', f"{self.summary_data['total_fail']}"],
            ['Overall Pass Rate', f"{self.summary_data['overall_pass_rate']:.2f}%"]
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
        
        # 2. Score distribution (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        scores = [result['score'] for result in self.all_results]
        ax2.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Pass/Fail Threshold')
        ax2.set_title('Quality Score Distribution', fontsize=14)
        ax2.set_xlabel('Quality Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Pass/Fail by folder (middle-left)
        ax3 = fig.add_subplot(gs[1, :2])
        folder_names = [folder['folder'] for folder in self.folder_data[:20]]  # Top 20 folders
        pass_rates = [folder['pass_rate'] for folder in self.folder_data[:20]]
        
        bars = ax3.barh(range(len(folder_names)), pass_rates, color='lightgreen', alpha=0.7)
        ax3.set_yticks(range(len(folder_names)))
        ax3.set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in folder_names], fontsize=8)
        ax3.set_xlabel('Pass Rate (%)')
        ax3.set_title('Pass Rate by Folder (Top 20)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add pass rate labels on bars
        for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
            ax3.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=8)
        
        # 4. Score distribution by status (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        pass_scores = [result['score'] for result in self.all_results if result['status'] == 'PASS']
        fail_scores = [result['score'] for result in self.all_results if result['status'] == 'FAIL']
        
        ax4.hist(pass_scores, bins=30, alpha=0.7, label='PASS', color='green', density=True)
        ax4.hist(fail_scores, bins=30, alpha=0.7, label='FAIL', color='red', density=True)
        ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax4.set_title('Score Distribution by Status', fontsize=14)
        ax4.set_xlabel('Quality Score')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Error analysis (bottom-left)
        ax5 = fig.add_subplot(gs[2, :2])
        error_types = []
        error_counts = []
        
        for folder in self.folder_data:
            if 'error_stats' in folder:
                for error_type, count in folder['error_stats']['error_counts'].items():
                    if count > 0:
                        error_types.append(error_type)
                        error_counts.append(count)
        
        if error_types:
            # Aggregate error counts
            error_summary = {}
            for error_type, count in zip(error_types, error_counts):
                error_summary[error_type] = error_summary.get(error_type, 0) + count
            
            ax5.bar(error_summary.keys(), error_summary.values(), color='orange', alpha=0.7)
            ax5.set_title('Error Type Distribution', fontsize=14)
            ax5.set_xlabel('Error Type')
            ax5.set_ylabel('Count')
            ax5.tick_params(axis='x', rotation=45)
        else:
            ax5.text(0.5, 0.5, 'No errors found', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Error Type Distribution', fontsize=14)
        
        # 6. Quality assessment distribution (bottom-right)
        ax6 = fig.add_subplot(gs[2, 2:])
        assessments = [result['assessment'] for result in self.all_results]
        assessment_counts = {}
        for assessment in assessments:
            assessment_counts[assessment] = assessment_counts.get(assessment, 0) + 1
        
        # Truncate long assessment names for display
        display_names = []
        for name in assessment_counts.keys():
            if len(name) > 30:
                display_names.append(name[:27] + '...')
            else:
                display_names.append(name)
        
        ax6.pie(assessment_counts.values(), labels=display_names, autopct='%1.1f%%', startangle=90)
        ax6.set_title('Quality Assessment Distribution', fontsize=14)
        
        # 7. Processing statistics (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        
        # Calculate processing stats
        total_files = self.summary_data['total_files']
        processed_files = self.summary_data['total_processed']
        unprocessed_files = total_files - processed_files
        
        processing_data = ['Processed', 'Unprocessed']
        processing_counts = [processed_files, unprocessed_files]
        processing_colors = ['lightblue', 'lightcoral']
        
        # Create horizontal bar chart
        bars = ax7.barh(processing_data, processing_counts, color=processing_colors, alpha=0.7)
        
        # Add count labels
        for bar, count in zip(bars, processing_counts):
            width = bar.get_width()
            ax7.text(width + total_files * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{count} ({count/total_files*100:.1f}%)', 
                    ha='left', va='center', fontsize=12)
        
        ax7.set_xlabel('Number of Files')
        ax7.set_title('File Processing Statistics', fontsize=14)
        ax7.grid(True, alpha=0.3)
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, 'echoquality_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary dashboard saved to: {dashboard_path}")
    
    def create_score_analysis_plots(self):
        """Create detailed score analysis visualizations."""
        print("Creating score analysis plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('EchoQuality Score Analysis', fontsize=16)
        
        scores = [result['score'] for result in self.all_results]
        
        # 1. Score histogram with statistics
        axes[0, 0].hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        axes[0, 0].axvline(np.median(scores), color='green', linestyle='--', label=f'Median: {np.median(scores):.3f}')
        axes[0, 0].axvline(0.5, color='orange', linestyle='--', label='Threshold: 0.5')
        axes[0, 0].set_title('Score Distribution with Statistics')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot by status
        pass_scores = [result['score'] for result in self.all_results if result['status'] == 'PASS']
        fail_scores = [result['score'] for result in self.all_results if result['status'] == 'FAIL']
        
        axes[0, 1].boxplot([pass_scores, fail_scores], labels=['PASS', 'FAIL'])
        axes[0, 1].set_title('Score Distribution by Status')
        axes[0, 1].set_ylabel('Quality Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        sorted_scores = np.sort(scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[0, 2].plot(sorted_scores, cumulative, linewidth=2)
        axes[0, 2].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 2].set_title('Cumulative Score Distribution')
        axes[0, 2].set_xlabel('Quality Score')
        axes[0, 2].set_ylabel('Cumulative Probability')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Score ranges analysis
        score_ranges = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', 
                       '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        range_counts = []
        
        for i in range(10):
            lower = i * 0.1
            upper = (i + 1) * 0.1
            count = sum(1 for score in scores if lower <= score < upper)
            range_counts.append(count)
        
        # Handle edge case for score = 1.0
        range_counts[-1] += sum(1 for score in scores if score == 1.0)
        
        bars = axes[1, 0].bar(score_ranges, range_counts, alpha=0.7, color='lightcoral')
        axes[1, 0].set_title('Score Distribution by Ranges')
        axes[1, 0].set_xlabel('Score Range')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, range_counts):
            if count > 0:
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(range_counts)*0.01,
                               str(count), ha='center', va='bottom', fontsize=8)
        
        # 5. Top and bottom scoring files
        sorted_results = sorted(self.all_results, key=lambda x: x['score'])
        
        # Top 10 scores
        top_scores = [result['score'] for result in sorted_results[-10:]]
        top_labels = [f"...{result['file_id'][-20:]}" for result in sorted_results[-10:]]
        
        axes[1, 1].barh(range(len(top_scores)), top_scores, color='green', alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_scores)))
        axes[1, 1].set_yticklabels(top_labels, fontsize=8)
        axes[1, 1].set_title('Top 10 Quality Scores')
        axes[1, 1].set_xlabel('Quality Score')
        
        # 6. Bottom 10 scores
        bottom_scores = [result['score'] for result in sorted_results[:10]]
        bottom_labels = [f"...{result['file_id'][-20:]}" for result in sorted_results[:10]]
        
        axes[1, 2].barh(range(len(bottom_scores)), bottom_scores, color='red', alpha=0.7)
        axes[1, 2].set_yticks(range(len(bottom_scores)))
        axes[1, 2].set_yticklabels(bottom_labels, fontsize=8)
        axes[1, 2].set_title('Bottom 10 Quality Scores')
        axes[1, 2].set_xlabel('Quality Score')
        
        plt.tight_layout()
        score_analysis_path = os.path.join(self.output_dir, 'score_analysis.png')
        plt.savefig(score_analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Score analysis plots saved to: {score_analysis_path}")
    
    def create_folder_comparison_plots(self):
        """Create folder comparison visualizations."""
        print("Creating folder comparison plots...")
        
        # Prepare folder statistics
        folder_stats = []
        for folder in self.folder_data:
            if folder['num_processed'] > 0:  # Only include folders with processed files
                folder_stats.append({
                    'folder': folder['folder'],
                    'pass_rate': folder['pass_rate'],
                    'num_files': folder['num_files'],
                    'num_processed': folder['num_processed'],
                    'pass_count': folder['pass_count'],
                    'fail_count': folder['fail_count']
                })
        
        # Sort by pass rate
        folder_stats.sort(key=lambda x: x['pass_rate'], reverse=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Folder Comparison Analysis', fontsize=16)
        
        # 1. Pass rate comparison (top folders)
        top_folders = folder_stats[:20]
        folder_names = [f['folder'][:25] + '...' if len(f['folder']) > 25 else f['folder'] for f in top_folders]
        pass_rates = [f['pass_rate'] for f in top_folders]
        
        bars = axes[0, 0].barh(range(len(folder_names)), pass_rates, color='lightgreen', alpha=0.7)
        axes[0, 0].set_yticks(range(len(folder_names)))
        axes[0, 0].set_yticklabels(folder_names, fontsize=8)
        axes[0, 0].set_xlabel('Pass Rate (%)')
        axes[0, 0].set_title('Top 20 Folders by Pass Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add pass rate labels
        for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
            axes[0, 0].text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=7)
        
        # 2. File count vs pass rate scatter
        file_counts = [f['num_processed'] for f in folder_stats]
        pass_rates_all = [f['pass_rate'] for f in folder_stats]
        
        scatter = axes[0, 1].scatter(file_counts, pass_rates_all, alpha=0.6, s=50)
        axes[0, 1].set_xlabel('Number of Processed Files')
        axes[0, 1].set_ylabel('Pass Rate (%)')
        axes[0, 1].set_title('Pass Rate vs File Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(file_counts, pass_rates_all)[0, 1]
        axes[0, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[0, 1].transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # 3. Pass/Fail distribution across folders
        pass_counts = [f['pass_count'] for f in folder_stats[:15]]
        fail_counts = [f['fail_count'] for f in folder_stats[:15]]
        folder_names_short = [f['folder'][:20] + '...' if len(f['folder']) > 20 else f['folder'] for f in folder_stats[:15]]
        
        x = np.arange(len(folder_names_short))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, pass_counts, width, label='Pass', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, fail_counts, width, label='Fail', color='red', alpha=0.7)
        
        axes[1, 0].set_xlabel('Folders')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Pass/Fail Distribution (Top 15 Folders)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(folder_names_short, rotation=45, ha='right', fontsize=8)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Pass rate distribution histogram
        all_pass_rates = [f['pass_rate'] for f in folder_stats]
        
        axes[1, 1].hist(all_pass_rates, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(np.mean(all_pass_rates), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_pass_rates):.1f}%')
        axes[1, 1].axvline(np.median(all_pass_rates), color='green', linestyle='--', 
                          label=f'Median: {np.median(all_pass_rates):.1f}%')
        axes[1, 1].set_xlabel('Pass Rate (%)')
        axes[1, 1].set_ylabel('Number of Folders')
        axes[1, 1].set_title('Distribution of Folder Pass Rates')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        folder_comparison_path = os.path.join(self.output_dir, 'folder_comparison.png')
        plt.savefig(folder_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Folder comparison plots saved to: {folder_comparison_path}")
    
    def create_interactive_plots(self):
        """Create interactive Plotly visualizations."""
        print("Creating interactive visualizations...")
        
        try:
            # 1. Interactive score distribution
            scores = [result['score'] for result in self.all_results]
            statuses = [result['status'] for result in self.all_results]
            folders = [result['folder'] for result in self.all_results]
            
            fig = go.Figure()
            
            # Add histogram for all scores
            fig.add_trace(go.Histogram(
                x=scores,
                nbinsx=50,
                name='All Scores',
                opacity=0.7,
                marker_color='skyblue'
            ))
            
            fig.update_layout(
                title='Interactive Quality Score Distribution',
                xaxis_title='Quality Score',
                yaxis_title='Frequency',
                width=1000,
                height=600
            )
            
            # Add threshold line
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                         annotation_text="Pass/Fail Threshold")
            
            interactive_scores_path = os.path.join(self.output_dir, 'interactive_score_distribution.html')
            fig.write_html(interactive_scores_path)
            print(f"Interactive score distribution saved to: {interactive_scores_path}")
            
            # 2. Interactive folder comparison
            folder_stats = []
            for folder in self.folder_data:
                if folder['num_processed'] > 0:
                    folder_stats.append({
                        'folder': folder['folder'],
                        'pass_rate': folder['pass_rate'],
                        'num_processed': folder['num_processed'],
                        'pass_count': folder['pass_count'],
                        'fail_count': folder['fail_count']
                    })
            
            df = pd.DataFrame(folder_stats)
            
            fig = px.scatter(df, x='num_processed', y='pass_rate', 
                           hover_data=['folder', 'pass_count', 'fail_count'],
                           title='Interactive Folder Analysis: Pass Rate vs File Count',
                           labels={'num_processed': 'Number of Processed Files',
                                  'pass_rate': 'Pass Rate (%)'})
            
            fig.update_layout(width=1000, height=600)
            
            interactive_folders_path = os.path.join(self.output_dir, 'interactive_folder_analysis.html')
            fig.write_html(interactive_folders_path)
            print(f"Interactive folder analysis saved to: {interactive_folders_path}")
            
            # 3. Interactive heatmap of folder performance
            if len(folder_stats) > 1:
                # Create a matrix for heatmap
                top_folders = sorted(folder_stats, key=lambda x: x['pass_rate'], reverse=True)[:20]
                
                folder_names = [f['folder'] for f in top_folders]
                metrics = ['pass_rate', 'num_processed', 'pass_count', 'fail_count']
                metric_labels = ['Pass Rate (%)', 'Files Processed', 'Pass Count', 'Fail Count']
                
                # Normalize metrics for better visualization
                matrix = []
                for metric in metrics:
                    values = [f[metric] for f in top_folders]
                    if metric == 'pass_rate':
                        normalized = values  # Keep pass rate as is
                    else:
                        # Normalize to 0-100 scale
                        max_val = max(values) if max(values) > 0 else 1
                        normalized = [(v / max_val) * 100 for v in values]
                    matrix.append(normalized)
                
                fig = go.Figure(data=go.Heatmap(
                    z=matrix,
                    x=folder_names,
                    y=metric_labels,
                    colorscale='Viridis',
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title='Interactive Folder Performance Heatmap (Top 20 Folders)',
                    xaxis_title='Folders',
                    yaxis_title='Metrics',
                    width=1200,
                    height=600
                )
                
                interactive_heatmap_path = os.path.join(self.output_dir, 'interactive_folder_heatmap.html')
                fig.write_html(interactive_heatmap_path)
                print(f"Interactive folder heatmap saved to: {interactive_heatmap_path}")
                
        except ImportError:
            print("Plotly not available. Skipping interactive visualizations.")
            print("Install plotly with: pip install plotly")
    
    def create_summary_report(self):
        """Create a text summary report."""
        print("Creating summary report...")
        
        report_lines = []
        report_lines.append("EchoQuality Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 20)
        report_lines.append(f"Total Folders: {self.summary_data['total_folders']}")
        report_lines.append(f"Successful Folders: {self.summary_data['successful_folders']}")
        report_lines.append(f"Failed Folders: {self.summary_data['failed_folders']}")
        report_lines.append(f"Total Files: {self.summary_data['total_files']}")
        report_lines.append(f"Processed Files: {self.summary_data['total_processed']}")
        report_lines.append(f"Pass Count: {self.summary_data['total_pass']}")
        report_lines.append(f"Fail Count: {self.summary_data['total_fail']}")
        report_lines.append(f"Overall Pass Rate: {self.summary_data['overall_pass_rate']:.2f}%")
        report_lines.append("")
        
        # Score statistics
        scores = [result['score'] for result in self.all_results]
        report_lines.append("SCORE STATISTICS")
        report_lines.append("-" * 20)
        report_lines.append(f"Mean Score: {np.mean(scores):.4f}")
        report_lines.append(f"Median Score: {np.median(scores):.4f}")
        report_lines.append(f"Standard Deviation: {np.std(scores):.4f}")
        report_lines.append(f"Minimum Score: {np.min(scores):.4f}")
        report_lines.append(f"Maximum Score: {np.max(scores):.4f}")
        report_lines.append("")
        
        # Top performing folders
        folder_stats = []
        for folder in self.folder_data:
            if folder['num_processed'] > 0:
                folder_stats.append({
                    'folder': folder['folder'],
                    'pass_rate': folder['pass_rate'],
                    'num_processed': folder['num_processed']
                })
        
        folder_stats.sort(key=lambda x: x['pass_rate'], reverse=True)
        
        report_lines.append("TOP 10 PERFORMING FOLDERS")
        report_lines.append("-" * 30)
        for i, folder in enumerate(folder_stats[:10], 1):
            report_lines.append(f"{i:2d}. {folder['folder'][:50]:<50} {folder['pass_rate']:6.1f}% ({folder['num_processed']} files)")
        report_lines.append("")
        
        # Bottom performing folders
        folder_stats.sort(key=lambda x: x['pass_rate'])
        report_lines.append("BOTTOM 10 PERFORMING FOLDERS")
        report_lines.append("-" * 30)
        for i, folder in enumerate(folder_stats[:10], 1):
            report_lines.append(f"{i:2d}. {folder['folder'][:50]:<50} {folder['pass_rate']:6.1f}% ({folder['num_processed']} files)")
        report_lines.append("")
        
        # Assessment distribution
        assessments = [result['assessment'] for result in self.all_results]
        assessment_counts = {}
        for assessment in assessments:
            assessment_counts[assessment] = assessment_counts.get(assessment, 0) + 1
        
        report_lines.append("QUALITY ASSESSMENT DISTRIBUTION")
        report_lines.append("-" * 35)
        for assessment, count in sorted(assessment_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.all_results)) * 100
            report_lines.append(f"{assessment[:60]:<60} {count:5d} ({percentage:5.1f}%)")
        report_lines.append("")
        
        # Error analysis
        total_errors = 0
        error_summary = {}
        for folder in self.folder_data:
            if 'error_stats' in folder:
                for error_type, count in folder['error_stats']['error_counts'].items():
                    if count > 0:
                        error_summary[error_type] = error_summary.get(error_type, 0) + count
                        total_errors += count
        
        if total_errors > 0:
            report_lines.append("ERROR ANALYSIS")
            report_lines.append("-" * 15)
            report_lines.append(f"Total Errors: {total_errors}")
            for error_type, count in sorted(error_summary.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_errors) * 100
                report_lines.append(f"{error_type:<25} {count:5d} ({percentage:5.1f}%)")
            report_lines.append("")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to: {report_path}")
    
    def run_complete_visualization(self):
        """Run the complete visualization pipeline."""
        print("Starting comprehensive EchoQuality visualization...")
        print("=" * 60)
        
        try:
            # Load data
            self.load_data()
            
            # Create all visualizations
            self.create_overall_summary_dashboard()
            self.create_score_analysis_plots()
            self.create_folder_comparison_plots()
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
    parser = argparse.ArgumentParser(description='Visualize EchoQuality inference results')
    parser.add_argument('--results_dir', type=str, default='results/inference_output',
                        help='Directory containing inference results (default: results/inference_output)')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                        help='Directory to save visualization results (default: visualization_output)')
    
    args = parser.parse_args()
    
    # Create visualizer and run visualization
    visualizer = EchoQualityVisualizer(args.results_dir, args.output_dir)
    visualizer.run_complete_visualization()


if __name__ == "__main__":
    main()
