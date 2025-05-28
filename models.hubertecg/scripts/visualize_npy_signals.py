#!/usr/bin/env python3
"""
Script to visualize ECG signals from NPY files.

This script creates comprehensive visualizations of ECG signals from converted NPY files.

Usage:
    python scripts/visualize_npy_signals.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set style for better plots
plt.style.use('default')


def create_ecg_visualization(ecg_data, title, output_path):
    """
    Create comprehensive ECG visualization.
    
    Args:
        ecg_data (np.array): ECG data with shape (n_leads, n_samples)
        title (str): Title for the plot
        output_path (str): Path to save the plot
    """
    n_leads, n_samples = ecg_data.shape
    sampling_rate = 500  # NPY files are normalized to 500Hz equivalent
    time_axis = np.arange(n_samples) / sampling_rate
    
    # Standard 12-lead ECG labels
    lead_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Create grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # Plot 1: All leads overview (stacked)
    ax1 = fig.add_subplot(gs[0, :])
    lead_spacing = np.max(np.abs(ecg_data)) * 2
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_leads, 12)))
    
    for i in range(min(n_leads, 12)):  # Show up to 12 leads
        offset = i * lead_spacing
        label = lead_labels[i] if i < len(lead_labels) else f'Lead {i+1}'
        ax1.plot(time_axis, ecg_data[i] + offset, 
                label=label, linewidth=0.8, color=colors[i])
        ax1.text(-0.02 * time_axis[-1], offset, label, 
                ha='right', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude (offset for clarity)')
    ax1.set_title('All ECG Leads Overview', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_axis[-1])
    
    # Plot 2: Individual leads (first 12 leads in a 3x4 grid)
    for i in range(min(12, n_leads)):
        row = (i // 4) + 1
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        
        label = lead_labels[i] if i < len(lead_labels) else f'Lead {i+1}'
        ax.plot(time_axis, ecg_data[i], 'b-', linewidth=1)
        ax.set_title(f'{label}', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(ecg_data[i])
        std_val = np.std(ecg_data[i])
        min_val = np.min(ecg_data[i])
        max_val = np.max(ecg_data[i])
        
        stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nRange=[{min_val:.3f}, {max_val:.3f}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add metadata text
    metadata_text = f"""
    Sampling Rate: {sampling_rate:.1f} Hz
    Duration: {n_samples/sampling_rate:.2f} seconds
    Number of Leads: {n_leads}
    Samples per Lead: {n_samples}
    Data Range: [{np.min(ecg_data):.3f}, {np.max(ecg_data):.3f}]
    """
    
    fig.text(0.02, 0.02, metadata_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to: {output_path}")


def create_signal_statistics_plot(ecg_data, title, output_path):
    """
    Create statistical analysis plots for ECG data.
    
    Args:
        ecg_data (np.array): ECG data
        title (str): Plot title
        output_path (str): Output path
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title} - Statistical Analysis', fontsize=16, fontweight='bold')
    
    # Standard 12-lead ECG labels
    lead_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Calculate statistics
    means = np.mean(ecg_data, axis=1)
    stds = np.std(ecg_data, axis=1)
    mins = np.min(ecg_data, axis=1)
    maxs = np.max(ecg_data, axis=1)
    
    lead_names = [lead_labels[i] if i < len(lead_labels) else f'Lead {i+1}' for i in range(len(means))]
    
    # Plot 1: Mean values
    axes[0, 0].bar(range(len(means)), means, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Mean Amplitude by Lead')
    axes[0, 0].set_xlabel('Lead')
    axes[0, 0].set_ylabel('Mean Amplitude')
    axes[0, 0].set_xticks(range(len(means)))
    axes[0, 0].set_xticklabels(lead_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation
    axes[0, 1].bar(range(len(stds)), stds, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Standard Deviation by Lead')
    axes[0, 1].set_xlabel('Lead')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_xticks(range(len(stds)))
    axes[0, 1].set_xticklabels(lead_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Range (min-max)
    ranges = maxs - mins
    axes[1, 0].bar(range(len(ranges)), ranges, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Signal Range by Lead')
    axes[1, 0].set_xlabel('Lead')
    axes[1, 0].set_ylabel('Range (Max - Min)')
    axes[1, 0].set_xticks(range(len(ranges)))
    axes[1, 0].set_xticklabels(lead_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distribution of all values
    all_values = ecg_data.flatten()
    axes[1, 1].hist(all_values, bins=50, color='plum', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of All ECG Values')
    axes[1, 1].set_xlabel('Amplitude')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
    Overall Statistics:
    Mean: {np.mean(all_values):.4f}
    Std: {np.std(all_values):.4f}
    Min: {np.min(all_values):.4f}
    Max: {np.max(all_values):.4f}
    """
    axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved statistics plot to: {output_path}")


def process_npy_files(input_dir, output_dir):
    """
    Process all NPY files and create visualizations.
    
    Args:
        input_dir (str): Directory containing NPY files
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find NPY files
    npy_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.npy'):
            npy_files.append(os.path.join(input_dir, file))
    
    if not npy_files:
        print(f"No NPY files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} NPY files")
    
    for npy_path in tqdm(npy_files, desc="Creating visualizations"):
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        print(f"\nProcessing: {base_name}")
        
        try:
            # Load ECG data
            ecg_data = np.load(npy_path)
            print(f"  Loaded ECG: {ecg_data.shape}")
            
            # Create visualization
            viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
            create_ecg_visualization(ecg_data, f"ECG Visualization - {base_name}", viz_path)
            
            # Create statistics plot
            stats_path = os.path.join(output_dir, f"{base_name}_statistics.png")
            create_signal_statistics_plot(ecg_data, f"ECG - {base_name}", stats_path)
            
        except Exception as e:
            print(f"  Error processing {base_name}: {e}")
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create comprehensive ECG signal visualizations from NPY files.')
    parser.add_argument('--input_dir', type=str, default='preprocessed_data/12L/processed',
                        help='Directory containing NPY files (default: preprocessed_data/12L/processed)')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data/12L/visualizations',
                        help='Directory to save visualizations (default: preprocessed_data/12L/visualizations)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        print("Please run 'make dicom-setup' first to convert DICOM files to NPY format")
        sys.exit(1)
    
    print(f"Creating ECG visualizations from NPY files in {args.input_dir}")
    print(f"Visualizations will be saved to {args.output_dir}")
    
    process_npy_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
