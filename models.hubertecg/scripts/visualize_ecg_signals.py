#!/usr/bin/env python3
"""
Script to visualize ECG signals from DICOM files and NPY files.

This script creates comprehensive visualizations of ECG signals including:
- Individual lead plots
- Multi-lead overview
- Signal statistics
- Comparison between original DICOM and converted NPY

Usage:
    python scripts/visualize_ecg_signals.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm
from datetime import datetime

# Set style for better plots
plt.style.use('default')


def extract_ecg_from_dicom(dicom_path):
    """
    Extract ECG waveform data from DICOM file.
    
    Args:
        dicom_path (str): Path to DICOM file
        
    Returns:
        tuple: (ecg_data, sampling_rate, channel_labels) or (None, None, None) if failed
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        
        if not hasattr(ds, 'WaveformSequence') or not ds.WaveformSequence:
            print(f"No waveform sequence found in {os.path.basename(dicom_path)}")
            return None, None, None
        
        waveform = ds.WaveformSequence[0]
        
        # Get basic parameters
        sampling_rate = float(waveform.SamplingFrequency)
        num_channels = int(waveform.NumberOfWaveformChannels)
        num_samples = int(waveform.NumberOfWaveformSamples)
        
        # Extract waveform data
        waveform_data = waveform.WaveformData
        
        # Convert to numpy array
        if hasattr(waveform, 'WaveformBitsAllocated'):
            bits_allocated = int(waveform.WaveformBitsAllocated)
            if bits_allocated == 16:
                dtype = np.int16
            elif bits_allocated == 8:
                dtype = np.int8
            else:
                dtype = np.int16
        else:
            dtype = np.int16
        
        # Reshape the data
        ecg_array = np.frombuffer(waveform_data, dtype=dtype)
        ecg_array = ecg_array.reshape(num_samples, num_channels).T
        
        # Apply channel sensitivity if available
        if hasattr(waveform, 'ChannelDefinitionSequence'):
            for i, channel in enumerate(waveform.ChannelDefinitionSequence):
                if hasattr(channel, 'ChannelSensitivity') and i < len(ecg_array):
                    sensitivity = float(channel.ChannelSensitivity)
                    if sensitivity != 0:
                        ecg_array[i] = ecg_array[i] * sensitivity
        
        # Get channel labels
        channel_labels = []
        if hasattr(waveform, 'ChannelDefinitionSequence'):
            for channel in waveform.ChannelDefinitionSequence:
                if hasattr(channel, 'ChannelLabel'):
                    channel_labels.append(str(channel.ChannelLabel))
                else:
                    channel_labels.append(f"Lead_{len(channel_labels)+1}")
        else:
            channel_labels = [f"Lead_{i+1}" for i in range(num_channels)]
        
        # Convert to float64 to avoid read-only issues
        ecg_array = ecg_array.astype(np.float64)
        
        return ecg_array, sampling_rate, channel_labels
        
    except Exception as e:
        print(f"Error extracting ECG from {os.path.basename(dicom_path)}: {e}")
        return None, None, None


def create_ecg_visualization(ecg_data, sampling_rate, channel_labels, title, output_path):
    """
    Create comprehensive ECG visualization.
    
    Args:
        ecg_data (np.array): ECG data with shape (n_leads, n_samples)
        sampling_rate (float): Sampling rate in Hz
        channel_labels (list): List of channel/lead labels
        title (str): Title for the plot
        output_path (str): Path to save the plot
    """
    n_leads, n_samples = ecg_data.shape
    time_axis = np.arange(n_samples) / sampling_rate
    
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
        ax1.plot(time_axis, ecg_data[i] + offset, 
                label=channel_labels[i] if i < len(channel_labels) else f'Lead {i+1}', 
                linewidth=0.8, color=colors[i])
        ax1.text(-0.02 * time_axis[-1], offset, 
                channel_labels[i] if i < len(channel_labels) else f'Lead {i+1}', 
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
        
        ax.plot(time_axis, ecg_data[i], 'b-', linewidth=1)
        ax.set_title(f'{channel_labels[i] if i < len(channel_labels) else f"Lead {i+1}"}', fontweight='bold')
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


def create_signal_comparison(dicom_data, npy_data, sampling_rate, channel_labels, title, output_path):
    """
    Create comparison visualization between DICOM and NPY data.
    
    Args:
        dicom_data (np.array): Original DICOM ECG data
        npy_data (np.array): Converted NPY ECG data
        sampling_rate (float): Sampling rate
        channel_labels (list): Channel labels
        title (str): Plot title
        output_path (str): Output path
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Create time axes
    time_dicom = np.arange(dicom_data.shape[1]) / sampling_rate
    time_npy = np.arange(npy_data.shape[1]) / 500  # NPY is normalized to 500Hz equivalent
    
    for i in range(min(12, dicom_data.shape[0], npy_data.shape[0])):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Plot both signals
        ax.plot(time_dicom, dicom_data[i], 'b-', linewidth=1, label='Original DICOM', alpha=0.7)
        ax.plot(time_npy, npy_data[i], 'r-', linewidth=1, label='Converted NPY', alpha=0.7)
        
        ax.set_title(f'{channel_labels[i] if i < len(channel_labels) else f"Lead {i+1}"}', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add correlation coefficient
        # Resample to same length for correlation
        min_len = min(len(dicom_data[i]), len(npy_data[i]))
        dicom_resampled = dicom_data[i][:min_len]
        npy_resampled = npy_data[i][:min_len]
        
        correlation = np.corrcoef(dicom_resampled, npy_resampled)[0, 1]
        ax.text(0.02, 0.98, f'Corr: {correlation:.3f}', transform=ax.transAxes, 
                fontsize=8, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison to: {output_path}")


def create_signal_statistics_plot(ecg_data, channel_labels, title, output_path):
    """
    Create statistical analysis plots for ECG data.
    
    Args:
        ecg_data (np.array): ECG data
        channel_labels (list): Channel labels
        title (str): Plot title
        output_path (str): Output path
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title} - Statistical Analysis', fontsize=16, fontweight='bold')
    
    # Calculate statistics
    means = np.mean(ecg_data, axis=1)
    stds = np.std(ecg_data, axis=1)
    mins = np.min(ecg_data, axis=1)
    maxs = np.max(ecg_data, axis=1)
    
    lead_names = [channel_labels[i] if i < len(channel_labels) else f'Lead {i+1}' for i in range(len(means))]
    
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


def process_files(input_dir, output_dir):
    """
    Process all DICOM files and create visualizations.
    
    Args:
        input_dir (str): Directory containing DICOM files
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find DICOM files
    dicom_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.dcm', '.dicom')):
            dicom_files.append(os.path.join(input_dir, file))
    
    if not dicom_files:
        print(f"No DICOM files found in {input_dir}")
        return
    
    print(f"Found {len(dicom_files)} DICOM files")
    
    # Check for processed NPY files
    npy_dir = os.path.join(input_dir, 'processed')
    npy_files_exist = os.path.exists(npy_dir)
    
    for dicom_path in tqdm(dicom_files, desc="Creating visualizations"):
        base_name = os.path.splitext(os.path.basename(dicom_path))[0]
        print(f"\nProcessing: {base_name}")
        
        # Extract ECG from DICOM
        ecg_data, sampling_rate, channel_labels = extract_ecg_from_dicom(dicom_path)
        
        if ecg_data is None:
            print(f"  Skipping {base_name} - could not extract ECG data")
            continue
        
        print(f"  Extracted ECG: {ecg_data.shape} at {sampling_rate} Hz")
        
        # Create DICOM visualization
        dicom_viz_path = os.path.join(output_dir, f"{base_name}_dicom_visualization.png")
        create_ecg_visualization(ecg_data, sampling_rate, channel_labels, 
                                f"DICOM ECG Visualization - {base_name}", dicom_viz_path)
        
        # Create statistics plot
        stats_path = os.path.join(output_dir, f"{base_name}_statistics.png")
        create_signal_statistics_plot(ecg_data, channel_labels, 
                                    f"DICOM ECG - {base_name}", stats_path)
        
        # If NPY file exists, create comparison
        if npy_files_exist:
            npy_path = os.path.join(npy_dir, f"{base_name}.npy")
            if os.path.exists(npy_path):
                try:
                    npy_data = np.load(npy_path)
                    print(f"  Loaded NPY data: {npy_data.shape}")
                    
                    # Create NPY visualization
                    npy_viz_path = os.path.join(output_dir, f"{base_name}_npy_visualization.png")
                    create_ecg_visualization(npy_data, 500, channel_labels[:len(npy_data)], 
                                           f"NPY ECG Visualization - {base_name}", npy_viz_path)
                    
                    # Create comparison
                    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
                    create_signal_comparison(ecg_data, npy_data, sampling_rate, channel_labels,
                                           f"DICOM vs NPY Comparison - {base_name}", comparison_path)
                    
                except Exception as e:
                    print(f"  Error loading NPY file: {e}")
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create comprehensive ECG signal visualizations.')
    parser.add_argument('--input_dir', type=str, default='raw_data/12L',
                        help='Directory containing DICOM files (default: raw_data/12L)')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data/12L/visualizations',
                        help='Directory to save visualizations (default: preprocessed_data/12L/visualizations)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    print(f"Creating ECG visualizations from files in {args.input_dir}")
    print(f"Visualizations will be saved to {args.output_dir}")
    
    process_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
