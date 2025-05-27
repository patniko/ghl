#!/usr/bin/env python3
"""
Script to convert DICOM ECG files to NPY format for use with HuBERT-ECG.

This script:
1. Reads DICOM files from the data/12L directory
2. Extracts ECG waveform data
3. Normalizes and resamples the data to match the expected format
4. Saves as .npy files compatible with HuBERT-ECG

Usage:
    python scripts/dicom_to_npy.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import numpy as np
import pydicom
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import interp1d


def extract_ecg_from_dicom(dicom_path):
    """
    Extract ECG waveform data from a DICOM file.
    
    Args:
        dicom_path (str): Path to the DICOM file
        
    Returns:
        tuple: (ecg_data, sampling_rate, num_leads) or (None, None, None) if extraction fails
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        
        # Print some basic info for debugging
        print(f"Processing: {os.path.basename(dicom_path)}")
        print(f"  Modality: {ds.get('Modality', 'N/A')}")
        print(f"  Manufacturer: {ds.get('Manufacturer', 'N/A')}")
        
        # Check if this is a waveform DICOM
        if hasattr(ds, 'WaveformSequence') and len(ds.WaveformSequence) > 0:
            return extract_from_waveform_sequence(ds)
        
        # Check if this is a multi-frame image with ECG data
        elif hasattr(ds, 'pixel_array'):
            return extract_from_pixel_array(ds)
        
        # Check for other ECG data storage methods
        else:
            print(f"  Warning: No recognized ECG data format found")
            return None, None, None
            
    except Exception as e:
        print(f"  Error reading DICOM file: {e}")
        return None, None, None


def extract_from_waveform_sequence(ds):
    """Extract ECG data from DICOM WaveformSequence."""
    try:
        waveform = ds.WaveformSequence[0]
        
        # Get sampling frequency
        sampling_rate = float(waveform.SamplingFrequency)
        print(f"  Sampling Rate: {sampling_rate} Hz")
        
        # Get number of channels and samples
        num_channels = int(waveform.NumberOfWaveformChannels)
        num_samples = int(waveform.NumberOfWaveformSamples)
        print(f"  Channels: {num_channels}, Samples: {num_samples}")
        
        # Get waveform data
        waveform_data = waveform.WaveformData
        
        # Convert to numpy array
        if hasattr(waveform_data, 'tobytes'):
            data_bytes = waveform_data.tobytes()
        else:
            data_bytes = bytes(waveform_data)
        
        # Determine data type (usually int16 for ECG)
        bits_allocated = getattr(waveform, 'WaveformBitsAllocated', 16)
        if bits_allocated == 16:
            dtype = np.int16
        elif bits_allocated == 8:
            dtype = np.int8
        else:
            dtype = np.int32
        
        # Convert bytes to numpy array
        raw_data = np.frombuffer(data_bytes, dtype=dtype)
        
        # Reshape to (channels, samples)
        if len(raw_data) == num_channels * num_samples:
            ecg_data = raw_data.reshape(num_channels, num_samples)
        else:
            # Try interleaved format
            ecg_data = raw_data[:num_channels * num_samples].reshape(num_samples, num_channels).T
        
        # Convert to float to avoid read-only issues and apply scaling if available
        ecg_data = ecg_data.astype(np.float64)
        
        if hasattr(waveform, 'ChannelDefinitionSequence'):
            for i, channel in enumerate(waveform.ChannelDefinitionSequence):
                if hasattr(channel, 'ChannelSensitivity') and hasattr(channel, 'ChannelSensitivityUnitsSequence'):
                    sensitivity = float(channel.ChannelSensitivity)
                    if i < ecg_data.shape[0]:
                        ecg_data[i] = ecg_data[i] * sensitivity
        
        print(f"  Extracted ECG shape: {ecg_data.shape}")
        return ecg_data, sampling_rate, num_channels
        
    except Exception as e:
        print(f"  Error extracting from waveform sequence: {e}")
        return None, None, None


def extract_from_pixel_array(ds):
    """Extract ECG data from DICOM pixel array (for image-based ECG storage)."""
    try:
        pixel_array = ds.pixel_array
        print(f"  Pixel array shape: {pixel_array.shape}")
        
        # This is a placeholder - the actual extraction would depend on how
        # the ECG data is encoded in the pixel array
        # You might need to implement specific logic based on your DICOM format
        
        # For now, return None to indicate this method needs implementation
        print(f"  Warning: Pixel array ECG extraction not implemented")
        return None, None, None
        
    except Exception as e:
        print(f"  Error extracting from pixel array: {e}")
        return None, None, None


def normalize_ecg_data(ecg_data, target_length=5000, target_leads=12):
    """
    Normalize ECG data to the expected format for HuBERT-ECG.
    
    Args:
        ecg_data (np.ndarray): ECG data with shape (leads, samples)
        target_length (int): Target number of samples (default: 5000)
        target_leads (int): Target number of leads (default: 12)
        
    Returns:
        np.ndarray: Normalized ECG data with shape (target_leads, target_length)
    """
    if ecg_data is None:
        return None
    
    leads, samples = ecg_data.shape
    print(f"  Normalizing from ({leads}, {samples}) to ({target_leads}, {target_length})")
    
    # Handle different number of leads
    if leads < target_leads:
        # Pad with zeros or duplicate leads
        padded_data = np.zeros((target_leads, samples))
        padded_data[:leads] = ecg_data
        # Duplicate some leads if we have fewer than target
        for i in range(leads, target_leads):
            padded_data[i] = ecg_data[i % leads]
        ecg_data = padded_data
    elif leads > target_leads:
        # Take the first target_leads
        ecg_data = ecg_data[:target_leads]
    
    # Handle different number of samples
    if samples != target_length:
        # Resample each lead to target length
        resampled_data = np.zeros((target_leads, target_length))
        x_old = np.linspace(0, 1, samples)
        x_new = np.linspace(0, 1, target_length)
        
        for i in range(target_leads):
            f = interp1d(x_old, ecg_data[i], kind='linear', bounds_error=False, fill_value='extrapolate')
            resampled_data[i] = f(x_new)
        
        ecg_data = resampled_data
    
    # Normalize to similar range as PTB-XL data (roughly -1 to 1)
    for i in range(target_leads):
        lead_data = ecg_data[i]
        # Remove DC offset
        lead_data = lead_data - np.mean(lead_data)
        # Normalize by standard deviation, but cap the scaling
        std_dev = np.std(lead_data)
        if std_dev > 0:
            # Scale to have similar amplitude as PTB-XL data
            ecg_data[i] = lead_data / (std_dev * 10)  # Adjust scaling factor as needed
    
    return ecg_data.astype(np.float64)


def process_dicom_files(input_dir, output_dir):
    """
    Process all DICOM files in the input directory and save as NPY files.
    
    Args:
        input_dir (str): Directory containing DICOM files
        output_dir (str): Directory to save NPY files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all DICOM files
    dicom_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.dcm', '.dicom')):
            dicom_files.append(os.path.join(input_dir, file))
    
    if not dicom_files:
        print(f"No DICOM files found in {input_dir}")
        return
    
    print(f"Found {len(dicom_files)} DICOM files")
    
    successful_conversions = 0
    
    for dicom_path in tqdm(dicom_files, desc="Processing DICOM files"):
        # Extract ECG data
        ecg_data, sampling_rate, num_leads = extract_ecg_from_dicom(dicom_path)
        
        if ecg_data is not None:
            # Normalize the data
            normalized_data = normalize_ecg_data(ecg_data)
            
            if normalized_data is not None:
                # Create output filename
                base_name = os.path.splitext(os.path.basename(dicom_path))[0]
                output_filename = f"{base_name}.npy"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save as NPY file
                np.save(output_path, normalized_data)
                print(f"  Saved: {output_filename} with shape {normalized_data.shape}")
                successful_conversions += 1
            else:
                print(f"  Failed to normalize data for {os.path.basename(dicom_path)}")
        else:
            print(f"  Failed to extract ECG data from {os.path.basename(dicom_path)}")
        
        print()  # Add blank line for readability
    
    print(f"Successfully converted {successful_conversions} out of {len(dicom_files)} DICOM files")


def main():
    parser = argparse.ArgumentParser(description='Convert DICOM ECG files to NPY format for HuBERT-ECG.')
    parser.add_argument('--input_dir', type=str, default='data/12L',
                        help='Directory containing DICOM files (default: data/12L)')
    parser.add_argument('--output_dir', type=str, default='data/12L/processed',
                        help='Directory to save NPY files (default: data/12L/processed)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    print(f"Converting DICOM files from {args.input_dir} to NPY format in {args.output_dir}")
    process_dicom_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
