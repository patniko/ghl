#!/usr/bin/env python3
"""
Script to convert AliveCor JSON ECG files to NPY format for use with HuBERT-ECG.

This script:
1. Reads JSON files from the raw_data/PATIENT-ALIVECOR directory
2. Extracts ECG waveform data from 6 leads
3. Normalizes and resamples the data to match the expected format
4. Saves as .npy files compatible with HuBERT-ECG

Usage:
    python preprocessors/alivecor_to_npy.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--data_type DATA_TYPE]
"""

import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d


def extract_ecg_from_alivecor_json(json_path, data_type='enhanced'):
    """
    Extract ECG waveform data from an AliveCor JSON file.
    
    Args:
        json_path (str): Path to the JSON file
        data_type (str): Either 'raw' or 'enhanced' to specify which data to use
        
    Returns:
        tuple: (ecg_data, sampling_rate, num_leads, duration) or (None, None, None, None) if extraction fails
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"Processing: {os.path.basename(json_path)}")
        
        # Extract metadata
        duration = data.get('duration', 0)  # in milliseconds
        patient_id = data.get('patientID', 'unknown')
        recorded_at = data.get('recordedAt', 'unknown')
        
        print(f"  Patient ID: {patient_id}")
        print(f"  Duration: {duration} ms")
        print(f"  Recorded: {recorded_at}")
        
        # Get the specified data type (raw or enhanced)
        if 'data' not in data or data_type not in data['data']:
            print(f"  Error: {data_type} data not found in JSON")
            return None, None, None, None
        
        ecg_data_section = data['data'][data_type]
        
        # Extract sampling parameters
        sampling_rate = ecg_data_section.get('frequency', 300)
        amplitude_resolution = ecg_data_section.get('amplitudeResolution', 500)
        num_leads = ecg_data_section.get('numLeads', 6)
        
        print(f"  Sampling Rate: {sampling_rate} Hz")
        print(f"  Amplitude Resolution: {amplitude_resolution}")
        print(f"  Number of Leads: {num_leads}")
        print(f"  Using: {data_type} data")
        
        # Extract lead data
        samples = ecg_data_section.get('samples', {})
        lead_names = ['leadI', 'leadII', 'leadIII', 'AVR', 'AVL', 'AVF']
        
        # Check if all expected leads are present
        missing_leads = [lead for lead in lead_names if lead not in samples]
        if missing_leads:
            print(f"  Warning: Missing leads: {missing_leads}")
        
        # Extract data for available leads
        ecg_leads = []
        actual_lead_names = []
        
        for lead_name in lead_names:
            if lead_name in samples and samples[lead_name]:
                lead_data = np.array(samples[lead_name], dtype=np.float64)
                # Apply amplitude resolution scaling
                lead_data = lead_data / amplitude_resolution
                ecg_leads.append(lead_data)
                actual_lead_names.append(lead_name)
                print(f"    {lead_name}: {len(lead_data)} samples")
        
        if not ecg_leads:
            print(f"  Error: No valid lead data found")
            return None, None, None, None
        
        # Convert to numpy array with shape (leads, samples)
        ecg_data = np.array(ecg_leads)
        
        print(f"  Extracted ECG shape: {ecg_data.shape}")
        print(f"  Lead names: {actual_lead_names}")
        
        return ecg_data, sampling_rate, len(actual_lead_names), duration
        
    except json.JSONDecodeError as e:
        print(f"  Error: Invalid JSON format: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"  Error reading JSON file: {e}")
        return None, None, None, None


def normalize_alivecor_data(ecg_data, sampling_rate, duration, target_length=5000):
    """
    Normalize AliveCor ECG data to the expected format for HuBERT-ECG.
    
    Args:
        ecg_data (np.ndarray): ECG data with shape (leads, samples)
        sampling_rate (float): Original sampling rate in Hz
        duration (int): Duration in milliseconds
        target_length (int): Target number of samples (default: 5000)
        
    Returns:
        np.ndarray: Normalized ECG data with shape (leads, target_length)
    """
    if ecg_data is None:
        return None
    
    leads, samples = ecg_data.shape
    print(f"  Normalizing from ({leads}, {samples}) to ({leads}, {target_length})")
    
    # Calculate expected samples based on duration and sampling rate
    expected_samples = int((duration / 1000.0) * sampling_rate)
    print(f"  Expected samples from duration: {expected_samples}")
    
    # Handle different number of samples by resampling
    if samples != target_length:
        # Resample each lead to target length
        resampled_data = np.zeros((leads, target_length))
        x_old = np.linspace(0, 1, samples)
        x_new = np.linspace(0, 1, target_length)
        
        for i in range(leads):
            f = interp1d(x_old, ecg_data[i], kind='linear', bounds_error=False, fill_value='extrapolate')
            resampled_data[i] = f(x_new)
        
        ecg_data = resampled_data
    
    # Normalize each lead
    for i in range(leads):
        lead_data = ecg_data[i]
        
        # Remove DC offset
        lead_data = lead_data - np.mean(lead_data)
        
        # Normalize by standard deviation, but cap the scaling
        std_dev = np.std(lead_data)
        if std_dev > 0:
            # Scale to have similar amplitude as PTB-XL data
            # AliveCor data is already scaled by amplitude resolution, so use gentler normalization
            ecg_data[i] = lead_data / (std_dev * 5)  # Adjust scaling factor as needed
        else:
            ecg_data[i] = lead_data
    
    return ecg_data.astype(np.float64)


def process_alivecor_folder(patient_dir, output_dir, data_type='enhanced', target_length=5000):
    """
    Process all JSON files in the AliveCor patient directory.
    
    Args:
        patient_dir (str): Directory containing JSON files for AliveCor patient
        output_dir (str): Directory to save NPY files for this patient
        data_type (str): Either 'raw' or 'enhanced' to specify which data to use
        target_length (int): Target number of samples for output
    """
    patient_name = os.path.basename(patient_dir)
    print(f"\nProcessing AliveCor patient: {patient_name}")
    
    # Create patient-specific output directory
    patient_output_dir = os.path.join(output_dir, patient_name)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # Find all JSON files in this patient directory
    json_files = []
    for file in os.listdir(patient_dir):
        if file.lower().endswith('.json'):
            json_files.append(os.path.join(patient_dir, file))
    
    if not json_files:
        print(f"  No JSON files found in {patient_dir}")
        return 0
    
    print(f"  Found {len(json_files)} JSON files")
    
    successful_conversions = 0
    
    for json_path in json_files:
        # Extract ECG data
        ecg_data, sampling_rate, num_leads, duration = extract_ecg_from_alivecor_json(json_path, data_type)
        
        if ecg_data is not None:
            # Normalize the data
            normalized_data = normalize_alivecor_data(ecg_data, sampling_rate, duration, target_length)
            
            if normalized_data is not None:
                # Create output filename
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                output_filename = f"{base_name}.npy"
                output_path = os.path.join(patient_output_dir, output_filename)
                
                # Save as NPY file
                np.save(output_path, normalized_data)
                print(f"    Saved: {output_filename} with shape {normalized_data.shape}")
                successful_conversions += 1
            else:
                print(f"    Failed to normalize data for {os.path.basename(json_path)}")
        else:
            print(f"    Failed to extract ECG data from {os.path.basename(json_path)}")
    
    print(f"  Patient {patient_name}: {successful_conversions}/{len(json_files)} files converted successfully")
    return successful_conversions


def process_alivecor_patients(data_dir, output_dir, data_type='enhanced', target_length=5000):
    """
    Process AliveCor patient folders in the data directory.
    
    Args:
        data_dir (str): Directory containing patient folders
        output_dir (str): Directory to save processed NPY files
        data_type (str): Either 'raw' or 'enhanced' to specify which data to use
        target_length (int): Target number of samples for output
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all directories that start with 'PATIENT-ALIVECOR'
    alivecor_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.startswith('PATIENT-ALIVECOR'):
            alivecor_dirs.append(item_path)
    
    if not alivecor_dirs:
        print(f"No AliveCor directories found in: {data_dir}")
        print(f"Looking for directories starting with 'PATIENT-ALIVECOR'")
        return
    
    # Sort directories for consistent processing order
    alivecor_dirs.sort()
    
    print(f"Found {len(alivecor_dirs)} AliveCor directories:")
    for dir_path in alivecor_dirs:
        print(f"  - {os.path.basename(dir_path)}")
    
    total_successful = 0
    total_files = 0
    
    # Process each AliveCor patient folder
    for alivecor_dir in alivecor_dirs:
        successful = process_alivecor_folder(alivecor_dir, output_dir, data_type, target_length)
        
        # Count files in this directory
        dir_files = len([f for f in os.listdir(alivecor_dir) if f.lower().endswith('.json')])
        
        total_successful += successful
        total_files += dir_files
    
    print(f"\n=== ALIVECOR CONVERSION SUMMARY ===")
    print(f"Data type used: {data_type}")
    print(f"Target sample length: {target_length}")
    print(f"Directories processed: {len(alivecor_dirs)}")
    print(f"Total JSON files: {total_files}")
    print(f"Successfully converted: {total_successful}")
    print(f"Success rate: {total_successful/total_files*100:.1f}%" if total_files > 0 else "No files to process")


def main():
    parser = argparse.ArgumentParser(description='Convert AliveCor JSON ECG files to NPY format for HuBERT-ECG.')
    parser.add_argument('--input_dir', type=str, default='raw_data',
                        help='Directory containing PATIENT-ALIVECOR folder with JSON files (default: raw_data)')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data',
                        help='Directory to save NPY files (default: preprocessed_data)')
    parser.add_argument('--data_type', type=str, choices=['raw', 'enhanced'], default='enhanced',
                        help='Type of ECG data to extract: raw or enhanced (default: enhanced)')
    parser.add_argument('--target_length', type=int, default=5000,
                        help='Target number of samples for output (default: 5000)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    print(f"Converting AliveCor JSON files from {args.input_dir} to NPY format in {args.output_dir}")
    print(f"Using {args.data_type} data with target length {args.target_length}")
    
    process_alivecor_patients(args.input_dir, args.output_dir, args.data_type, args.target_length)


if __name__ == "__main__":
    main()
