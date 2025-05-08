#!/usr/bin/env python3
"""
Script to download and process the PTB-XL dataset for use with HuBERT-ECG.

This script:
1. Downloads the PTB-XL dataset from PhysioNet
2. Extracts the dataset
3. Processes the ECG data to create .npy files that match the filenames in the CSV files

Usage:
    python setup_ptbxl.py [--data_dir DATA_DIR]
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

def download_file(url, destination):
    """Download a file from a URL to a destination."""
    print(f"Downloading {url} to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress bar
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(f"\rDownloaded: {downloaded} / {total_size} bytes ({percent:.2f}%)")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, reporthook=report_progress)
    print("\nDownload complete!")

def extract_zip(zip_path, extract_to):
    """Extract a zip file to a directory."""
    print(f"Extracting {zip_path} to {extract_to}")
    
    # Create directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print("Extraction complete!")

def process_ecg_data(data_dir, output_dir):
    """Process the ECG data to create .npy files."""
    print("Processing ECG data...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the PTB-XL database
    ptbxl_database = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    
    # Process each ECG record
    for idx, row in tqdm(ptbxl_database.iterrows(), total=len(ptbxl_database), desc="Processing ECG records"):
        # Get the path to the ECG record
        record_path = os.path.join(data_dir, row['filename_hr'])
        
        try:
            # Load the ECG record
            record = wfdb.rdrecord(record_path)
            
            # Get the ECG signal
            signal = record.p_signal.T  # Transpose to get (leads, samples)
            
            # Create the output filename
            output_filename = f"HR{os.path.basename(row['filename_hr']).replace('_hr', '.hea.npy')}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the ECG signal as a .npy file
            np.save(output_path, signal)
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
    
    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Download and process the PTB-XL dataset for use with HuBERT-ECG.')
    parser.add_argument('--data_dir', type=str, default='data/ptbxl',
                        help='Directory to store the PTB-XL dataset (default: data/ptbxl)')
    
    args = parser.parse_args()
    
    # URLs and paths
    ptbxl_url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip"
    zip_path = os.path.join(args.data_dir, "ptbxl.zip")
    extract_path = args.data_dir
    output_dir = os.path.join(args.data_dir, "processed")
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(zip_path):
        download_file(ptbxl_url, zip_path)
    else:
        print(f"Dataset already downloaded to {zip_path}")
    
    # Extract the dataset if it doesn't exist
    database_csv_path = os.path.join(extract_path, "ptbxl_database.csv")
    if not os.path.exists(database_csv_path):
        extract_zip(zip_path, extract_path)
        
        # Check if extraction created a subdirectory (like ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2)
        # and find the actual path to the database file
        if not os.path.exists(database_csv_path):
            # Look for subdirectories that might contain the database file
            for item in os.listdir(extract_path):
                potential_dir = os.path.join(extract_path, item)
                if os.path.isdir(potential_dir) and "ptb-xl" in item.lower():
                    potential_csv = os.path.join(potential_dir, "ptbxl_database.csv")
                    if os.path.exists(potential_csv):
                        # Found the database in a subdirectory, update the extract_path
                        extract_path = potential_dir
                        print(f"Found PTB-XL database in subdirectory: {extract_path}")
                        break
    else:
        print(f"Dataset already extracted to {extract_path}")
    
    # Process the ECG data
    process_ecg_data(extract_path, output_dir)
    
    print(f"PTB-XL dataset setup complete! Processed data is available in {output_dir}")
    print("You can now use the HuBERT-ECG-Demo.ipynb notebook to explore the model.")

if __name__ == "__main__":
    main()
