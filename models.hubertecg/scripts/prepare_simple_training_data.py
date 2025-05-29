#!/usr/bin/env python3
"""
Simple training data preparation script for HuBERT-ECG.
This script processes ECG files dropped into training_data/raw/ and prepares them for training.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import json
from typing import List, Dict, Any
import glob

# Add the project root to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_directory_structure():
    """Create the necessary directory structure for training data."""
    base_dir = Path("training_data")
    directories = [
        base_dir / "raw",
        base_dir / "processed", 
        base_dir / "metadata"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return base_dir

def find_ecg_files(raw_dir: Path) -> List[Path]:
    """Find all ECG files in the raw directory."""
    supported_extensions = ['.npy', '.csv', '.json', '.txt']
    ecg_files = []
    
    for ext in supported_extensions:
        pattern = str(raw_dir / f"**/*{ext}")
        files = glob.glob(pattern, recursive=True)
        ecg_files.extend([Path(f) for f in files])
    
    logger.info(f"Found {len(ecg_files)} ECG files")
    return ecg_files

def process_npy_file(file_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Process a .npy ECG file."""
    try:
        # Load the ECG data
        ecg_data = np.load(file_path)
        
        # Validate ECG data shape
        if len(ecg_data.shape) != 2:
            logger.warning(f"Unexpected ECG shape {ecg_data.shape} for {file_path}. Expected 2D array.")
            return None
            
        # Ensure we have 12 leads
        if ecg_data.shape[0] != 12:
            logger.warning(f"Expected 12 leads, got {ecg_data.shape[0]} for {file_path}")
            # Try to transpose if it looks like leads are in the second dimension
            if ecg_data.shape[1] == 12:
                ecg_data = ecg_data.T
                logger.info(f"Transposed ECG data for {file_path}")
            else:
                logger.error(f"Cannot process {file_path}: wrong number of leads")
                return None
        
        # Ensure minimum length (5 seconds at 500Hz = 2500 samples)
        min_samples = 2500
        if ecg_data.shape[1] < min_samples:
            logger.warning(f"ECG too short ({ecg_data.shape[1]} samples) for {file_path}. Padding to {min_samples}")
            # Pad with zeros
            padding = min_samples - ecg_data.shape[1]
            ecg_data = np.pad(ecg_data, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        
        # Save processed file
        output_filename = file_path.stem + ".npy"
        output_path = output_dir / output_filename
        np.save(output_path, ecg_data)
        
        return {
            "filename": output_filename,
            "original_path": str(file_path),
            "shape": ecg_data.shape,
            "duration_seconds": ecg_data.shape[1] / 500,  # Assuming 500Hz sampling rate
            "status": "processed"
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def create_default_labels(num_files: int) -> pd.DataFrame:
    """Create default labels for training. Users can modify this CSV later."""
    # Create a simple binary classification task (normal vs abnormal)
    # In a real scenario, users would provide their own labels
    
    labels_data = []
    for i in range(num_files):
        # Default: assign random labels for demonstration
        # Users should replace this with their actual labels
        labels_data.append({
            "normal": np.random.choice([0, 1]),
            "abnormal": np.random.choice([0, 1]),
            "arrhythmia": np.random.choice([0, 1])
        })
    
    return pd.DataFrame(labels_data)

def create_training_csv(processed_files: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Create the training CSV file required by the ECGDataset."""
    
    # Filter out failed processing
    valid_files = [f for f in processed_files if f is not None]
    
    if not valid_files:
        raise ValueError("No valid ECG files were processed")
    
    # Create DataFrame with filenames
    df_data = []
    for file_info in valid_files:
        df_data.append({
            "filename": file_info["filename"],
            "original_path": file_info["original_path"],
            "shape": str(file_info["shape"]),
            "duration_seconds": file_info["duration_seconds"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Add default labels (users should modify these)
    default_labels = create_default_labels(len(df))
    
    # Combine filename info with labels
    final_df = pd.concat([df, default_labels], axis=1)
    
    # Save training CSV
    csv_path = output_dir / "training_dataset.csv"
    final_df.to_csv(csv_path, index=False)
    
    logger.info(f"Created training CSV with {len(final_df)} samples: {csv_path}")
    
    # Create a validation split (20% of data)
    val_size = max(1, len(final_df) // 5)
    val_df = final_df.sample(n=val_size, random_state=42)
    train_df = final_df.drop(val_df.index)
    
    train_csv_path = output_dir / "train_dataset.csv"
    val_csv_path = output_dir / "val_dataset.csv"
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    logger.info(f"Created train/val split: {len(train_df)} train, {len(val_df)} val samples")
    
    return csv_path, train_csv_path, val_csv_path

def create_training_config(base_dir: Path, train_csv: Path, val_csv: Path) -> Path:
    """Create a training configuration file."""
    config = {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "processed_data_dir": str(base_dir / "processed"),
        "vocab_size": 3,  # Number of labels (normal, abnormal, arrhythmia)
        "batch_size": 8,
        "epochs": 10,
        "learning_rate": 1e-5,
        "patience": 5,
        "task": "multi_label",
        "target_metric": "f1_score"
    }
    
    config_path = base_dir / "metadata" / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created training config: {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description="Prepare ECG training data for HuBERT-ECG")
    parser.add_argument("--input_dir", type=str, default="training_data/raw", 
                       help="Directory containing raw ECG files")
    parser.add_argument("--output_dir", type=str, default="training_data", 
                       help="Base output directory for processed data")
    
    args = parser.parse_args()
    
    logger.info("Starting ECG training data preparation...")
    
    # Create directory structure
    base_dir = create_directory_structure()
    
    # Find ECG files
    raw_dir = Path(args.input_dir)
    if not raw_dir.exists():
        logger.error(f"Input directory {raw_dir} does not exist!")
        logger.info("Please create the directory and add your ECG files (.npy format)")
        logger.info("Example: mkdir -p training_data/raw && cp your_ecg_files.npy training_data/raw/")
        return
    
    ecg_files = find_ecg_files(raw_dir)
    
    if not ecg_files:
        logger.warning(f"No ECG files found in {raw_dir}")
        logger.info("Please add ECG files in .npy format to the raw directory")
        logger.info("ECG files should be numpy arrays with shape (12, n_samples) representing 12-lead ECGs")
        return
    
    # Process ECG files
    processed_dir = base_dir / "processed"
    processed_files = []
    
    logger.info(f"Processing {len(ecg_files)} ECG files...")
    for ecg_file in ecg_files:
        if ecg_file.suffix == '.npy':
            result = process_npy_file(ecg_file, processed_dir)
            processed_files.append(result)
        else:
            logger.warning(f"Skipping unsupported file type: {ecg_file}")
    
    # Create training CSV
    try:
        csv_path, train_csv, val_csv = create_training_csv(processed_files, base_dir / "metadata")
        
        # Create training config
        config_path = create_training_config(base_dir, train_csv, val_csv)
        
        logger.success("Training data preparation complete!")
        logger.info(f"Training CSV: {train_csv}")
        logger.info(f"Validation CSV: {val_csv}")
        logger.info(f"Training config: {config_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review and edit the labels in the CSV files")
        logger.info("2. Run 'make train' to start training")
        
    except Exception as e:
        logger.error(f"Failed to create training files: {e}")

if __name__ == "__main__":
    main()
