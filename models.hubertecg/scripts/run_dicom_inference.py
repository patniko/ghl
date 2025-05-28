#!/usr/bin/env python3
"""
Script to run HuBERT-ECG inference on converted DICOM NPY files.

This script:
1. Loads converted NPY files from the DICOM conversion
2. Preprocesses them for the HuBERT-ECG model
3. Runs inference using a pre-trained model
4. Saves the results

Usage:
    python scripts/run_dicom_inference.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--model_size MODEL_SIZE]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel

# Add the training/models directory to the path
sys.path.append('training/models')
sys.path.append('.')

# Import HuBERT-ECG modules
from hubert_ecg import HuBERTECG, HuBERTECGConfig
from hubert_ecg_classification import HuBERTForECGClassification


def preprocess_ecg(ecg_data, sampling_rate=500, target_length=5000, downsampling_factor=5):
    """
    Preprocess ECG data according to the HuBERT-ECG paper specifications.
    
    Args:
        ecg_data: Raw ECG data with shape (n_leads, n_samples)
        sampling_rate: Original sampling rate of the ECG data
        target_length: Target length of the ECG data after preprocessing
        downsampling_factor: Factor to downsample the ECG data
        
    Returns:
        Preprocessed ECG data with shape (batch_size, n_samples) for HuBERT model
    """
    # Ensure the ECG data has the right shape
    if len(ecg_data.shape) == 1:
        ecg_data = ecg_data.reshape(1, -1)  # Single lead
    
    # Pad or truncate to target length
    n_leads, n_samples = ecg_data.shape
    if n_samples < target_length:
        # Pad with zeros
        padded_data = np.zeros((n_leads, target_length))
        padded_data[:, :n_samples] = ecg_data
        ecg_data = padded_data
    elif n_samples > target_length:
        # Truncate
        ecg_data = ecg_data[:, :target_length]
    
    # Downsample
    if downsampling_factor > 1:
        ecg_data = ecg_data[:, ::downsampling_factor]
    
    # For multi-lead ECG, we need to flatten the leads into a single channel
    # HuBERT expects input of shape (batch_size, sequence_length)
    # We'll concatenate all leads into a single sequence
    flattened_ecg = ecg_data.reshape(-1)  # Flatten all leads into a single sequence
    
    # Convert to tensor and add batch dimension
    ecg_tensor = torch.tensor(flattened_ecg, dtype=torch.float32).unsqueeze(0)
    
    return ecg_tensor


def run_inference(model, ecg_data):
    """
    Run inference on ECG data using the pre-trained model.
    
    Args:
        model: Pre-trained HuBERT-ECG model
        ecg_data: Preprocessed ECG data
        
    Returns:
        Model predictions
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        # Ensure the input has the correct shape for the model
        # HuBERT expects input of shape (batch_size, sequence_length)
        if len(ecg_data.shape) > 2:
            # If we have a shape like [1, 12, 1000], reshape to [1, 12000]
            batch_size = ecg_data.shape[0]
            ecg_data = ecg_data.reshape(batch_size, -1)
        
        outputs = model(ecg_data.to(device))
    
    return outputs


def load_model(model_size='base', device='cpu'):
    """
    Load a pre-trained HuBERT-ECG model from Hugging Face.
    
    Args:
        model_size (str): Model size ('small', 'base', or 'large')
        device (str): Device to load the model on
        
    Returns:
        Loaded model
    """
    model_name = f"Edoardo-BS/hubert-ecg-{model_size}"
    print(f"Loading model: {model_name}")
    
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"Error loading model from Hugging Face: {e}")
        print("Please check your internet connection or download the model manually")
        return None


def process_patient_npy_files(patient_dir, output_dir, model, device):
    """
    Process all NPY files in a single patient directory.
    
    Args:
        patient_dir (str): Directory containing NPY files for one patient
        output_dir (str): Directory to save results for this patient
        model: Pre-trained HuBERT-ECG model
        device (str): Device for computation
        
    Returns:
        list: Results for this patient
    """
    patient_name = os.path.basename(patient_dir)
    print(f"\nProcessing patient: {patient_name}")
    
    # Create patient-specific output directory
    patient_output_dir = os.path.join(output_dir, patient_name)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # Find all NPY files in this patient directory
    npy_files = []
    for file in os.listdir(patient_dir):
        if file.lower().endswith('.npy'):
            npy_files.append(os.path.join(patient_dir, file))
    
    if not npy_files:
        print(f"  No NPY files found in {patient_dir}")
        return []
    
    print(f"  Found {len(npy_files)} NPY files")
    
    results = []
    
    for npy_path in npy_files:
        try:
            # Load ECG data
            ecg_data = np.load(npy_path)
            print(f"    Processing: {os.path.basename(npy_path)}")
            print(f"      Original shape: {ecg_data.shape}")
            
            # Preprocess ECG data
            preprocessed_ecg = preprocess_ecg(ecg_data)
            print(f"      Preprocessed shape: {preprocessed_ecg.shape}")
            
            # Run inference
            with torch.no_grad():
                outputs = run_inference(model, preprocessed_ecg)
                
                # Extract features or predictions depending on model output
                if hasattr(outputs, 'last_hidden_state'):
                    # If it's a transformer output, get the last hidden state
                    features = outputs.last_hidden_state
                    print(f"      Features shape: {features.shape}")
                elif isinstance(outputs, torch.Tensor):
                    # If it's a direct tensor output
                    features = outputs
                    print(f"      Output shape: {features.shape}")
                else:
                    # If it's a tuple or other structure, take the first element
                    features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    print(f"      Output shape: {features.shape}")
            
            # Save results
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            
            # Save features as NPY
            features_path = os.path.join(patient_output_dir, f"{base_name}_features.npy")
            features_cpu = features.cpu().numpy()
            np.save(features_path, features_cpu)
            
            # Store metadata for CSV
            results.append({
                'patient': patient_name,
                'filename': os.path.basename(npy_path),
                'original_shape': str(ecg_data.shape),
                'preprocessed_shape': str(preprocessed_ecg.shape),
                'features_shape': str(features_cpu.shape),
                'features_file': f"{patient_name}/{base_name}_features.npy",
                'status': 'success'
            })
            
            print(f"      Saved features to: {features_path}")
            
        except Exception as e:
            print(f"      Error processing {os.path.basename(npy_path)}: {e}")
            results.append({
                'patient': patient_name,
                'filename': os.path.basename(npy_path),
                'original_shape': 'N/A',
                'preprocessed_shape': 'N/A',
                'features_shape': 'N/A',
                'features_file': 'N/A',
                'status': f'error: {str(e)}'
            })
    
    print(f"  Patient {patient_name}: {len([r for r in results if r['status'] == 'success'])}/{len(npy_files)} files processed successfully")
    return results


def process_all_patients_npy(input_dir, output_dir, model, device):
    """
    Process all patient folders containing NPY files and run inference.
    
    Args:
        input_dir (str): Directory containing patient folders with NPY files
        output_dir (str): Directory to save results
        model: Pre-trained HuBERT-ECG model
        device (str): Device for computation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all patient directories
    patient_dirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.') and item != '__pycache__':
            patient_dirs.append(item_path)
    
    if not patient_dirs:
        print(f"No patient directories found in {input_dir}")
        return
    
    print(f"Found {len(patient_dirs)} patient directories: {[os.path.basename(d) for d in patient_dirs]}")
    
    all_results = []
    total_successful = 0
    total_files = 0
    
    for patient_dir in patient_dirs:
        patient_results = process_patient_npy_files(patient_dir, output_dir, model, device)
        all_results.extend(patient_results)
        
        # Count successes and total files
        patient_successful = len([r for r in patient_results if r['status'] == 'success'])
        total_successful += patient_successful
        total_files += len(patient_results)
    
    # Save combined results summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv_path = os.path.join(output_dir, 'inference_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nResults summary saved to: {results_csv_path}")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total patients processed: {len(patient_dirs)}")
    print(f"Total NPY files: {total_files}")
    print(f"Successfully processed: {total_successful}")
    print(f"Success rate: {total_successful/total_files*100:.1f}%" if total_files > 0 else "No files to process")


def analyze_features(output_dir):
    """
    Analyze the extracted features and provide summary statistics.
    
    Args:
        output_dir (str): Directory containing the results
    """
    results_csv_path = os.path.join(output_dir, 'inference_results.csv')
    
    if not os.path.exists(results_csv_path):
        print("No results CSV found for analysis")
        return
    
    df = pd.read_csv(results_csv_path)
    successful_results = df[df['status'] == 'success']
    
    if len(successful_results) == 0:
        print("No successful results to analyze")
        return
    
    print(f"\nFeature Analysis:")
    print(f"  Successfully processed files: {len(successful_results)}")
    
    # Analyze feature shapes
    feature_shapes = successful_results['features_shape'].value_counts()
    print(f"  Feature shapes:")
    for shape, count in feature_shapes.items():
        print(f"    {shape}: {count} files")
    
    # Load and analyze a sample feature file
    sample_file = successful_results.iloc[0]['features_file']
    sample_path = os.path.join(output_dir, sample_file)
    
    if os.path.exists(sample_path):
        sample_features = np.load(sample_path)
        print(f"  Sample feature statistics:")
        print(f"    Shape: {sample_features.shape}")
        print(f"    Data type: {sample_features.dtype}")
        print(f"    Value range: {sample_features.min():.6f} to {sample_features.max():.6f}")
        print(f"    Mean: {sample_features.mean():.6f}")
        print(f"    Std: {sample_features.std():.6f}")


def main():
    parser = argparse.ArgumentParser(description='Run HuBERT-ECG inference on converted DICOM NPY files.')
    parser.add_argument('--input_dir', type=str, default='data',
                        help='Directory containing NPY files (default: data)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save inference results (default: results)')
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'],
                        help='HuBERT-ECG model size (default: base)')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Only analyze existing results without running inference')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_features(args.output_dir)
        return
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        print("Please run 'make dicom-setup' first to convert DICOM files to NPY format")
        sys.exit(1)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_size, device)
    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    print(f"Running inference on NPY files from {args.input_dir}")
    print(f"Results will be saved to {args.output_dir}")
    
    # Process files
    process_all_patients_npy(args.input_dir, args.output_dir, model, device)
    
    # Analyze results
    analyze_features(args.output_dir)


if __name__ == "__main__":
    main()
