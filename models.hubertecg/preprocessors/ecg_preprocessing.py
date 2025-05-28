"""
ECG data preprocessing utilities for HuBERT-ECG.

This module contains functions for preprocessing ECG data according to the 
HuBERT-ECG paper specifications.
"""

import torch
import numpy as np


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


def normalize_ecg(ecg_data, method='z_score'):
    """
    Normalize ECG data using specified method.
    
    Args:
        ecg_data: ECG data tensor
        method: Normalization method ('z_score', 'min_max', 'robust')
        
    Returns:
        Normalized ECG data
    """
    if method == 'z_score':
        mean = torch.mean(ecg_data)
        std = torch.std(ecg_data)
        return (ecg_data - mean) / (std + 1e-8)
    elif method == 'min_max':
        min_val = torch.min(ecg_data)
        max_val = torch.max(ecg_data)
        return (ecg_data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = torch.median(ecg_data)
        mad = torch.median(torch.abs(ecg_data - median))
        return (ecg_data - median) / (mad + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def filter_ecg(ecg_data, sampling_rate=500, low_freq=0.5, high_freq=40):
    """
    Apply bandpass filter to ECG data.
    
    Args:
        ecg_data: ECG data tensor
        sampling_rate: Sampling rate of the ECG data
        low_freq: Low cutoff frequency
        high_freq: High cutoff frequency
        
    Returns:
        Filtered ECG data
    """
    # This is a placeholder for filtering functionality
    # In practice, you would use scipy.signal or similar for filtering
    # For now, we'll just return the original data
    return ecg_data


def prepare_ecg_for_inference(ecg_data, sampling_rate=500, target_length=5000, 
                             downsampling_factor=5, normalize=True):
    """
    Complete preprocessing pipeline for ECG inference.
    
    Args:
        ecg_data: Raw ECG data
        sampling_rate: Original sampling rate
        target_length: Target length after preprocessing
        downsampling_factor: Downsampling factor
        normalize: Whether to normalize the data
        
    Returns:
        Preprocessed ECG data ready for model inference
    """
    # Preprocess
    processed_ecg = preprocess_ecg(
        ecg_data, 
        sampling_rate=sampling_rate,
        target_length=target_length,
        downsampling_factor=downsampling_factor
    )
    
    # Normalize if requested
    if normalize:
        processed_ecg = normalize_ecg(processed_ecg, method='z_score')
    
    return processed_ecg
