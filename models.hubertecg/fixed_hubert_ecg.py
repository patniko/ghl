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
