import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import wfdb
from typing import List, Dict, Optional, Tuple
from .augmentations import ECGAugmentation


class PTBXLDataset(Dataset):
    """PTB-XL ECG dataset loader with support for asymmetric learning.

    Args:
        root_dir: Path to PTB-XL dataset directory
        sample_rate: Sampling frequency in Hz (default: 100)
        duration: ECG segment duration in seconds (default: 10.0)
        leads: List of lead names to include (default: ['I'])
        augment: Enable data augmentation (default: False)
        metadata: Include patient metadata (default: True)
        lazy_load: Load ECG data on demand (default: False)
        strat_fold: Which stratification fold to use (1-10, None for all)
    """

    # Standard 12-lead ECG order in PTB-XL
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def __init__(self,
                 root_dir: str,
                 sample_rate: int = 100,
                 duration: float = 10.0,
                 leads: List[str] = ['I'],
                 augment: bool = False,
                 metadata: bool = True,
                 lazy_load: bool = False,
                 strat_fold: Optional[int] = None):

        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.total_samples = int(sample_rate * duration)
        self.leads = leads
        self.augment = augment
        self.metadata = metadata
        self.lazy_load = lazy_load
        self.strat_fold = strat_fold

        # Validate leads
        self._validate_leads()

        # Load records and ECG data
        self.records = self._load_records()
        self.ecg_data = {} if lazy_load else self._load_all_ecg_data()

        # Initialize augmentation
        self.augmentor = ECGAugmentation(sample_rate) if augment else None

    def _validate_leads(self) -> None:
        """Validate that requested leads are available."""
        invalid_leads = [lead for lead in self.leads if lead not in self.LEAD_NAMES]
        if invalid_leads:
            raise ValueError(f"Invalid leads specified: {invalid_leads}. "
                             f"Valid leads are: {self.LEAD_NAMES}")

    def _load_records(self) -> pd.DataFrame:
        """Load and preprocess PTB-XL metadata records."""
        records_path = os.path.join(self.root_dir, 'records.csv')
        if not os.path.exists(records_path):
            raise FileNotFoundError(f"PTB-XL records.csv not found at {records_path}")

        records = pd.read_csv(records_path)

        # Filter by stratification fold if specified
        if self.strat_fold is not None:
            if not 1 <= self.strat_fold <= 10:
                raise ValueError("strat_fold must be between 1 and 10")
            records = records[records['strat_fold'] == self.strat_fold]

        # Convert age to numeric (handles 'nan' strings)
        records['age'] = pd.to_numeric(records['age'], errors='coerce')

        return records

    def _load_all_ecg_data(self) -> Dict[int, np.ndarray]:
        """Preload all ECG data into memory."""
        ecg_data = {}
        for idx, row in self.records.iterrows():
            ecg_path = os.path.join(self.root_dir, row['filename_hr'])
            ecg_data[idx] = self._load_ecg_file(ecg_path)
        return ecg_data

    def _load_ecg_file(self, path: str) -> np.ndarray:
        """Load single ECG file from WFDB format."""
        try:
            # Remove file extension if present
            if path.endswith('.mat'):
                path = path[:-4]
            elif path.endswith('.hea') or path.endswith('.dat'):
                path = path[:-4]
                
            # Read the WFDB record
            record = wfdb.rdrecord(path)
            
            # Transpose to get shape (12, time_steps)
            return record.p_signal.T
        except Exception as e:
            raise RuntimeError(f"Failed to load ECG file {path}: {str(e)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get a sample from the dataset.

        Returns:
            Tuple of (ecg_tensor, metadata_dict)
        """
        record = self.records.iloc[idx]

        # Load ECG data
        if self.lazy_load:
            ecg_path = os.path.join(self.root_dir, record['filename_hr'])
            ecg = self._load_ecg_file(ecg_path)
        else:
            ecg = self.ecg_data[idx]

        # Select requested leads
        lead_indices = [self.LEAD_NAMES.index(lead) for lead in self.leads]
        ecg = ecg[lead_indices]

        # Convert to tensor and process
        ecg_tensor = self._process_ecg(ecg)

        # Get metadata if enabled
        metadata = self._get_metadata(record) if self.metadata else {}

        return ecg_tensor, metadata

    def _process_ecg(self, ecg: np.ndarray) -> torch.Tensor:
        """Convert ECG to tensor, normalize, and adjust length."""
        ecg_tensor = torch.from_numpy(ecg.astype(np.float32))

        # Normalize per lead
        mean = ecg_tensor.mean(dim=-1, keepdim=True)
        std = ecg_tensor.std(dim=-1, keepdim=True)
        ecg_tensor = (ecg_tensor - mean) / (std + 1e-6)

        # Adjust length
        if ecg_tensor.shape[-1] > self.total_samples:
            start = torch.randint(0, ecg_tensor.shape[-1] - self.total_samples, (1,)).item()
            ecg_tensor = ecg_tensor[..., start:start + self.total_samples]
        elif ecg_tensor.shape[-1] < self.total_samples:
            pad_len = self.total_samples - ecg_tensor.shape[-1]
            ecg_tensor = F.pad(ecg_tensor, (0, pad_len))

        # Apply augmentation if enabled
        if self.augment and self.augmentor:
            ecg_tensor = self.augmentor(ecg_tensor)

        return ecg_tensor

    def _get_metadata(self, record: pd.Series) -> Dict[str, torch.Tensor]:
        """Extract and normalize metadata from record."""
        metadata = {}

        # Age (normalized to [0,1] range)
        if pd.notna(record['age']):
            metadata['age'] = torch.tensor(record['age'] / 100.0)

        # Sex (0 for female, 1 for male)
        if pd.notna(record['sex']):
            # Handle both string and numeric values
            if isinstance(record['sex'], str):
                metadata['sex'] = torch.tensor(0 if record['sex'].lower() == 'female' else 1)
            else:
                # Assume 0 is female, 1 is male (or any non-zero value is male)
                metadata['sex'] = torch.tensor(int(record['sex'] != 0))

        # Height in meters (normalized assuming max 2m)
        if pd.notna(record['height']):
            metadata['height'] = torch.tensor(record['height'] / 200.0)

        # Weight in kg (normalized assuming max 150kg)
        if pd.notna(record['weight']):
            metadata['weight'] = torch.tensor(record['weight'] / 150.0)

        return metadata

    @property
    def sample_shape(self) -> Tuple[int, int]:
        """Get the shape of ECG samples (leads, timesteps)."""
        return (len(self.leads), self.total_samples)

    @property
    def metadata_fields(self) -> List[str]:
        """Get available metadata fields."""
        return ['age', 'sex', 'height', 'weight'] if self.metadata else []
