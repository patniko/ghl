import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from .augmentations import ECGAugmentation


class CustomECGDataset(Dataset):
    """Custom dataset for asymmetric ECG learning with metadata support.

    Handles loading of custom ECG datasets with:
    - Different lead configurations for student/teacher models
    - Patient metadata integration
    - Data augmentation
    - Flexible data loading
    """

    # Standard 12-lead ECG order for reference
    STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def __init__(self,
                 data_path: str,
                 sample_rate: int = 100,
                 duration: float = 10.0,
                 student_leads: List[str] = ['I'],
                 teacher_leads: List[str] = STANDARD_LEADS,
                 augment: bool = False,
                 metadata: bool = True,
                 metadata_schema: Optional[Dict] = None):
        """
        Args:
            data_path: Path to directory containing ECG data
            sample_rate: Sampling frequency in Hz
            duration: Duration of ECG segments in seconds
            student_leads: List of leads for student model
            teacher_leads: List of leads for teacher model
            augment: Whether to apply data augmentation
            metadata: Whether to load metadata
            metadata_schema: Dictionary specifying metadata fields and normalization
        """
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.total_samples = int(sample_rate * duration)
        self.student_leads = student_leads
        self.teacher_leads = teacher_leads
        self.augment = augment
        self.metadata = metadata
        self.metadata_schema = metadata_schema or {
            'age': {'scale': 100.0},
            'sex': {'categories': ['F', 'M']},
            'height': {'scale': 200.0},
            'weight': {'scale': 150.0}
        }

        # Validate lead configurations
        self._validate_leads()

        # Initialize data structures
        self.records = self._load_records()
        self.augmentor = ECGAugmentation(sample_rate) if augment else None

    def _validate_leads(self) -> None:
        """Validate that requested leads are available."""
        invalid_student = [lead for lead in self.student_leads if lead not in self.STANDARD_LEADS]
        invalid_teacher = [lead for lead in self.teacher_leads if lead not in self.STANDARD_LEADS]

        if invalid_student:
            raise ValueError(f"Invalid student leads: {invalid_student}. Valid leads are: {self.STANDARD_LEADS}")
        if invalid_teacher:
            raise ValueError(f"Invalid teacher leads: {invalid_teacher}. Valid leads are: {self.STANDARD_LEADS}")

    def _load_records(self) -> List[Dict]:
        """Load dataset records from directory structure.

        Expected directory structure:
        data_path/
        ├── ecg/
        │   ├── record1.npy
        │   ├── record2.npy
        │   └── ...
        └── metadata.csv (optional)

        Returns:
            List of dictionaries containing 'ecg_path' and 'metadata'
        """
        records = []
        ecg_dir = os.path.join(self.data_path, 'ecg')

        if not os.path.exists(ecg_dir):
            raise FileNotFoundError(f"ECG directory not found at {ecg_dir}")

        # Load ECG files
        for ecg_file in sorted(os.listdir(ecg_dir)):
            if ecg_file.endswith('.npy'):
                record = {
                    'ecg_path': os.path.join(ecg_dir, ecg_file),
                    'metadata': None
                }
                records.append(record)

        # Load metadata if available
        if self.metadata:
            metadata_path = os.path.join(self.data_path, 'metadata.csv')
            if os.path.exists(metadata_path):
                import pandas as pd
                metadata_df = pd.read_csv(metadata_path)
                for i, record in enumerate(records):
                    if i < len(metadata_df):
                        record['metadata'] = metadata_df.iloc[i].to_dict()

        if not records:
            raise RuntimeError(f"No valid ECG records found in {ecg_dir}")

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a sample from the dataset.

        Returns:
            Tuple of (student_ecg, teacher_ecg, metadata)
        """
        record = self.records[idx]

        # Load ECG data
        ecg = np.load(record['ecg_path'])  # Shape: (leads, time)

        # Select requested leads
        student_ecg = self._select_leads(ecg, self.student_leads)
        teacher_ecg = self._select_leads(ecg, self.teacher_leads)

        # Convert to tensors and normalize
        student_ecg = self._process_ecg(student_ecg)
        teacher_ecg = self._process_ecg(teacher_ecg)

        # Process metadata
        metadata = self._process_metadata(record.get('metadata')) if self.metadata else {}

        return student_ecg, teacher_ecg, metadata

    def _select_leads(self, ecg: np.ndarray, leads: List[str]) -> np.ndarray:
        """Select specified leads from ECG data."""
        lead_indices = [self.STANDARD_LEADS.index(lead) for lead in leads]
        return ecg[lead_indices]

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

    def _process_metadata(self, metadata: Optional[Dict]) -> Dict:
        """Normalize and encode metadata."""
        if not metadata:
            return {}

        processed = {}

        for field, schema in self.metadata_schema.items():
            if field in metadata:
                if 'scale' in schema:
                    # Continuous value normalization
                    processed[field] = torch.tensor(metadata[field] / schema['scale'])
                elif 'categories' in schema:
                    # Categorical value encoding
                    processed[field] = torch.tensor(schema['categories'].index(metadata[field]))

        return processed

    @property
    def sample_shape(self) -> Dict[str, Tuple[int]]:
        """Get the shape of samples for each component."""
        return {
            'student_ecg': (len(self.student_leads), self.total_samples),
            'teacher_ecg': (len(self.teacher_leads), self.total_samples),
            'metadata': (len(self.metadata_schema),) if self.metadata else (0,)
        }