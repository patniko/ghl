import pytest
import torch
import numpy as np
from data.ptbxl import PTBXLDataset
from data.custom_dataset import CustomECGDataset
from data.augmentations import ECGAugmentation


@pytest.fixture
def mock_ptbxl_data(tmp_path):
    """Create mock PTB-XL data structure for testing."""
    # Create mock records.csv
    records_content = """ecg_id,filename_hr,strat_fold,age,sex,height,weight
1,ecg_1.mat,1,50,male,175,80
2,ecg_2.mat,2,60,female,165,70"""
    (tmp_path / "records.csv").write_text(records_content)

    # Create mock ECG files
    os.makedirs(tmp_path / "ecg", exist_ok=True)
    for i in range(1, 3):
        np.save(str(tmp_path / f"ecg/ecg_{i}.npy"), np.random.randn(12, 1000))
    return tmp_path


def test_ptbxl_dataset_loading(mock_ptbxl_data):
    """Test PTB-XL dataset loads correctly."""
    dataset = PTBXLDataset(root_dir=mock_ptbxl_data, lazy_load=True)
    assert len(dataset) == 2
    ecg, metadata = dataset[0]
    assert ecg.shape == (12, 1000)  # Default leads=all
    assert 'age' in metadata


def test_custom_dataset_loading(tmp_path):
    """Test custom dataset initialization."""
    # Create mock data structure
    os.makedirs(tmp_path / "ecg", exist_ok=True)
    np.save(str(tmp_path / "ecg/sample1.npy"), np.random.randn(12, 1000))

    metadata_content = """age,sex,height,weight
50,M,175,80"""
    (tmp_path / "metadata.csv").write_text(metadata_content)

    dataset = CustomECGDataset(data_path=tmp_path)
    assert len(dataset) == 1
    student_ecg, teacher_ecg, metadata = dataset[0]
    assert student_ecg.shape[0] == 1  # Default student leads=['I']


def test_augmentations():
    """Test ECG augmentation transforms."""
    augmentor = ECGAugmentation(sample_rate=100)
    ecg = torch.randn(12, 1000)
    augmented = augmentor(ecg)
    assert ecg.shape == augmented.shape