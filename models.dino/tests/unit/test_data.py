import pytest
import numpy as np
from data.ptbxl import PTBXLDataset
from data.custom_dataset import CustomECGDataset


class TestPTBXLDataset:
    def test_length(self, mock_ptbxl_data):
        dataset = PTBXLDataset(root_dir=mock_ptbxl_data)
        assert len(dataset) == 2

    def test_item_shape(self, mock_ptbxl_data):
        dataset = PTBXLDataset(root_dir=mock_ptbxl_data, leads=['I'])
        ecg, _ = dataset[0]
        assert ecg.shape == (1, 1000)  # Single lead, 10s at 100Hz


class TestCustomECGDataset:
    def test_lead_selection(self, tmp_path):
        os.makedirs(tmp_path / "ecg", exist_ok=True)
        np.save(str(tmp_path / "ecg/sample1.npy"), np.random.randn(12, 1000))

        dataset = CustomECGDataset(data_path=tmp_path,
                                   student_leads=['I'],
                                   teacher_leads=['II', 'V5'])
        student_ecg, teacher_ecg, _ = dataset[0]
        assert student_ecg.shape[0] == 1  # 1 student lead
        assert teacher_ecg.shape[0] == 2  # 2 teacher leads