import pytest
import torch
from data.ptbxl import PTBXLDataset
from models.asymmetric_dino import DINOV2
from models.ecg_transformer import ECGTransformer


@pytest.mark.integration
def test_full_pipeline(tmp_path):
    """Test complete pipeline from data loading to model forward."""
    # Create mock data
    os.makedirs(tmp_path / "ecg", exist_ok=True)
    np.save(str(tmp_path / "ecg/sample1.npy"), np.random.randn(12, 1000))

    # Initialize components
    dataset = PTBXLDataset(root_dir=tmp_path, leads=['I', 'II'])
    student = ECGTransformer(in_channels=1, embed_dim=64)
    teacher = ECGTransformer(in_channels=2, embed_dim=64)
    model = DINOV2(student, teacher)

    # Test pipeline
    ecg, metadata = dataset[0]
    student_input = ecg[0:1]  # First lead for student
    teacher_input = ecg  # All leads for teacher

    student_out, teacher_out = model(student_input.unsqueeze(0),
                                     teacher_input.unsqueeze(0))
    assert student_out.shape == teacher_out.shape