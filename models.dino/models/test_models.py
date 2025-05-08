import pytest
import torch
from models.ecg_transformer import ECGTransformer
from models.asymmetric_dino import DINOV2


@pytest.fixture
def sample_ecg():
    return torch.randn(2, 1000)  # batch of 2, 1000 samples


def test_ecg_transformer_forward(sample_ecg):
    """Test ECG Transformer forward pass."""
    model = ECGTransformer(in_channels=1, embed_dim=64)
    output = model(sample_ecg)
    assert output.shape == (2, 64)  # batch_size, embed_dim


def test_ecg_transformer_with_metadata():
    """Test ECG Transformer with metadata."""
    model = ECGTransformer(in_channels=1, embed_dim=64, use_metadata=True)
    ecg = torch.randn(2, 1000)
    metadata = {'age': torch.tensor([0.5, 0.6])}
    output = model(ecg, metadata)
    assert output.shape == (2, 64)


def test_dino_model_forward(sample_ecg):
    """Test DINO model forward pass."""
    student = ECGTransformer(in_channels=1, embed_dim=64)
    teacher = ECGTransformer(in_channels=12, embed_dim=128)
    model = DINOV2(student, teacher)

    student_out, teacher_out = model(
        student_input=sample_ecg,
        teacher_input=torch.randn(2, 12, 1000)
    )
    assert student_out.shape == teacher_out.shape