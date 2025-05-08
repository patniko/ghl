import pytest
import torch
from models.ecg_transformer import ECGTransformer
from models.asymmetric_dino import DINOV2


class TestECGTransformer:
    def test_output_shape(self):
        model = ECGTransformer(in_channels=1, embed_dim=128)
        x = torch.randn(3, 1000)  # batch of 3, 1000 samples
        out = model(x)
        assert out.shape == (3, 128)

    def test_metadata_integration(self):
        model = ECGTransformer(in_channels=1, embed_dim=64, use_metadata=True)
        x = torch.randn(2, 1000)
        metadata = {'age': torch.tensor([0.5, 0.6])}
        out = model(x, metadata)
        assert not torch.isnan(out).any()


class TestDINOV2:
    def test_asymmetry(self):
        student = ECGTransformer(in_channels=1, embed_dim=64)
        teacher = ECGTransformer(in_channels=12, embed_dim=256)
        model = DINOV2(student, teacher)

        student_out, teacher_out = model(
            torch.randn(2, 1000),
            torch.randn(2, 12, 1000)
        )
        assert student_out.shape[1] == teacher_out.shape[1]  # Same projection dim