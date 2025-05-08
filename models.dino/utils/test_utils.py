import pytest
from utils.config import load_config
from utils.metrics import compute_metrics
from utils.logger import setup_logger
import tempfile

def test_config_loading():
    """Test YAML config loading."""
    config_content = """
    data:
      ptbxl_path: "./data/ptbxl"
    model:
      student_dim: 256
    training:
      epochs: 100
    """
    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(config_content)
        f.flush()
        config = load_config(f.name)
        assert config.model.student_dim == 256
        assert config.training.epochs == 100

def test_metrics_computation():
    """Test metric calculations."""
    student = torch.randn(4, 128)
    teacher = torch.randn(4, 128)
    metrics = compute_metrics(student, teacher)
    assert 'cosine_sim' in metrics
    assert 0 <= metrics['cosine_sim'] <= 1

def test_logger_setup(tmp_path):
    """Test logger initialization."""
    logger = setup_logger(tmp_path)
    assert len(logger.handlers) == 2  # File and stream handlers