import yaml
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class DataConfig:
    ptbxl_path: str
    custom_dataset_path: str
    batch_size: int
    num_workers: int
    sample_rate: int = 100
    duration: float = 10.0
    lead_names: List[str] = field(default_factory=lambda: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

@dataclass
class ModelConfig:
    student_dim: int = 256
    teacher_dim: int = 768
    student_depth: int = 8
    teacher_depth: int = 12
    student_num_heads: int = 8
    teacher_num_heads: int = 12
    mlp_ratio: float = 4.0
    patch_size: int = 32
    overlap: int = 8
    num_patches: int = 128
    dropout: float = 0.1
    out_dim: int = 65536
    metadata_dim: int = 64  # Dimension for metadata embedding

@dataclass
class TrainingConfig:
    epochs: int = 100
    warmup_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.04
    momentum_teacher: float = 0.996
    center_momentum: float = 0.9
    teacher_temp: float = 0.04
    warmup_teacher_temp: float = 0.04
    freeze_teacher_backbone: bool = True
    freeze_student_backbone: bool = False
    output_dir: str = "./output"
    save_every: int = 10
    log_every: int = 10

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    phase: str = "pretrain"  # or "finetune"

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    data_config = DataConfig(**config_dict['data'])
    model_config = ModelConfig(**config_dict['model'])
    training_config = TrainingConfig(**config_dict['training'])
    
    if 'phase' in config_dict:
        phase = config_dict['phase']
    else:
        phase = "pretrain"
    
    return Config(data=data_config, model=model_config, training=training_config, phase=phase)
