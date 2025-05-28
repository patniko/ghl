#!/usr/bin/env python3
"""
Training Utilities and Configuration for EchoPrime Fine-tuning

This script provides additional utilities and configuration options
to make EchoPrime training easier and more flexible.
"""

import os
import yaml
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EchoPrimeConfigManager:
    """Configuration manager for EchoPrime training"""
    
    DEFAULT_CONFIG = {
        "model": {
            "video_encoder": "mvit_v2_s",
            "text_encoder": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "embedding_dim": 512,
            "video_input_size": [224, 224, 16],
            "max_text_length": 512,
            "num_views": 58,
            "temperature": 1.0
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 4e-5,
            "weight_decay": 1e-6,
            "num_epochs": 60,
            "warmup_epochs": 5,
            "gradient_clip_norm": 1.0,
            "accumulation_steps": 1
        },
        "fine_tuning": {
            "mode": "full",  # Options: "full", "lora", "linear_probe", "adapter"
            "freeze_video_layers": 6,
            "freeze_text_layers": 6,
            "unfreeze_schedule": None,  # Gradual unfreezing schedule
            "lora_config": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"]
            }
        },
        "data": {
            "augmentation": {
                "horizontal_flip": 0.5,
                "rotation": 10,
                "brightness": 0.2,
                "contrast": 0.2,
                "temporal_crop": True,
                "temporal_shift": 0.1
            },
            "preprocessing": {
                "normalize_videos": True,
                "resize_videos": True,
                "target_fps": 30,
                "min_frames": 8,
                "max_frames": 32
            }
        },
        "optimization": {
            "optimizer": "adamw",  # Options: "adamw", "sgd", "adam"
            "scheduler": "cosine",  # Options: "cosine", "plateau", "step", "linear"
            "warmup_steps": 1000,
            "cosine_restarts": False,
            "plateau_patience": 5,
            "step_size": 10,
            "step_gamma": 0.1
        },
        "regularization": {
            "dropout": 0.1,
            "label_smoothing": 0.1,
            "mixup_alpha": 0.2,
            "cutmix_alpha": 1.0,
            "use_ema": False,
            "ema_decay": 0.999
        },
        "logging": {
            "use_wandb": True,
            "wandb_project": "echoprime-finetuning",
            "log_interval": 100,
            "eval_interval": 1000,
            "save_interval": 5,
            "log_gradients": False,
            "log_weights": False
        },
        "paths": {
            "data_dir": "./data",
            "output_dir": "./outputs",
            "pretrained_path": None,
            "resume_from": None
        },
        "hardware": {
            "num_workers": 4,
            "pin_memory": True,
            "mixed_precision": True,
            "compile_model": False,
            "distributed": False
        }
    }
    
    @classmethod
    def create_config(cls, config_path: str = None, **overrides) -> Dict[str, Any]:
        """Create configuration from file and overrides"""
        config = cls.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
            config = cls._deep_update(config, file_config)
        
        # Apply overrides
        config = cls._deep_update(config, overrides)
        
        return config
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], path: str):
        """Save configuration to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(config, f, default_flow_style=False, indent=2)
            else:
                json.dump(config, f, indent=2)
    
    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively update nested dictionary"""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = EchoPrimeConfigManager._deep_update(result[key], value)
            else:
                result[key] = value
        return result


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation (LoRA) layer"""
    
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scale


class EchoPrimeModelWrapper:
    """Wrapper for EchoPrime model with various fine-tuning strategies"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.mode = config['fine_tuning']['mode']
        
        if self.mode == "lora":
            self._add_lora_adapters()
        elif self.mode == "linear_probe":
            self._freeze_backbone()
        elif self.mode == "adapter":
            self._add_adapter_layers()
    
    def _add_lora_adapters(self):
        """Add LoRA adapters to the model"""
        lora_config = self.config['fine_tuning']['lora_config']
        target_modules = lora_config['target_modules']
        
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace linear layer with LoRA version
                    lora_layer = self._create_lora_linear(module, lora_config)
                    self._replace_module(name, lora_layer)
        
        # Freeze original parameters
        for name, param in self.model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
    
    def _create_lora_linear(self, original_layer: nn.Linear, lora_config: Dict) -> nn.Module:
        """Create LoRA version of linear layer"""
        class LoRALinear(nn.Module):
            def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float):
                super().__init__()
                self.original = original
                self.lora = LoRAAdapter(
                    original.in_features, 
                    original.out_features, 
                    rank, 
                    alpha, 
                    dropout
                )
                
                # Freeze original parameters
                for param in self.original.parameters():
                    param.requires_grad = False
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.original(x) + self.lora(x)
        
        return LoRALinear(
            original_layer,
            lora_config['rank'],
            lora_config['alpha'],
            lora_config['dropout']
        )
    
    def _freeze_backbone(self):
        """Freeze backbone for linear probing"""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name and 'head' not in name:
                param.requires_grad = False
    
    def _add_adapter_layers(self):
        """Add adapter layers to the model"""
        # This is a simplified adapter implementation
        # In practice, you might want to use more sophisticated adapters
        pass
    
    def _replace_module(self, name: str, new_module: nn.Module):
        """Replace a module in the model"""
        parts = name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)


class TrainingScheduler:
    """Advanced training scheduler with gradual unfreezing"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.unfreeze_schedule = config['fine_tuning'].get('unfreeze_schedule')
        
    def step(self, epoch: int):
        """Update training schedule based on epoch"""
        if self.unfreeze_schedule:
            self._gradual_unfreeze(epoch)
    
    def _gradual_unfreeze(self, epoch: int):
        """Gradually unfreeze layers based on schedule"""
        for schedule_item in self.unfreeze_schedule:
            if epoch >= schedule_item['epoch']:
                layer_pattern = schedule_item['layers']
                self._unfreeze_layers(layer_pattern)
    
    def _unfreeze_layers(self, pattern: str):
        """Unfreeze layers matching pattern"""
        for name, param in self.model.named_parameters():
            if pattern in name:
                param.requires_grad = True
                logger.info(f"Unfroze layer: {name}")


class MetricsTracker:
    """Track and compute training metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def compute_averages(self, window: int = None) -> Dict[str, float]:
        """Compute average metrics over window"""
        averages = {}
        for key, values in self.metrics.items():
            if window:
                values = values[-window:]
            averages[key] = sum(values) / len(values) if values else 0
        return averages
    
    def reset(self):
        """Reset metrics"""
        self.metrics = {}
    
    def save_history(self, epoch: int):
        """Save current metrics to history"""
        current_metrics = self.compute_averages()
        current_metrics['epoch'] = epoch
        self.history.append(current_metrics)
    
    def get_best_metric(self, metric_name: str, mode: str = 'min') -> float:
        """Get best metric value from history"""
        if not self.history:
            return float('inf') if mode == 'min' else float('-inf')
        
        values = [h.get(metric_name, float('inf') if mode == 'min' else float('-inf')) 
                 for h in self.history]
        
        return min(values) if mode == 'min' else max(values)


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        
    def __call__(self, metric: float) -> bool:
        """Check if training should stop"""
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.mode == 'min':
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta
        
        if improved:
            self.best_score = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class ModelUtils:
    """Utility functions for model operations"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    
    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    @staticmethod
    def save_model_info(model: nn.Module, path: str):
        """Save detailed model information"""
        info = {
            'parameters': ModelUtils.count_parameters(model),
            'size_mb': ModelUtils.get_model_size(model),
            'architecture': str(model),
            'state_dict_keys': list(model.state_dict().keys())
        }
        
        with open(path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
    
    @staticmethod
    def load_pretrained_weights(model: nn.Module, pretrained_path: str, strict: bool = False):
        """Load pretrained weights with error handling"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out incompatible keys
            model_keys = set(model.state_dict().keys())
            pretrained_keys = set(state_dict.keys())
            
            missing_keys = model_keys - pretrained_keys
            unexpected_keys = pretrained_keys - model_keys
            
            if missing_keys:
                logger.warning(f"Missing keys in pretrained model: {list(missing_keys)[:10]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in pretrained model: {list(unexpected_keys)[:10]}...")
            
            # Load compatible weights
            compatible_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            model.load_state_dict(compatible_state_dict, strict=strict)
            
            logger.info(f"Loaded {len(compatible_state_dict)} layers from pretrained model")
            
        except Exception as e:
            logger.error(f"Error loading pretrained weights: {e}")
            raise


class DataAugmentation:
    """Advanced data augmentation for echocardiogram videos"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['data']['augmentation']
    
    def __call__(self, video: torch.Tensor, text: str = None) -> torch.Tensor:
        """Apply augmentations to video"""
        if self.config.get('horizontal_flip', 0) > 0:
            video = self._random_horizontal_flip(video)
        
        if self.config.get('rotation', 0) > 0:
            video = self._random_rotation(video)
        
        if self.config.get('brightness', 0) > 0 or self.config.get('contrast', 0) > 0:
            video = self._random_color_jitter(video)
        
        if self.config.get('temporal_crop', False):
            video = self._temporal_crop(video)
        
        if self.config.get('temporal_shift', 0) > 0:
            video = self._temporal_shift(video)
        
        return video
    
    def _random_horizontal_flip(self, video: torch.Tensor) -> torch.Tensor:
        """Randomly flip video horizontally"""
        if torch.rand(1) < self.config['horizontal_flip']:
            return torch.flip(video, [-1])
        return video
    
    def _random_rotation(self, video: torch.Tensor) -> torch.Tensor:
        """Randomly rotate video frames"""
        angle = (torch.rand(1) - 0.5) * 2 * self.config['rotation']
        # Note: This is a simplified rotation - in practice you'd use torchvision transforms
        return video
    
    def _random_color_jitter(self, video: torch.Tensor) -> torch.Tensor:
        """Randomly adjust brightness and contrast"""
        brightness_factor = 1.0 + (torch.rand(1) - 0.5) * 2 * self.config.get('brightness', 0)
        contrast_factor = 1.0 + (torch.rand(1) - 0.5) * 2 * self.config.get('contrast', 0)
        
        video = video * brightness_factor
        video = (video - 0.5) * contrast_factor + 0.5
        video = torch.clamp(video, 0, 1)
        
        return video
    
    def _temporal_crop(self, video: torch.Tensor) -> torch.Tensor:
        """Randomly crop temporal dimension"""
        T = video.size(0)
        if T > 16:
            start_idx = torch.randint(0, T - 16, (1,)).item()
            video = video[start_idx:start_idx + 16]
        return video
    
    def _temporal_shift(self, video: torch.Tensor) -> torch.Tensor:
        """Randomly shift frames in time"""
        shift_amount = int(torch.rand(1) * self.config['temporal_shift'] * video.size(0))
        if shift_amount > 0:
            video = torch.roll(video, shift_amount, dims=0)
        return video


class ExperimentTracker:
    """Track experiments and hyperparameter sweeps"""
    
    def __init__(self, base_config: Dict[str, Any], output_dir: str):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.experiments = []
    
    def add_experiment(self, name: str, config_overrides: Dict[str, Any]):
        """Add an experiment to the tracker"""
        experiment_config = EchoPrimeConfigManager._deep_update(
            self.base_config.copy(), 
            config_overrides
        )
        
        experiment = {
            'name': name,
            'config': experiment_config,
            'output_dir': self.output_dir / name,
            'status': 'pending'
        }
        
        self.experiments.append(experiment)
        return experiment
    
    def run_experiments(self, run_function):
        """Run all experiments"""
        for experiment in self.experiments:
            if experiment['status'] == 'pending':
                try:
                    logger.info(f"Running experiment: {experiment['name']}")
                    experiment['status'] = 'running'
                    
                    # Create output directory
                    experiment['output_dir'].mkdir(parents=True, exist_ok=True)
                    
                    # Save experiment config
                    config_path = experiment['output_dir'] / 'config.yaml'
                    EchoPrimeConfigManager.save_config(experiment['config'], str(config_path))
                    
                    # Run experiment
                    results = run_function(experiment['config'])
                    
                    # Save results
                    results_path = experiment['output_dir'] / 'results.json'
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    experiment['status'] = 'completed'
                    experiment['results'] = results
                    
                    logger.info(f"Completed experiment: {experiment['name']}")
                    
                except Exception as e:
                    logger.error(f"Failed experiment {experiment['name']}: {e}")
                    experiment['status'] = 'failed'
                    experiment['error'] = str(e)
    
    def save_summary(self):
        """Save experiment summary"""
        summary = {
            'total_experiments': len(self.experiments),
            'completed': len([e for e in self.experiments if e['status'] == 'completed']),
            'failed': len([e for e in self.experiments if e['status'] == 'failed']),
            'experiments': self.experiments
        }
        
        summary_path = self.output_dir / 'experiment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


def create_hyperparameter_sweep(base_config: Dict[str, Any], 
                               sweep_params: Dict[str, List],
                               output_dir: str) -> ExperimentTracker:
    """Create hyperparameter sweep experiments"""
    
    tracker = ExperimentTracker(base_config, output_dir)
    
    # Generate all combinations
    import itertools
    
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    
    for i, combination in enumerate(itertools.product(*values)):
        config_overrides = {}
        name_parts = []
        
        for key, value in zip(keys, combination):
            # Handle nested config keys
            if '.' in key:
                parts = key.split('.')
                current = config_overrides
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_overrides[key] = value
            
            name_parts.append(f"{key.split('.')[-1]}_{value}")
        
        experiment_name = f"sweep_{i:03d}_" + "_".join(name_parts)
        tracker.add_experiment(experiment_name, config_overrides)
    
    return tracker


def setup_distributed_training():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        return True, rank, world_size, local_rank
    
    return False, 0, 1, 0


# Example usage and configuration templates

EXAMPLE_CONFIGS = {
    "quick_test": {
        "training": {
            "batch_size": 8,
            "num_epochs": 5,
            "learning_rate": 1e-4
        },
        "model": {
            "video_input_size": [112, 112, 8]
        }
    },
    
    "full_finetuning": {
        "training": {
            "batch_size": 32,
            "num_epochs": 60,
            "learning_rate": 4e-5
        },
        "fine_tuning": {
            "mode": "full",
            "freeze_video_layers": 0,
            "freeze_text_layers": 0
        }
    },
    
    "lora_efficient": {
        "training": {
            "batch_size": 64,
            "num_epochs": 30,
            "learning_rate": 1e-3
        },
        "fine_tuning": {
            "mode": "lora",
            "lora_config": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1
            }
        }
    },
    
    "linear_probe": {
        "training": {
            "batch_size": 128,
            "num_epochs": 20,
            "learning_rate": 1e-2
        },
        "fine_tuning": {
            "mode": "linear_probe"
        }
    }
}

SWEEP_EXAMPLES = {
    "learning_rate_sweep": {
        "training.learning_rate": [1e-5, 4e-5, 1e-4, 4e-4],
        "training.batch_size": [16, 32]
    },
    
    "architecture_sweep": {
        "fine_tuning.freeze_video_layers": [0, 3, 6],
        "fine_tuning.freeze_text_layers": [0, 3, 6],
        "training.learning_rate": [4e-5, 1e-4]
    },
    
    "lora_sweep": {
        "fine_tuning.lora_config.rank": [8, 16, 32],
        "fine_tuning.lora_config.alpha": [16, 32, 64],
        "training.learning_rate": [1e-4, 1e-3]
    }
}


def main():
    """Example usage of the configuration system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EchoPrime Training Utilities")
    parser.add_argument("--create_config", type=str, choices=list(EXAMPLE_CONFIGS.keys()),
                       help="Create example configuration")
    parser.add_argument("--output_path", type=str, default="./config.yaml",
                       help="Output path for configuration")
    parser.add_argument("--create_sweep", type=str, choices=list(SWEEP_EXAMPLES.keys()),
                       help="Create hyperparameter sweep")
    parser.add_argument("--sweep_output", type=str, default="./experiments",
                       help="Output directory for sweep")
    
    args = parser.parse_args()
    
    if args.create_config:
        # Create example configuration
        base_config = EchoPrimeConfigManager.DEFAULT_CONFIG
        example_overrides = EXAMPLE_CONFIGS[args.create_config]
        
        config = EchoPrimeConfigManager.create_config(overrides=example_overrides)
        EchoPrimeConfigManager.save_config(config, args.output_path)
        
        print(f"Created {args.create_config} configuration at {args.output_path}")
    
    elif args.create_sweep:
        # Create hyperparameter sweep
        base_config = EchoPrimeConfigManager.DEFAULT_CONFIG
        sweep_params = SWEEP_EXAMPLES[args.create_sweep]
        
        tracker = create_hyperparameter_sweep(base_config, sweep_params, args.sweep_output)
        
        print(f"Created {len(tracker.experiments)} experiments for {args.create_sweep}")
        print(f"Experiment configurations saved to {args.sweep_output}")
        
        # Save sweep summary
        tracker.save_summary()
    
    else:
        print("Available example configurations:")
        for name, config in EXAMPLE_CONFIGS.items():
            print(f"  {name}: {config.get('description', 'No description')}")
        
        print("\nAvailable sweep examples:")
        for name in SWEEP_EXAMPLES.keys():
            print(f"  {name}")


if __name__ == "__main__":
    main()