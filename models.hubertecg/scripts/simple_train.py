#!/usr/bin/env python3
"""
Simple training wrapper for HuBERT-ECG.
This script provides an easy interface to train the model with prepared data.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from loguru import logger

# Add the project root and training/models to the path so we can import training modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'training'))
sys.path.append(os.path.join(project_root, 'training', 'models'))

def load_training_config(config_path: Path) -> dict:
    """Load training configuration from JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded training config from {config_path}")
    return config

def validate_training_data(config: dict) -> bool:
    """Validate that training data exists and is properly formatted."""
    train_csv = Path(config["train_csv"])
    val_csv = Path(config["val_csv"])
    processed_dir = Path(config["processed_data_dir"])
    
    if not train_csv.exists():
        logger.error(f"Training CSV not found: {train_csv}")
        return False
    
    if not val_csv.exists():
        logger.error(f"Validation CSV not found: {val_csv}")
        return False
    
    if not processed_dir.exists():
        logger.error(f"Processed data directory not found: {processed_dir}")
        return False
    
    # Check if processed directory has .npy files
    npy_files = list(processed_dir.glob("*.npy"))
    if not npy_files:
        logger.error(f"No .npy files found in {processed_dir}")
        return False
    
    logger.info(f"Found {len(npy_files)} processed ECG files")
    return True

def run_training(config: dict, args: argparse.Namespace):
    """Run the training process using the existing finetune.py script."""
    
    # Import the training module
    try:
        # Import and patch the checkpoint path
        import training.finetune as finetune_module
        from training.finetune import finetune
        import argparse as training_argparse
        
        # Override the hardcoded checkpoint path
        checkpoint_base = str(Path("training_data") / "checkpoints")
        finetune_module.SUPERVISED_MODEL_CKPT_PATH = checkpoint_base + "/"
        
        logger.info(f"Successfully imported training modules")
        logger.info(f"Set checkpoint path to: {finetune_module.SUPERVISED_MODEL_CKPT_PATH}")
        
    except ImportError as e:
        logger.error(f"Failed to import training modules: {e}")
        logger.error("Make sure you're running this from the models.hubertecg directory")
        logger.error("Also ensure all dependencies are installed with 'poetry install'")
        return False
    
    # Create training arguments based on config and user input
    train_args = training_argparse.Namespace()
    
    # Required arguments
    train_args.train_iteration = args.train_iteration
    train_args.path_to_dataset_csv_train = config["train_csv"]
    train_args.path_to_dataset_csv_val = config["val_csv"]
    train_args.vocab_size = config["vocab_size"]
    train_args.patience = config["patience"]
    train_args.batch_size = config["batch_size"]
    train_args.target_metric = config["target_metric"]
    train_args.task = config["task"]
    
    # Optional arguments with defaults
    train_args.sweep_dir = "simple_training"
    train_args.ramp_up_perc = 0.08
    train_args.training_steps = None
    train_args.epochs = config.get("epochs", 10)
    train_args.val_interval = None
    train_args.downsampling_factor = None
    train_args.random_crop = False
    train_args.accumulation_steps = 1
    train_args.label_start_index = 4  # After filename, original_path, shape, duration_seconds
    train_args.freezing_steps = None
    train_args.resume_finetuning = False
    train_args.unfreeze_conv_embedder = False
    train_args.transformer_blocks_to_unfreeze = 0
    train_args.lr = config.get("learning_rate", 1e-5)
    train_args.layer_wise_lr = False
    train_args.load_path = args.pretrained_model if args.pretrained_model else None
    train_args.classifier_hidden_size = None
    train_args.use_label_embedding = False
    train_args.intervals_for_penalty = 3
    train_args.dynamic_reg = False
    train_args.use_loss_weights = True
    train_args.random_init = args.random_init
    train_args.largeness = args.model_size if args.random_init else None
    train_args.weight_decay_mult = 1
    train_args.model_dropout_mult = 0
    train_args.finetuning_layerdrop = 0.1
    train_args.wandb_run_name = f"simple_training_{args.train_iteration}"
    
    # Create output directory for checkpoints
    checkpoint_dir = Path("training_data") / "checkpoints" / train_args.sweep_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the checkpoint path that finetune.py expects
    os.environ['SUPERVISED_MODEL_CKPT_PATH'] = str(Path("training_data") / "checkpoints")
    
    logger.info("Starting training with the following configuration:")
    logger.info(f"  Train CSV: {train_args.path_to_dataset_csv_train}")
    logger.info(f"  Val CSV: {train_args.path_to_dataset_csv_val}")
    logger.info(f"  Vocab size: {train_args.vocab_size}")
    logger.info(f"  Batch size: {train_args.batch_size}")
    logger.info(f"  Epochs: {train_args.epochs}")
    logger.info(f"  Learning rate: {train_args.lr}")
    logger.info(f"  Task: {train_args.task}")
    logger.info(f"  Target metric: {train_args.target_metric}")
    logger.info(f"  Checkpoints will be saved to: {checkpoint_dir}")
    
    try:
        # Run the training
        finetune(train_args)
        logger.success("Training completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple training for HuBERT-ECG")
    parser.add_argument("--config", type=str, default="training_data/metadata/training_config.json",
                       help="Path to training configuration file")
    parser.add_argument("--train_iteration", type=int, choices=[1, 2, 3], default=1,
                       help="Training iteration (1, 2, or 3)")
    parser.add_argument("--pretrained_model", type=str, default=None,
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--random_init", action="store_true",
                       help="Use random initialization instead of pretrained model")
    parser.add_argument("--model_size", type=str, choices=["small", "base", "large"], default="base",
                       help="Model size for random initialization")
    parser.add_argument("--prepare_data", action="store_true",
                       help="Run data preparation before training")
    
    args = parser.parse_args()
    
    logger.info("Starting simple HuBERT-ECG training...")
    
    # Run data preparation if requested
    if args.prepare_data:
        logger.info("Running data preparation...")
        from scripts.prepare_simple_training_data import main as prepare_main
        prepare_main()
    
    # Load training configuration
    config_path = Path(args.config)
    try:
        config = load_training_config(config_path)
    except FileNotFoundError:
        logger.error(f"Training config not found: {config_path}")
        logger.info("Run 'make prepare-data' first to create training data and config")
        return
    
    # Validate training data
    if not validate_training_data(config):
        logger.error("Training data validation failed")
        logger.info("Run 'make prepare-data' to prepare your training data")
        return
    
    # Check for pretrained model or random init
    if not args.random_init and not args.pretrained_model:
        logger.warning("No pretrained model specified and random_init not set")
        logger.info("Using random initialization. For better results, download a pretrained model:")
        logger.info("  https://huggingface.co/Edoardo-BS")
        args.random_init = True
    
    # Run training
    success = run_training(config, args)
    
    if success:
        logger.success("Training completed successfully!")
        logger.info("Check training_data/checkpoints/ for saved models")
    else:
        logger.error("Training failed")

if __name__ == "__main__":
    main()
