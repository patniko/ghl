#!/usr/bin/env python3
"""
EchoPrime Training Runner

Simple script to run EchoPrime training with different configurations.
This script integrates all the components and provides an easy interface.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch

# Add current directory to path to import local modules
sys.path.append(str(Path(__file__).parent))

try:
    from echoprime_finetune import EchoPrimeConfig, EchoPrimeModel, EchoPrimeTrainer, create_data_loaders, setup_model
    from training_utils import EchoPrimeConfigManager, ModelUtils, EXAMPLE_CONFIGS
    from data_preparation import EchoDataPreprocessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all scripts are in the same directory")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EchoPrimeRunner:
    """Main runner class for EchoPrime training"""
    
    def __init__(self, config_path: str = None, **config_overrides):
        """Initialize runner with configuration"""
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config_dict = EchoPrimeConfigManager.create_config(config_path, **config_overrides)
        else:
            self.config_dict = EchoPrimeConfigManager.create_config(**config_overrides)
        
        # Convert to legacy config object for compatibility
        self.config = EchoPrimeConfig()
        self._update_config_from_dict()
        
        # Setup logging
        self._setup_logging()
        
    def _update_config_from_dict(self):
        """Update config object from dictionary"""
        # Model settings
        model_config = self.config_dict.get('model', {})
        self.config.embedding_dim = model_config.get('embedding_dim', 512)
        self.config.video_input_size = tuple(model_config.get('video_input_size', [224, 224, 16]))
        self.config.max_text_length = model_config.get('max_text_length', 512)
        self.config.num_views = model_config.get('num_views', 58)
        self.config.temperature = model_config.get('temperature', 1.0)
        
        # Training settings
        training_config = self.config_dict.get('training', {})
        self.config.batch_size = training_config.get('batch_size', 32)
        self.config.learning_rate = training_config.get('learning_rate', 4e-5)
        self.config.weight_decay = training_config.get('weight_decay', 1e-6)
        self.config.num_epochs = training_config.get('num_epochs', 60)
        
        # Fine-tuning settings
        ft_config = self.config_dict.get('fine_tuning', {})
        self.config.freeze_video_layers = ft_config.get('freeze_video_layers', 6)
        self.config.freeze_text_layers = ft_config.get('freeze_text_layers', 6)
        self.config.finetune_mode = ft_config.get('mode', 'full')
        
        # Paths
        paths_config = self.config_dict.get('paths', {})
        self.config.data_dir = paths_config.get('data_dir', './data')
        self.config.output_dir = paths_config.get('output_dir', './outputs')
        self.config.pretrained_path = paths_config.get('pretrained_path')
        
        # Logging
        logging_config = self.config_dict.get('logging', {})
        self.config.use_wandb = logging_config.get('use_wandb', True)
        self.config.log_interval = logging_config.get('log_interval', 100)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
    
    def prepare_data(self, 
                    data_type: str = "sample",
                    input_dir: str = None,
                    num_samples: int = 100,
                    **kwargs):
        """Prepare training data"""
        logger.info(f"Preparing {data_type} data...")
        
        if input_dir is None:
            input_dir = self.config.data_dir
        
        preprocessor = EchoDataPreprocessor(
            input_dir=input_dir,
            output_dir=self.config.data_dir,
            test_size=kwargs.get('test_size', 0.2),
            val_size=kwargs.get('val_size', 0.1)
        )
        
        if data_type == "sample":
            preprocessor.create_sample_data(num_samples)
        elif data_type == "dicom":
            preprocessor.prepare_from_dicom(
                kwargs['dicom_dir'],
                kwargs['reports_file'],
                kwargs.get('view_labels_file')
            )
        elif data_type == "videos":
            preprocessor.prepare_from_videos(
                kwargs['videos_dir'],
                kwargs['metadata_file']
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Validate prepared data
        if not preprocessor.validate_data():
            raise RuntimeError("Data validation failed")
        
        logger.info("Data preparation completed successfully")
    
    def train(self):
        """Run training"""
        logger.info("Starting EchoPrime training...")
        
        # Print configuration summary
        self._print_config_summary()
        
        # Check if data exists
        if not self._check_data_exists():
            logger.error("Training data not found. Please run data preparation first.")
            return False
        
        try:
            # Create data loaders
            logger.info("Creating data loaders...")
            train_loader, val_loader = create_data_loaders(self.config)
            logger.info(f"Training samples: {len(train_loader.dataset)}")
            logger.info(f"Validation samples: {len(val_loader.dataset)}")
            
            # Setup model
            logger.info("Setting up model...")
            model = setup_model(self.config)
            
            # Print model info
            model_info = ModelUtils.count_parameters(model)
            logger.info(f"Model parameters: {model_info}")
            
            # Save model info
            ModelUtils.save_model_info(model, os.path.join(self.config.output_dir, 'model_info.json'))
            
            # Create trainer
            logger.info("Creating trainer...")
            trainer = EchoPrimeTrainer(self.config, model, train_loader, val_loader)
            
            # Save configuration
            config_path = os.path.join(self.config.output_dir, 'config.yaml')
            EchoPrimeConfigManager.save_config(self.config_dict, config_path)
            
            # Start training
            logger.info("Starting training loop...")
            trainer.train()
            
            logger.info("Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, checkpoint_path: str = None):
        """Evaluate trained model"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config.output_dir, 'best_model.pth')
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        
        logger.info(f"Evaluating model from {checkpoint_path}")
        
        # Load model
        model = setup_model(self.config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create test data loader
        from echoprime_finetune import EchoDataset
        from torch.utils.data import DataLoader
        
        test_dataset = EchoDataset(self.config.data_dir, "test", self.config)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Run evaluation
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch, return_loss=True)
                total_loss += outputs['loss'].item()
        
        avg_loss = total_loss / len(test_loader)
        logger.info(f"Test loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _check_data_exists(self) -> bool:
        """Check if training data exists"""
        data_dir = Path(self.config.data_dir)
        required_files = ['train_reports.csv', 'val_reports.csv']
        
        for file in required_files:
            if not (data_dir / file).exists():
                return False
        
        videos_dir = data_dir / 'videos'
        if not videos_dir.exists() or len(list(videos_dir.glob('*.avi'))) == 0:
            return False
        
        return True
    
    def _print_config_summary(self):
        """Print configuration summary"""
        logger.info("=" * 60)
        logger.info("ECHOPRIME TRAINING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Data directory: {self.config.data_dir}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Number of epochs: {self.config.num_epochs}")
        logger.info(f"Fine-tuning mode: {self.config.finetune_mode}")
        logger.info(f"Frozen video layers: {self.config.freeze_video_layers}")
        logger.info(f"Frozen text layers: {self.config.freeze_text_layers}")
        logger.info(f"Use wandb: {self.config.use_wandb}")
        if self.config.pretrained_path:
            logger.info(f"Pretrained weights: {self.config.pretrained_path}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="EchoPrime Training Runner")
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Prepare data command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare training data')
    prepare_parser.add_argument('--data_type', choices=['sample', 'dicom', 'videos'], 
                               default='sample', help='Type of data to prepare')
    prepare_parser.add_argument('--input_dir', type=str, help='Input data directory')
    prepare_parser.add_argument('--output_dir', type=str, default='./data', 
                               help='Output directory for processed data')
    prepare_parser.add_argument('--num_samples', type=int, default=100,
                               help='Number of sample data entries (for sample mode)')
    
    # DICOM specific arguments
    prepare_parser.add_argument('--dicom_dir', type=str, help='DICOM files directory')
    prepare_parser.add_argument('--reports_file', type=str, help='Reports CSV file')
    prepare_parser.add_argument('--view_labels_file', type=str, help='View labels CSV file')
    
    # Video specific arguments
    prepare_parser.add_argument('--videos_dir', type=str, help='Video files directory')
    prepare_parser.add_argument('--metadata_file', type=str, help='Metadata CSV file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, help='Configuration file path')
    train_parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    train_parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    train_parser.add_argument('--pretrained_path', type=str, help='Pretrained model path')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=4e-5, help='Learning rate')
    train_parser.add_argument('--num_epochs', type=int, default=60, help='Number of epochs')
    train_parser.add_argument('--mode', choices=['full', 'lora', 'linear_probe'], 
                             default='full', help='Fine-tuning mode')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--config', type=str, help='Configuration file path')
    eval_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    eval_parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    
    # Quick start command
    quick_parser = subparsers.add_parser('quickstart', help='Quick start with sample data')
    quick_parser.add_argument('--num_samples', type=int, default=50, help='Number of sample videos')
    quick_parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    quick_parser.add_argument('--output_dir', type=str, default='./quickstart_output', 
                             help='Output directory')
    
    # Example configs command
    example_parser = subparsers.add_parser('example', help='Create example configuration')
    example_parser.add_argument('config_name', choices=list(EXAMPLE_CONFIGS.keys()),
                               help='Example configuration name')
    example_parser.add_argument('--output', type=str, default='./example_config.yaml',
                               help='Output configuration file')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        # Prepare data
        runner = EchoPrimeRunner(data_dir=args.output_dir)
        
        kwargs = {}
        if args.data_type == 'dicom':
            kwargs.update({
                'dicom_dir': args.dicom_dir,
                'reports_file': args.reports_file,
                'view_labels_file': args.view_labels_file
            })
        elif args.data_type == 'videos':
            kwargs.update({
                'videos_dir': args.videos_dir,
                'metadata_file': args.metadata_file
            })
        
        runner.prepare_data(
            data_type=args.data_type,
            input_dir=args.input_dir,
            num_samples=args.num_samples,
            **kwargs
        )
    
    elif args.command == 'train':
        # Train model
        config_overrides = {
            'paths': {
                'data_dir': args.data_dir,
                'output_dir': args.output_dir,
                'pretrained_path': args.pretrained_path
            },
            'training': {
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'num_epochs': args.num_epochs
            },
            'fine_tuning': {
                'mode': args.mode
            }
        }
        
        runner = EchoPrimeRunner(args.config, **config_overrides)
        success = runner.train()
        
        if success:
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Model saved to: {args.output_dir}")
            print(f"Logs saved to: {args.output_dir}/logs")
            print(f"Best model: {args.output_dir}/best_model.pth")
            print("\nTo evaluate the model, run:")
            print(f"python {__file__} eval --data_dir {args.data_dir} --checkpoint {args.output_dir}/best_model.pth")
        else:
            print("Training failed. Check logs for details.")
            sys.exit(1)
    
    elif args.command == 'eval':
        # Evaluate model
        config_overrides = {
            'paths': {
                'data_dir': args.data_dir
            }
        }
        
        runner = EchoPrimeRunner(args.config, **config_overrides)
        test_loss = runner.evaluate(args.checkpoint)
        
        print(f"\nEvaluation Results:")
        print(f"Test Loss: {test_loss:.4f}")
    
    elif args.command == 'quickstart':
        # Quick start with sample data
        print("="*60)
        print("ECHOPRIME QUICKSTART")
        print("="*60)
        print("This will:")
        print(f"1. Create {args.num_samples} sample echocardiogram videos")
        print(f"2. Train a model for {args.epochs} epochs")
        print(f"3. Save everything to {args.output_dir}")
        print("="*60)
        
        # Use quick test configuration
        config_overrides = EXAMPLE_CONFIGS['quick_test'].copy()
        config_overrides.update({
            'paths': {
                'data_dir': f"{args.output_dir}/data",
                'output_dir': f"{args.output_dir}/model"
            },
            'training': {
                'num_epochs': args.epochs,
                'batch_size': 8  # Small batch for quick test
            }
        })
        
        runner = EchoPrimeRunner(**config_overrides)
        
        # Prepare sample data
        print("Step 1: Preparing sample data...")
        runner.prepare_data(
            data_type='sample',
            num_samples=args.num_samples
        )
        
        # Train model
        print("Step 2: Training model...")
        success = runner.train()
        
        if success:
            print("\n" + "="*60)
            print("QUICKSTART COMPLETED!")
            print("="*60)
            print(f"Sample data: {args.output_dir}/data")
            print(f"Trained model: {args.output_dir}/model")
            print(f"Configuration: {args.output_dir}/model/config.yaml")
            print("\nYou can now:")
            print("1. Explore the sample data structure")
            print("2. Modify the configuration for your real data")
            print("3. Replace sample data with your real echocardiogram data")
        else:
            print("Quickstart failed. Check logs for details.")
            sys.exit(1)
    
    elif args.command == 'example':
        # Create example configuration
        config = EchoPrimeConfigManager.create_config(
            overrides=EXAMPLE_CONFIGS[args.config_name]
        )
        EchoPrimeConfigManager.save_config(config, args.output)
        
        print(f"Created {args.config_name} configuration at {args.output}")
        print(f"You can now run: python {__file__} train --config {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()