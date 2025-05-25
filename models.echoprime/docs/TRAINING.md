# EchoPrime Fine-tuning Framework

A comprehensive framework for fine-tuning the EchoPrime multi-view echocardiography vision-language model. This repository provides easy-to-use scripts for data preparation, model training, and evaluation.

## 🚀 Quick Start

### Option 1: Quick Demo (Recommended for first-time users)

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick demo with sample data
python run_training.py quickstart --num_samples 50 --epochs 5
```

This will create 50 sample echocardiogram videos and train a model for 5 epochs. Perfect for testing the framework!

### Option 2: Full Training Pipeline

```bash
# 1. Prepare your data
python run_training.py prepare --data_type videos \
    --videos_dir /path/to/your/videos \
    --metadata_file /path/to/metadata.csv \
    --output_dir ./processed_data

# 2. Train the model
python run_training.py train \
    --data_dir ./processed_data \
    --output_dir ./trained_model \
    --batch_size 32 \
    --learning_rate 4e-5 \
    --num_epochs 60

# 3. Evaluate the model
python run_training.py eval \
    --data_dir ./processed_data \
    --checkpoint ./trained_model/best_model.pth
```

## 📁 Project Structure

```
echoprime-finetuning/
├── echoprime_finetune.py      # Main training script
├── data_preparation.py        # Data preprocessing utilities
├── training_utils.py          # Training utilities and configurations
├── run_training.py           # Easy-to-use runner script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 📊 Data Format

### For Video Data
Your data should be organized as follows:

```
data/
├── videos/
│   ├── video_001.avi
│   ├── video_002.avi
│   └── ...
├── train_reports.csv
├── val_reports.csv
└── test_reports.csv
```

### CSV Format
Each CSV file should contain:
- `video_id`: Unique identifier for the video (without extension)
- `report_text`: Echocardiogram report text
- `view_label`: View classification (0-57 for standard views)
- `labels`: JSON string with additional labels (optional)

Example CSV:
```csv
video_id,report_text,view_label,labels
video_001,"Normal left ventricular function. LVEF 60%.",5,"{\"ejection_fraction\": 60}"
video_002,"Mild mitral regurgitation. Normal chamber sizes.",12,"{\"mitral_regurg\": 1}"
```

## 🛠️ Training Modes

### 1. Full Fine-tuning
```bash
python run_training.py train --mode full --learning_rate 4e-5
```
Trains all parameters. Best performance but requires more GPU memory.

### 2. LoRA (Low-Rank Adaptation)
```bash
python run_training.py train --mode lora --learning_rate 1e-3
```
Parameter-efficient fine-tuning. Good performance with less memory.

### 3. Linear Probing
```bash
python run_training.py train --mode linear_probe --learning_rate 1e-2
```
Only trains the final classification layers. Fastest but limited adaptation.

## ⚙️ Configuration

### Using Configuration Files
Create custom configurations:

```bash
# Create example configuration
python run_training.py example lora_efficient --output my_config.yaml

# Use configuration file
python run_training.py train --config my_config.yaml
```

### Configuration Options

Key configuration sections:

- **Model**: Architecture settings, embedding dimensions
- **Training**: Batch size, learning rate, epochs
- **Fine-tuning**: Freezing strategies, LoRA parameters
- **Data**: Augmentation, preprocessing options
- **Logging**: Wandb integration, logging intervals

## 🔧 Advanced Features

### Hyperparameter Sweeps
```python
from training_utils import create_hyperparameter_sweep, EchoPrimeConfigManager

base_config = EchoPrimeConfigManager.DEFAULT_CONFIG
sweep_params = {
    "training.learning_rate": [1e-5, 4e-5, 1e-4],
    "training.batch_size": [16, 32, 64]
}

tracker = create_hyperparameter_sweep(base_config, sweep_params, "./experiments")
```

### Custom Data Augmentation
```python
from training_utils import DataAugmentation

config = {
    "data": {
        "augmentation": {
            "horizontal_flip": 0.5,
            "rotation": 10,
            "brightness": 0.2,
            "temporal_crop": True
        }
    }
}
```

### Distributed Training
```bash
# Multi-GPU training
torchrun --nproc_per_node=4 echoprime_finetune.py \
    --data_dir ./data \
    --batch_size 32
```

## 📈 Monitoring and Logging

### Weights & Biases Integration
The framework automatically logs to Wandb when enabled:

- Training and validation losses
- Learning rate schedules
- Model gradients (optional)
- System metrics

### Local Logging
All training logs are saved to:
- `{output_dir}/logs/training.log`: Detailed training logs
- `{output_dir}/model_info.json`: Model architecture info
- `{output_dir}/config.yaml`: Training configuration

## 🏥 Working with Medical Data

### DICOM Support
```bash
python run_training.py prepare --data_type dicom \
    --dicom_dir /path/to/dicom/files \
    --reports_file /path/to/reports.csv \
    --view_labels_file /path/to/views.csv
```

### Privacy and Security
- All data is processed locally
- No data is sent to external services (except Wandb if enabled)
- DICOM files are automatically de-identified during processing

## 🔍 Evaluation and Analysis

### Model Evaluation
```bash
# Evaluate on test set
python run_training.py eval \
    --data_dir ./data \
    --checkpoint ./model/best_model.pth
```

### Custom Evaluation Metrics
The framework supports evaluation on:
- Contrastive retrieval tasks
- View classification accuracy
- Clinical label prediction
- Cross-modal retrieval performance

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
   ```yaml
   training:
     batch_size: 16
     accumulation_steps: 2
   ```

2. **Slow Training**: Enable mixed precision
   ```yaml
   hardware:
     mixed_precision: true
   ```

3. **Data Loading Errors**: Check video file formats and paths
   ```bash
   python data_preparation.py --validate
   ```

### Memory Optimization
- Use smaller video input sizes for quick experiments
- Enable gradient checkpointing for large models
- Use LoRA for parameter-efficient training

## 📚 Citation

If you use this framework, please cite the original EchoPrime paper:

```bibtex
@article{vukadinovic2024echoprime,
  title={EchoPrime: A Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation},
  author={Vukadinovic, Milos and Tang, Xiu and Yuan, Neal and Cheng, Paul and Li, Debiao and Cheng, Susan and He, Bryan and Ouyang, David},
  journal={arXiv preprint arXiv:2410.09704},
  year={2024}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the original EchoPrime repository for details.

## 🆘 Support

- **Issues**: Report bugs or request features via GitHub issues
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check the inline documentation in the code

## 🔄 Version History

- **v1.0.0**: Initial release with full fine-tuning support
- **v1.1.0**: Added LoRA and configuration management
- **v1.2.0**: Added data preparation utilities and quick start

---

**Happy Training! 🎉**

For more information about the EchoPrime model, visit the [original repository](https://github.com/echonet/EchoPrime) and read the [paper](https://arxiv.org/pdf/2410.09704).