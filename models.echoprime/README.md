# EchoPrime: Multi-View Echocardiography AI

[![Paper](https://img.shields.io/badge/arXiv-2410.09704-b31b1b.svg)](https://arxiv.org/abs/2410.09704)
[![Demo](https://img.shields.io/badge/Demo-Video-blue.svg)](https://x.com/i/status/1846321746900558097)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art AI system that automatically analyzes echocardiogram videos to generate comprehensive clinical reports and predict cardiac metrics. Unlike traditional single-view models, EchoPrime intelligently combines information from multiple echocardiographic views to provide expert-level cardiac assessments.

![EchoPrime Demo](demo_image.png)

## 🎯 Key Features

- **🧠 Multi-View Intelligence**: Automatically identifies and combines information from multiple echocardiographic views
- **📋 Comprehensive Reports**: Generates detailed clinical reports covering 15 anatomical sections
- **🔍 Quality Assessment**: Built-in video quality evaluation and filtering
- **⚡ Expert-Level Accuracy**: Performance comparable to experienced cardiologists
- **🐳 Easy Deployment**: Docker support and simple Makefile commands

## 🚀 Quick Start

### Option 1: Simple Setup (Recommended)
```bash
# 1. Setup environment and download models
make init
make download-weights

# 2. Place your DICOM files in raw_data/
# 3. Run analysis
make inference

# 4. View results in results/inference_output/
```

### Option 2: Interactive Notebooks
```bash
# Setup and launch Jupyter
make init
make download-weights
make jupyter

# Then open EchoPrime-Demo.ipynb
```

### Option 3: Docker
```bash
# Build and run with Docker
make build-docker
make run-docker
```

## 📚 Documentation

For comprehensive documentation, see the [`docs/`](docs/) directory:

- **[🚀 Getting Started](docs/GETTING_STARTED.md)** - Complete setup and first run guide
- **[⚙️ System Architecture](docs/ARCHITECTURE.md)** - Technical overview of the multi-view system
- **[📖 Command Reference](docs/COMMAND_REFERENCE.md)** - All available commands and options
- **[📁 Project Structure](docs/STRUCTURE.md)** - Project organization and file layout
- **[🎓 Training Guide](docs/TRAINING.md)** - Model training and fine-tuning
- **[❗ Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[🔍 Model Weights Guide](docs/MODEL_WEIGHTS_EXPLAINED.md)** - Understanding AI components
- **[🔧 DICOM Processing](docs/README_DICOM_UNPACKER.md)** - DICOM file handling

## 🔬 Research Paper

**EchoPrime: A Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation**  
*Milos Vukadinovic, Xiu Tang, Neal Yuan, Paul Cheng, Debiao Li, Susan Cheng, Bryan He, David Ouyang*

```bibtex
@article{vukadinovic2024echoprime,
  title={EchoPrime: A Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation},
  author={Vukadinovic, Milos and Tang, Xiu and Yuan, Neal and Cheng, Paul and Li, Debiao and Cheng, Susan and He, Bryan and Ouyang, David},
  journal={arXiv preprint arXiv:2410.09704},
  year={2024}
}
```

## 🆘 Support

- **Issues**: Report bugs via GitHub Issues
- **Questions**: Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- **Documentation**: Browse the comprehensive [docs/](docs/) directory

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
