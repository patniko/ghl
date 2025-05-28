# GHL (GoHealth Labs) - Machine Learning Platform

A comprehensive machine learning platform for healthcare applications, featuring multiple specialized models for ECG analysis, echocardiogram processing, and medical data analysis.

## 🏗️ Project Structure

```
ghl/
├── hub/                    # Core platform services
│   ├── ph-api/            # Python API backend
│   └── ph-ui/             # React frontend
├── models.dino/           # DINO-based ECG foundation model
├── models.echo/           # Echo analysis models
├── models.echoprime/      # EchoPrime research model
├── models.echoquality/    # Echo quality assessment
├── models.hubertecg/      # HuBERT-ECG foundation model
├── mlflow/                # MLflow experiment tracking
├── tools/                 # Shared utilities and tools
├── samples/               # Sample data and examples
└── docs/                  # Documentation
```

## 🔧 System Requirements

### Minimum Dependencies

Before setting up this project, ensure you have the following installed on your system:

#### 1. Python 3.11+
```bash
# Check your Python version
python3 --version

# Should output Python 3.11.x or higher
```

**Installation:**
- **Ubuntu/Debian:** `sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev`
- **macOS:** `brew install python@3.11`
- **Windows:** Download from [python.org](https://www.python.org/downloads/)

#### 2. Poetry 2.1+
```bash
# Check your Poetry version
poetry --version

# Should output Poetry (version 2.1.x) or higher
```

**Installation:**
```bash
# Install Poetry using the official installer
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to your PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="/home/$USER/.local/bin:$PATH"

# Reload your shell or run:
source ~/.bashrc
```

#### 3. Docker & Docker Compose
```bash
# Check Docker installation
docker --version
docker-compose --version
```

**Installation:**
- **Ubuntu/Debian:**
  ```bash
  sudo apt update
  sudo apt install docker.io docker-compose
  sudo usermod -aG docker $USER
  # Log out and back in for group changes to take effect
  ```
- **macOS:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Windows:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

### Optional Dependencies

- **CUDA Toolkit** (for GPU acceleration): CUDA 11.8+ recommended
- **Git LFS** (for large model files): `git lfs install`

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ghl
```

### 2. Set Up Individual Components

Each component has its own setup process. Navigate to the specific directory and run:

```bash
# For Python-based components (models, API)
cd models.hubertecg  # or any other model directory
make setup

# For the React frontend
cd hub/ph-ui
npm install
```

### 3. Start Core Services

#### API Backend
```bash
cd hub/ph-api
make setup
make run
```

#### Frontend
```bash
cd hub/ph-ui
npm run dev
```

#### MLflow Tracking
```bash
cd mlflow
make setup
make start
```

## 📋 Component-Specific Setup

### Models
Each model directory contains its own `Makefile` with standardized commands:
- `make setup` - Install dependencies and set up environment
- `make train` - Train the model (where applicable)
- `make test` - Run tests
- `make clean` - Clean up generated files

### Hub Services
- **API (`hub/ph-api/`)**: FastAPI-based backend with PostgreSQL
- **UI (`hub/ph-ui/`)**: React-based frontend with TypeScript

### MLflow
Experiment tracking and model registry service accessible at `http://localhost:5000`

## 🔍 Verification

After setup, verify your installation:

```bash
# Check Python version
python3 --version

# Check Poetry version
poetry --version

# Check Docker
docker run hello-world

# Test a model setup (example)
cd models.hubertecg
make setup
```

## 🐛 Troubleshooting

### Common Issues

1. **Poetry Configuration Error**
   - Ensure you're using Poetry 2.1+
   - Older versions don't support `package-mode` and dependency groups

2. **Docker Permission Denied**
   - Add your user to the docker group: `sudo usermod -aG docker $USER`
   - Log out and back in

3. **Python Version Conflicts**
   - Use `pyenv` to manage multiple Python versions
   - Ensure Poetry is using the correct Python version

4. **CUDA/GPU Issues**
   - Install appropriate CUDA toolkit for your GPU
   - Verify with `nvidia-smi`

## 📚 Documentation

- [Project Overview](docs/overview.md)
- [Pipeline Documentation](docs/pipeline.md)
- [TODO and Roadmap](docs/todo.md)

## 🤝 Contributing

1. Ensure all system requirements are met
2. Set up pre-commit hooks: `pre-commit install`
3. Follow the coding standards defined in each component
4. Run tests before submitting PRs

## 📄 License

This project contains multiple components with different licenses. Check individual directories for specific license information.

---

**Need Help?** Check the component-specific README files or open an issue for support.
