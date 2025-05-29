# GHL

## 🏗️ Project Structure

```
ghl/
├── models.echoprime/      # EchoPrime foundation model
├── models.echoquality/    # EchoPrime quality assessment model
├── models.hubertecg/      # HuBERT-ECG foundation model
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

### Optional Dependencies

- **CUDA Toolkit** (for GPU acceleration): CUDA 11.8+ recommended

## 📚 Documentation

- [Project Overview](docs/overview.md)
- [Pipeline Documentation](docs/pipeline.md)
