# ❗ EchoPrime Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using EchoPrime for echocardiography analysis.

## 🚨 Quick Diagnosis

### Check System Status
```bash
# Verify Poetry environment
poetry --version
poetry show

# Check Python version
python --version

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify model weights
ls -la weights/
```

### Common Error Patterns

| Error Pattern | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| `ModuleNotFoundError` | Missing dependencies | `make init` |
| `FileNotFoundError: weights/` | Missing model weights | `make download-weights` |
| `CUDA out of memory` | GPU memory insufficient | Use `--device cpu` |
| `No DICOM files found` | Incorrect directory structure | Check `raw_data/` organization |
| `Green-tinted images` | Dependency version mismatch | Reinstall with Poetry |

---

## 🛠️ Installation Issues

### Poetry Installation Problems

#### Issue: "Poetry not found"
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

#### Issue: "Python version not supported"
```bash
# Check current Python version
python --version

# Install compatible Python version (3.8-3.12)
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.10 python3.10-venv

# Tell Poetry to use specific Python version
poetry env use python3.10
```

#### Issue: "Poetry install fails with dependency conflicts"
```bash
# Clear Poetry cache
poetry cache clear pypi --all

# Remove existing environment
poetry env remove python

# Reinstall
poetry install
```

---

### Model Weight Download Issues

#### Issue: "wget command not found"
```bash
# On Ubuntu/Debian:
sudo apt install wget

# On macOS:
brew install wget

# Alternative: Use curl
curl -L -o weights/echo_prime_encoder.pt https://github.com/patniko/ghl/releases/download/v1.0.0/echo_prime_encoder.pt
```

#### Issue: "Download interrupted or corrupted"
```bash
# Remove partial downloads
rm -rf weights/
mkdir weights

# Re-download
make download-weights

# Verify file integrity
ls -lh weights/
# echo_prime_encoder.pt should be ~138MB
# view_classifier.ckpt should be ~350MB
```

#### Issue: "Insufficient disk space"
```bash
# Check available space
df -h .

# Clean up space
make clean

# Download to different location and symlink
mkdir /path/to/large/storage/echoprime_weights
ln -s /path/to/large/storage/echoprime_weights weights
make download-weights
```

---

## 🖥️ Runtime Issues

### GPU and CUDA Problems

#### Issue: "CUDA out of memory"
**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```bash
# Solution 1: Use CPU processing
poetry run python -m inference.inference --device cpu --data_dir ./raw_data --output ./results

# Solution 2: Reduce batch size
poetry run python -m inference.inference --batch_size 1 --data_dir ./raw_data --output ./results

# Solution 3: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Solution 4: Use memory-efficient mode
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
make inference
```

#### Issue: "CUDA driver version mismatch"
```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
# For CUDA 11.8:
poetry add torch torchvision --source https://download.pytorch.org/whl/cu118
```

#### Issue: "No GPU detected"
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
nvcc --version

# Test PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

---

### Memory Issues

#### Issue: "System runs out of RAM"
**Symptoms:**
- System becomes unresponsive
- Process killed by OS
- "MemoryError" in Python

**Solutions:**
```bash
# Monitor memory usage
htop  # or top

# Reduce number of workers
poetry run python -m inference.inference --num_workers 1 --data_dir ./raw_data --output ./results

# Process studies one at a time
for study in raw_data/*/; do
    poetry run python -m inference.inference --data_dir "$study" --output "./results/$(basename "$study")"
done

# Enable swap if needed (Linux)
sudo swapon --show
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 📁 Data Issues

### DICOM File Problems

#### Issue: "No DICOM files found"
**Check directory structure:**
```bash
# Verify structure
tree raw_data/  # or ls -la raw_data/

# Correct structure should be:
# raw_data/
# ├── study_1/
# │   ├── file1.dcm
# │   └── file2.dcm
# └── study_2/
#     └── file3.dcm
```

**Solutions:**
```bash
# Move files to correct structure
mkdir -p raw_data/study_001
mv *.dcm raw_data/study_001/

# Check file extensions
find raw_data/ -name "*.dcm" -type f
find raw_data/ -name "*.DCM" -type f  # Check uppercase

# Rename if needed
find raw_data/ -name "*.DCM" -exec rename 's/\.DCM$/.dcm/' {} \;
```

#### Issue: "Corrupted DICOM files"
**Symptoms:**
- "Invalid DICOM file" errors
- Processing stops unexpectedly

**Diagnosis:**
```bash
# Test DICOM files with pydicom
python -c "
import pydicom
import sys
try:
    ds = pydicom.dcmread(sys.argv[1])
    print(f'Valid DICOM: {ds.PatientID if hasattr(ds, \"PatientID\") else \"Unknown\"}')
except Exception as e:
    print(f'Invalid DICOM: {e}')
" path/to/file.dcm
```

**Solutions:**
```bash
# Find and remove corrupted files
find raw_data/ -name "*.dcm" -exec python -c "
import pydicom
import sys
import os
try:
    pydicom.dcmread(sys.argv[1])
except:
    print(f'Removing corrupted file: {sys.argv[1]}')
    os.remove(sys.argv[1])
" {} \;

# Validate all DICOM files
poetry run python scripts/validate_dicom_files.py --input_dir raw_data/
```

#### Issue: "DICOM files don't contain video data"
**Symptoms:**
- "No video frames found" errors
- Empty preprocessed output

**Diagnosis:**
```bash
# Check DICOM metadata
python -c "
import pydicom
ds = pydicom.dcmread('path/to/file.dcm')
print(f'Modality: {ds.get(\"Modality\", \"Unknown\")}')
print(f'SOP Class: {ds.get(\"SOPClassUID\", \"Unknown\")}')
print(f'Has pixel data: {hasattr(ds, \"pixel_array\")}')
"
```

---

### Processing Quality Issues

#### Issue: "All videos marked as poor quality"
**Symptoms:**
- Quality scores consistently below 0.3
- No usable results generated

**Diagnosis:**
```bash
# Check quality scores
cat results/inference_output/study_name/quality_scores.json

# Review preprocessing output
ls -la preprocessed_data/study_name/
```

**Solutions:**
```bash
# Lower quality threshold
poetry run python -m inference.inference --min_quality 0.1 --data_dir ./raw_data --output ./results

# Debug preprocessing
poetry run python scripts/debug_preprocessing.py --input raw_data/study_name/file.dcm

# Check for green-tinted images (dependency issue)
# If images appear green, reinstall dependencies:
make init
```

#### Issue: "View classification errors"
**Symptoms:**
- All views classified as "Other" or "Poor Quality"
- Inconsistent view predictions

**Diagnosis:**
```bash
# Check view classifications
cat results/inference_output/study_name/view_classifications.json

# Manually inspect first frames
poetry run python scripts/extract_first_frames.py --input raw_data/study_name/
```

**Solutions:**
```bash
# Re-download view classifier weights
rm weights/view_classifier.ckpt
make download-weights

# Check for preprocessing issues
poetry run python scripts/debug_view_classification.py --input raw_data/study_name/
```

---

## 🔧 Performance Issues

### Slow Processing

#### Issue: "Processing takes too long"
**Diagnosis:**
```bash
# Monitor resource usage
htop
nvidia-smi  # For GPU usage

# Time individual components
poetry run python -m inference.inference --profile --data_dir ./raw_data --output ./results
```

**Solutions:**
```bash
# Use GPU if available
poetry run python -m inference.inference --device cuda

# Increase batch size (if memory allows)
poetry run python -m inference.inference --batch_size 8

# Use more CPU workers
poetry run python -m inference.inference --num_workers 8

# Process in parallel
# Split studies across multiple processes
```

#### Issue: "High memory usage"
**Solutions:**
```bash
# Monitor memory usage
watch -n 1 'free -h'

# Reduce batch size
poetry run python -m inference.inference --batch_size 1

# Process studies sequentially
for study in raw_data/*/; do
    poetry run python -m inference.inference --data_dir "$study" --output "./results/$(basename "$study")"
    # Clear cache between studies
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
done
```

---

## 🐛 Output Issues

### Missing or Incomplete Results

#### Issue: "No clinical reports generated"
**Check:**
```bash
# Verify output directory structure
ls -la results/inference_output/

# Check for error logs
find results/ -name "*.log" -exec cat {} \;

# Verify candidate data
ls -la weights/candidates_data/
```

**Solutions:**
```bash
# Re-download candidate data
rm -rf weights/candidates_data/
make download-weights

# Check processing logs
poetry run python -m inference.inference --verbose --data_dir ./raw_data --output ./results
```

#### Issue: "Inconsistent metric predictions"
**Diagnosis:**
```bash
# Compare predictions across similar studies
cat results/inference_output/*/metrics_predictions.json

# Check quality scores
grep -r "overall_study_quality" results/inference_output/
```

**Solutions:**
```bash
# Ensure sufficient view coverage
# Check that studies include multiple standard views

# Verify model weights integrity
md5sum weights/echo_prime_encoder.pt
# Should match expected checksum

# Re-run with debug mode
poetry run python -m inference.inference --debug --data_dir ./raw_data --output ./results
```

---

## 🔍 Debug Tools

### Built-in Debugging

#### Enable Verbose Logging
```bash
# Set environment variable
export ECHOPRIME_LOG_LEVEL=DEBUG

# Or use command line flag
poetry run python -m inference.inference --verbose --debug
```

#### Profile Performance
```bash
# Enable profiling
poetry run python -m inference.inference --profile --data_dir ./raw_data --output ./results

# Use Python profiler
python -m cProfile -o profile_output.prof -m inference.inference --data_dir ./raw_data --output ./results

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(20)
"
```

### Custom Debug Scripts

#### Test Single DICOM File
```python
# Create debug_single_file.py
import sys
from inference.inference import EchoPrimeInference

def debug_single_file(dicom_path):
    model = EchoPrimeInference(device='cuda')
    
    try:
        # Test preprocessing
        processed = model.preprocess_dicom(dicom_path)
        print(f"Preprocessing successful: {processed.shape}")
        
        # Test view classification
        view = model.classify_view(processed)
        print(f"View classification: {view}")
        
        # Test feature extraction
        features = model.extract_features(processed)
        print(f"Feature extraction: {features.shape}")
        
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_file(sys.argv[1])
```

```bash
# Use the debug script
poetry run python debug_single_file.py raw_data/study_001/file.dcm
```

---

## 🆘 Getting Help

### Log Collection
When reporting issues, collect these logs:

```bash
# System information
echo "=== System Info ===" > debug_info.txt
uname -a >> debug_info.txt
python --version >> debug_info.txt
poetry --version >> debug_info.txt

# GPU information
echo "=== GPU Info ===" >> debug_info.txt
nvidia-smi >> debug_info.txt 2>&1

# Python environment
echo "=== Python Environment ===" >> debug_info.txt
poetry show >> debug_info.txt

# EchoPrime version and weights
echo "=== EchoPrime Info ===" >> debug_info.txt
ls -la weights/ >> debug_info.txt

# Recent error logs
echo "=== Recent Errors ===" >> debug_info.txt
find . -name "*.log" -mtime -1 -exec cat {} \; >> debug_info.txt
```

### Common Solutions Summary

| Problem Category | First Try | If That Fails |
|------------------|-----------|---------------|
| **Installation** | `make init` | Clear cache, reinstall Poetry |
| **Missing Weights** | `make download-weights` | Manual download with curl |
| **GPU Issues** | `--device cpu` | Update CUDA drivers |
| **Memory Issues** | `--batch_size 1` | Use CPU, add swap |
| **No DICOM Found** | Check directory structure | Verify file extensions |
| **Poor Quality** | Lower `--min_quality` | Debug preprocessing |
| **Slow Performance** | Use GPU | Increase batch size |
| **Missing Results** | Check logs | Re-download candidate data |

### Emergency Recovery

If EchoPrime is completely broken:

```bash
# Nuclear option: Complete reinstall
rm -rf .venv/
rm poetry.lock
rm -rf weights/
rm -rf results/
rm -rf preprocessed_data/

# Start fresh
make init
make download-weights
make inference
```

---

## 📞 Support Channels

- **Documentation**: Check other guides in `docs/`
- **GitHub Issues**: Report bugs and feature requests
- **Community**: Join discussions for usage questions
- **Email**: Contact maintainers for critical issues

Remember to include your `debug_info.txt` when seeking help!
