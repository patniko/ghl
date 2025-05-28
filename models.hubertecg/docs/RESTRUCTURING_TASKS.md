# HuBERT-ECG Restructuring Task List

## Overview
This document outlines the complete restructuring plan to make models.hubertecg consistent with the clean organization of models.echoprime.

## Phase 1: Create New Directory Structure ✅

### 1.1 Create Core Directories
- [x] Create `docs/` directory
- [ ] Create `data/` directory with `__init__.py`
- [ ] Create `inference/` directory with `__init__.py`
- [ ] Create `preprocessors/` directory with `__init__.py`
- [ ] Create `results/` directory with `__init__.py`
- [ ] Create `tools/` directory with `__init__.py`
- [ ] Create `training/` directory with `__init__.py`
- [ ] Create `training/models/` subdirectory with `__init__.py`
- [ ] Create `weights/` directory with `__init__.py`

## Phase 2: Move and Refactor Core Components

### 2.1 Move Model Files
- [ ] Move `code/hubert_ecg.py` → `training/models/hubert_ecg.py`
- [ ] Move `code/hubert_ecg_classification.py` → `training/models/hubert_ecg_classification.py`

### 2.2 Move Training Files
- [ ] Move `code/pretrain.py` → `training/pretrain.py`
- [ ] Move `code/finetune.py` → `training/finetune.py`
- [ ] Move `code/dataset.py` → `training/dataset.py`
- [ ] Move `code/utils.py` → `training/utils.py`

### 2.3 Move Tool Files
- [ ] Move `code/cluster.py` → `tools/clustering.py`
- [ ] Move `code/dumping.py` → `tools/feature_extraction.py`
- [ ] Move `code/metrics.py` → `tools/evaluation.py`
- [ ] Move `code/test.py` → `tools/test.py`

### 2.4 Move Inference Files
- [ ] Move `HuBERT-ECG-Demo.ipynb` → `inference/HuBERT-ECG-Demo.ipynb`
- [ ] Move `EchoPrimeDemo.ipynb` → `inference/EchoPrimeDemo.ipynb`

### 2.5 Move Results and Figures
- [ ] Move `figures/` → `results/figures/`

### 2.6 Create Preprocessors
- [ ] Extract preprocessing functions from existing code into `preprocessors/ecg_preprocessing.py`

## Phase 3: Update Documentation

### 3.1 Create New Documentation Files
- [ ] Create `docs/STRUCTURE.md` - Document the new organization
- [ ] Create `docs/TRAINING.md` - Training documentation
- [ ] Move `reproducibility/README.md` → `docs/REPRODUCIBILITY.md`
- [ ] Create `docs/__init__.py`

### 3.2 Update Existing Documentation
- [ ] Update main `README.md` to reflect new structure
- [ ] Update `code/README.md` references

## Phase 4: Update Import Statements and Scripts

### 4.1 Update Python Import Statements
- [ ] Update imports in moved training files
- [ ] Update imports in moved tool files
- [ ] Update imports in moved model files
- [ ] Update imports in Jupyter notebooks
- [ ] Update imports in scripts

### 4.2 Update Shell Scripts
- [ ] Update `code/finetune.sh` to use new paths
- [ ] Update `code/test.sh` to use new paths
- [ ] Move shell scripts to appropriate locations

### 4.3 Update Configuration Files
- [ ] Update Makefile commands to work with new structure
- [ ] Update any configuration files with new paths

## Phase 5: Clean Up and Validation

### 5.1 Remove Old Structure
- [ ] Remove empty `code/` directory after moving all files
- [ ] Clean up any remaining obsolete files

### 5.2 Validation
- [ ] Test that all imports work correctly
- [ ] Test that shell scripts execute properly
- [ ] Test that Jupyter notebooks run without errors
- [ ] Verify Makefile commands work
- [ ] Test training and inference workflows

## Phase 6: Final Documentation Updates

### 6.1 Complete Documentation
- [ ] Finalize `docs/STRUCTURE.md` with complete structure
- [ ] Update all README files with new structure references
- [ ] Create migration guide for existing users

## Success Criteria
- [ ] All functionality works with new structure
- [ ] Clear separation of concerns (inference, training, preprocessing, tools)
- [ ] Consistent with models.echoprime organization
- [ ] Comprehensive documentation of new structure
- [ ] All existing workflows continue to function
- [ ] Easy to navigate and understand for new users

## Notes
- Maintain backward compatibility where possible
- Update all cross-references and documentation
- Ensure all `__init__.py` files are properly created
- Test thoroughly after each major phase
