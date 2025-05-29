#!/usr/bin/env python
"""
Prepare Training Data for EchoQuality Model

This script identifies high and low quality DICOM files from inference results,
copies them to the training_data folder, and creates the annotations CSV.
"""

import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def clear_training_data_folder(training_data_dir):
    """Clear and recreate the training_data folder."""
    if training_data_dir.exists():
        print(f"Clearing existing training_data folder: {training_data_dir}")
        shutil.rmtree(training_data_dir)
    
    # Create fresh directory structure
    training_data_dir.mkdir(exist_ok=True)
    (training_data_dir / "echo_videos").mkdir(exist_ok=True)
    
    print(f"Created fresh training_data directory: {training_data_dir}")

def prepare_training_data(max_high_quality=800, max_low_quality=300, 
                         high_quality_threshold=0.95, low_quality_threshold=0.1):
    """
    Prepare training data by copying DICOM files and creating annotations.
    
    Args:
        max_high_quality: Maximum number of high quality samples
        max_low_quality: Maximum number of low quality samples  
        high_quality_threshold: Minimum score for high quality samples
        low_quality_threshold: Maximum score for low quality samples
    """
    
    print("Preparing Training Data for EchoQuality Model")
    print("=" * 50)
    
    # Set up directories
    training_data_dir = Path("training_data")
    echo_videos_dir = training_data_dir / "echo_videos"
    
    # Clear and recreate training_data folder
    clear_training_data_folder(training_data_dir)
    
    # Load inference results
    results_dir = Path("results/inference_output")
    summary_file = results_dir / "summary.json"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    print(f"Loading inference results from: {summary_file}")
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    # Extract samples based on quality criteria
    high_quality_samples = []
    low_quality_samples = []
    medium_quality_samples = []
    
    print("Analyzing inference results...")
    for folder_result in tqdm(data.get('folder_results', [])):
        folder_name = folder_result['folder']
        results = folder_result.get('results', {})
        
        for file_id, result in results.items():
            score = result.get('score', 0)
            status = result.get('status', 'FAIL')
            assessment = result.get('assessment', '')
            file_path = result.get('path', '')
            
            # Extract machine type from folder name
            machine_type = folder_name.split('-')[0] if '-' in folder_name else 'unknown'
            
            # Create sample metadata
            sample = {
                'file_id': file_id,
                'folder': folder_name,
                'machine_type': machine_type,
                'score': score,
                'status': status,
                'assessment': assessment,
                'original_dicom_path': file_path,  # Original DICOM file path
            }
            
            # Categorize based on quality
            if (score >= high_quality_threshold and 
                status == 'PASS' and 
                'Excellent quality' in assessment):
                sample['quality_label'] = 1
                high_quality_samples.append(sample)
                
            elif (score <= low_quality_threshold and 
                  status == 'FAIL' and 
                  'Critical issues' in assessment):
                sample['quality_label'] = 0
                low_quality_samples.append(sample)
                
            else:
                sample['quality_label'] = -1  # Medium quality, not used for training
                medium_quality_samples.append(sample)
    
    print(f"\nQuality Distribution:")
    print(f"  High quality samples (≥{high_quality_threshold}): {len(high_quality_samples)}")
    print(f"  Low quality samples (≤{low_quality_threshold}): {len(low_quality_samples)}")
    print(f"  Medium quality samples (excluded): {len(medium_quality_samples)}")
    
    # Limit samples if needed
    if len(high_quality_samples) > max_high_quality:
        print(f"Randomly selecting {max_high_quality} from {len(high_quality_samples)} high quality samples")
        high_quality_samples = np.random.choice(
            high_quality_samples, max_high_quality, replace=False
        ).tolist()
    
    if len(low_quality_samples) > max_low_quality:
        print(f"Randomly selecting {max_low_quality} from {len(low_quality_samples)} low quality samples")
        low_quality_samples = np.random.choice(
            low_quality_samples, max_low_quality, replace=False
        ).tolist()
    
    # Combine all training samples
    training_samples = high_quality_samples + low_quality_samples
    
    print(f"\nFinal Training Dataset:")
    print(f"  High quality samples: {len(high_quality_samples)}")
    print(f"  Low quality samples: {len(low_quality_samples)}")
    print(f"  Total training samples: {len(training_samples)}")
    
    # Copy DICOM files to training_data/echo_videos/
    print(f"\nCopying DICOM files to {echo_videos_dir}...")
    copied_samples = []
    missing_files = 0
    
    for i, sample in enumerate(tqdm(training_samples)):
        original_path = Path(sample['original_dicom_path'])
        
        if original_path.exists():
            # Create new filename with quality prefix and machine type
            quality_prefix = "hq" if sample['quality_label'] == 1 else "lq"
            new_filename = f"{quality_prefix}_{sample['machine_type']}_{i:04d}_{sample['file_id'][:8]}.dcm"
            new_path = echo_videos_dir / new_filename
            
            try:
                # Copy the DICOM file
                shutil.copy2(original_path, new_path)
                
                # Update sample with new path
                sample['video_path'] = f"echo_videos/{new_filename}"
                sample['filename'] = new_filename
                copied_samples.append(sample)
                
            except Exception as e:
                print(f"Error copying {original_path}: {e}")
                missing_files += 1
        else:
            print(f"Missing file: {original_path}")
            missing_files += 1
    
    print(f"Successfully copied {len(copied_samples)} DICOM files")
    print(f"Missing files: {missing_files}")
    
    # Create annotations CSV
    print("\nCreating echo_annotations.csv...")
    df = pd.DataFrame(copied_samples)
    
    # Add training-specific columns
    df['quality_score'] = df['quality_label']  # Binary label for training
    
    # Reorder columns for training (match expected format)
    training_columns = [
        'video_path',           # Path to DICOM file (relative to training_data/)
        'quality_score',        # Binary quality label (0/1)
        'filename',             # DICOM filename
        'machine_type',         # Ultrasound machine type
        'score',               # Original inference score
        'status',              # Original inference status
        'assessment',          # Original quality assessment
        'folder',              # Original folder name
        'file_id'              # Original file identifier
    ]
    
    df_training = df[training_columns].copy()
    
    # Save annotations CSV to training_data folder
    csv_path = training_data_dir / "echo_annotations.csv"
    df_training.to_csv(csv_path, index=False)
    
    print(f"Training annotations saved to: {csv_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total samples: {len(df_training)}")
    print(f"High quality (label=1): {len(df_training[df_training['quality_score'] == 1])}")
    print(f"Low quality (label=0): {len(df_training[df_training['quality_score'] == 0])}")
    
    print("\nMachine type distribution:")
    machine_counts = df_training['machine_type'].value_counts()
    for machine_type, count in machine_counts.items():
        print(f"  {machine_type}: {count} samples")
    
    print("\nScore distribution:")
    high_quality_scores = df_training[df_training['quality_score'] == 1]['score']
    low_quality_scores = df_training[df_training['quality_score'] == 0]['score']
    
    if len(high_quality_scores) > 0:
        print(f"  High quality scores: min={high_quality_scores.min():.3f}, max={high_quality_scores.max():.3f}, mean={high_quality_scores.mean():.3f}")
    if len(low_quality_scores) > 0:
        print(f"  Low quality scores: min={low_quality_scores.min():.3f}, max={low_quality_scores.max():.3f}, mean={low_quality_scores.mean():.3f}")
    
    # Save detailed analysis
    analysis_path = training_data_dir / "dataset_analysis.json"
    analysis = {
        'total_samples': len(df_training),
        'high_quality_count': len(df_training[df_training['quality_score'] == 1]),
        'low_quality_count': len(df_training[df_training['quality_score'] == 0]),
        'machine_type_distribution': machine_counts.to_dict(),
        'high_quality_threshold': high_quality_threshold,
        'low_quality_threshold': low_quality_threshold,
        'missing_files': missing_files,
        'files_copied': len(copied_samples)
    }
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Dataset analysis saved to: {analysis_path}")
    
    print("\n" + "=" * 50)
    print("Training data preparation complete!")
    print(f"Training data folder: {training_data_dir}")
    print(f"DICOM files copied to: {echo_videos_dir}")
    print(f"Annotations file: {csv_path}")
    print("\nNext steps:")
    print("1. Review the dataset statistics above")
    print("2. Run training with: make train")
    print("3. Monitor training progress with MLflow")
    
    return str(csv_path)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare training data for EchoQuality model")
    parser.add_argument("--max-high-quality", type=int, default=800,
                       help="Maximum number of high quality samples")
    parser.add_argument("--max-low-quality", type=int, default=300,
                       help="Maximum number of low quality samples")
    parser.add_argument("--high-quality-threshold", type=float, default=0.95,
                       help="Minimum score for high quality samples")
    parser.add_argument("--low-quality-threshold", type=float, default=0.1,
                       help="Maximum score for low quality samples")
    
    args = parser.parse_args()
    
    csv_path = prepare_training_data(
        max_high_quality=args.max_high_quality,
        max_low_quality=args.max_low_quality,
        high_quality_threshold=args.high_quality_threshold,
        low_quality_threshold=args.low_quality_threshold
    )
    
    print(f"\nTraining data ready!")
    print(f"CSV file: {csv_path}")

if __name__ == "__main__":
    main()
