#!/usr/bin/env python3
"""
Data Preparation Script for EchoPrime Fine-tuning

This script helps prepare your echocardiogram data for training with the
EchoPrime fine-tuning script.
"""

import os
import json
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EchoDataPreprocessor:
    """Data preprocessor for echocardiogram videos and reports"""
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 test_size: float = 0.2,
                 val_size: float = 0.1):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.val_size = val_size
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        
    def prepare_from_dicom(self, 
                          dicom_dir: str,
                          reports_file: str,
                          view_labels_file: str = None):
        """
        Prepare data from DICOM files and reports
        
        Args:
            dicom_dir: Directory containing DICOM files
            reports_file: CSV file with columns ['study_id', 'report_text']
            view_labels_file: Optional CSV with ['study_id', 'view_label']
        """
        logger.info("Converting DICOM files to videos...")
        
        # Load reports
        reports_df = pd.read_csv(reports_file)
        
        # Load view labels if available
        view_labels_df = None
        if view_labels_file and os.path.exists(view_labels_file):
            view_labels_df = pd.read_csv(view_labels_file)
        
        # Process DICOM files
        video_metadata = []
        dicom_files = list(Path(dicom_dir).glob("**/*.dcm"))
        
        for dicom_path in dicom_files:
            try:
                video_id = self._convert_dicom_to_video(dicom_path)
                if video_id:
                    metadata = self._extract_metadata(dicom_path, video_id, reports_df, view_labels_df)
                    if metadata:
                        video_metadata.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to process {dicom_path}: {e}")
        
        # Create train/val/test splits
        self._create_splits(video_metadata)
        
    def prepare_from_videos(self, 
                           videos_dir: str,
                           metadata_file: str):
        """
        Prepare data from existing video files
        
        Args:
            videos_dir: Directory containing video files (.avi, .mp4)
            metadata_file: CSV with columns ['video_id', 'report_text', 'view_label', 'labels']
        """
        logger.info("Processing existing video files...")
        
        # Load metadata
        metadata_df = pd.read_csv(metadata_file)
        
        # Copy and process videos
        videos_path = Path(videos_dir)
        video_metadata = []
        
        for _, row in metadata_df.iterrows():
            video_file = videos_path / f"{row['video_id']}.avi"
            if not video_file.exists():
                video_file = videos_path / f"{row['video_id']}.mp4"
            
            if video_file.exists():
                # Copy video to output directory
                output_video_path = self.output_dir / "videos" / f"{row['video_id']}.avi"
                self._process_video(video_file, output_video_path)
                
                # Prepare metadata
                metadata = {
                    'video_id': row['video_id'],
                    'report_text': row['report_text'],
                    'view_label': row.get('view_label', -1),
                    'labels': row.get('labels', '{}')
                }
                video_metadata.append(metadata)
            else:
                logger.warning(f"Video file not found: {row['video_id']}")
        
        # Create train/val/test splits
        self._create_splits(video_metadata)
    
    def _convert_dicom_to_video(self, dicom_path: Path) -> str:
        """Convert DICOM file to video format"""
        try:
            import pydicom
            from pydicom.pixel_data_handlers import convert_color_space
            
            # Read DICOM
            ds = pydicom.dcmread(str(dicom_path))
            
            # Extract video ID from filename or DICOM tags
            video_id = dicom_path.stem
            
            # Check if it's a multi-frame DICOM
            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                # Extract frames
                frames = ds.pixel_array
                
                # Convert to standard format if needed
                if len(frames.shape) == 4:  # Color video
                    frames = frames.astype(np.uint8)
                elif len(frames.shape) == 3:  # Grayscale video
                    frames = np.stack([frames] * 3, axis=-1).astype(np.uint8)
                
                # Save as AVI
                output_path = self.output_dir / "videos" / f"{video_id}.avi"
                self._save_video(frames, output_path)
                
                return video_id
            else:
                logger.warning(f"DICOM {dicom_path} is not a video")
                return None
                
        except Exception as e:
            logger.error(f"Error converting DICOM {dicom_path}: {e}")
            return None
    
    def _save_video(self, frames: np.ndarray, output_path: Path):
        """Save frames as video file"""
        height, width = frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
    
    def _process_video(self, input_path: Path, output_path: Path):
        """Process and standardize video"""
        cap = cv2.VideoCapture(str(input_path))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
    
    def _extract_metadata(self, 
                         dicom_path: Path, 
                         video_id: str, 
                         reports_df: pd.DataFrame,
                         view_labels_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Extract metadata for a video"""
        try:
            import pydicom
            ds = pydicom.dcmread(str(dicom_path))
            
            # Extract study ID for matching with reports
            study_id = getattr(ds, 'StudyInstanceUID', video_id)
            
            # Find matching report
            report_match = reports_df[reports_df['study_id'] == study_id]
            if report_match.empty:
                logger.warning(f"No report found for study {study_id}")
                return None
            
            report_text = report_match.iloc[0]['report_text']
            
            # Find view label if available
            view_label = -1
            if view_labels_df is not None:
                view_match = view_labels_df[view_labels_df['study_id'] == study_id]
                if not view_match.empty:
                    view_label = view_match.iloc[0]['view_label']
            
            # Extract additional labels from DICOM tags
            labels = {}
            if hasattr(ds, 'PatientAge'):
                labels['age'] = ds.PatientAge
            if hasattr(ds, 'PatientSex'):
                labels['sex'] = ds.PatientSex
            
            return {
                'video_id': video_id,
                'report_text': report_text,
                'view_label': view_label,
                'labels': json.dumps(labels)
            }
            
        except Exception as e:
            logger.error(f"Error extracting metadata for {dicom_path}: {e}")
            return None
    
    def _create_splits(self, video_metadata: List[Dict[str, Any]]):
        """Create train/validation/test splits"""
        df = pd.DataFrame(video_metadata)
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            random_state=42,
            stratify=df['view_label'] if 'view_label' in df.columns else None
        )
        
        # Second split: separate train and validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.val_size / (1 - self.test_size),
            random_state=42,
            stratify=train_val_df['view_label'] if 'view_label' in train_val_df.columns else None
        )
        
        # Save splits
        train_df.to_csv(self.output_dir / "train_reports.csv", index=False)
        val_df.to_csv(self.output_dir / "val_reports.csv", index=False)
        test_df.to_csv(self.output_dir / "test_reports.csv", index=False)
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
    
    def create_sample_data(self, num_samples: int = 100):
        """Create sample data for testing"""
        logger.info(f"Creating {num_samples} sample data entries...")
        
        # Sample view labels (58 standard views)
        view_labels = [
            "PLAX", "PSAX_AV", "PSAX_MV", "PSAX_PM", "PSAX_APEX",
            "A4C", "A2C", "A3C", "A5C", "SUBCOSTAL_4C", "SUBCOSTAL_IVC",
            "SUPRASTERNAL", "PARASTERNAL_RV_INFLOW", "PARASTERNAL_RV_OUTFLOW"
        ] + [f"VIEW_{i}" for i in range(44)]  # Additional views
        
        # Sample report templates
        report_templates = [
            "The left ventricle is normal in size with normal systolic function. LVEF is estimated at {ef}%. No wall motion abnormalities. The right ventricle is normal in size and function.",
            "Mild left ventricular hypertrophy. Left ventricular systolic function is {function}. LVEF approximately {ef}%. The aortic valve is {av_desc}.",
            "The left atrium is {la_size}. Mitral valve shows {mv_desc}. No significant mitral regurgitation. Tricuspid valve is normal.",
            "Normal cardiac chambers. No pericardial effusion. The aorta is normal in caliber. Pulmonary artery pressure is {pa_pressure} mmHg.",
            "Echocardiogram shows {overall_desc}. Left ventricular ejection fraction is {ef}%. {additional_findings}"
        ]
        
        sample_data = []
        for i in range(num_samples):
            video_id = f"sample_video_{i:04d}"
            
            # Generate random values
            ef = np.random.randint(35, 70)
            view_label = np.random.randint(0, len(view_labels))
            
            # Generate report
            report_template = np.random.choice(report_templates)
            report = report_template.format(
                ef=ef,
                function=np.random.choice(["normal", "mildly reduced", "moderately reduced"]),
                av_desc=np.random.choice(["normal", "mildly thickened", "calcified"]),
                la_size=np.random.choice(["normal", "mildly dilated", "moderately dilated"]),
                mv_desc=np.random.choice(["normal", "mild thickening", "prolapse"]),
                pa_pressure=np.random.randint(25, 45),
                overall_desc=np.random.choice(["normal study", "mild abnormalities", "moderate dysfunction"]),
                additional_findings=np.random.choice(["No other abnormalities.", "Trace tricuspid regurgitation.", "Normal coronary flow."])
            )
            
            # Generate sample labels
            labels = {
                "ejection_fraction": ef,
                "lv_function": np.random.choice([0, 1, 2]),  # normal, mild, moderate dysfunction
                "chamber_size": np.random.choice([0, 1, 2]),  # normal, mild, moderate enlargement
            }
            
            sample_data.append({
                'video_id': video_id,
                'report_text': report,
                'view_label': view_label,
                'labels': json.dumps(labels)
            })
            
            # Create dummy video file
            self._create_dummy_video(video_id)
        
        # Create splits
        self._create_splits(sample_data)
        
        logger.info("Sample data created successfully!")
    
    def _create_dummy_video(self, video_id: str):
        """Create a dummy video file for testing"""
        output_path = self.output_dir / "videos" / f"{video_id}.avi"
        
        # Create random frames
        frames = np.random.randint(0, 255, (30, 224, 224, 3), dtype=np.uint8)
        
        # Add some structure to make it look more like an echocardiogram
        center_y, center_x = 112, 112
        for i, frame in enumerate(frames):
            # Add circular pattern (simulating heart chambers)
            y, x = np.ogrid[:224, :224]
            mask = (x - center_x)**2 + (y - center_y)**2 < (50 + 10 * np.sin(i * 0.2))**2
            frame[mask] = [255, 255, 255]
        
        self._save_video(frames, output_path)
    
    def validate_data(self):
        """Validate the prepared data"""
        logger.info("Validating prepared data...")
        
        errors = []
        
        # Check if required files exist
        required_files = ["train_reports.csv", "val_reports.csv", "test_reports.csv"]
        for file in required_files:
            if not (self.output_dir / file).exists():
                errors.append(f"Missing required file: {file}")
        
        # Check video files
        videos_dir = self.output_dir / "videos"
        if not videos_dir.exists():
            errors.append("Videos directory does not exist")
        else:
            for split_file in required_files:
                if (self.output_dir / split_file).exists():
                    df = pd.read_csv(self.output_dir / split_file)
                    missing_videos = []
                    for video_id in df['video_id']:
                        video_path = videos_dir / f"{video_id}.avi"
                        if not video_path.exists():
                            missing_videos.append(video_id)
                    if missing_videos:
                        errors.append(f"Missing videos in {split_file}: {missing_videos[:5]}...")
        
        # Check data format
        for split_file in required_files:
            if (self.output_dir / split_file).exists():
                df = pd.read_csv(self.output_dir / split_file)
                required_columns = ['video_id', 'report_text', 'view_label', 'labels']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    errors.append(f"Missing columns in {split_file}: {missing_columns}")
        
        if errors:
            logger.error("Data validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        else:
            logger.info("Data validation passed!")
            return True


def main():
    parser = argparse.ArgumentParser(description="EchoPrime Data Preparation Script")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing raw data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--data_type", type=str, choices=["dicom", "videos", "sample"],
                       required=True, help="Type of input data")
    
    # For DICOM processing
    parser.add_argument("--dicom_dir", type=str,
                       help="Directory containing DICOM files")
    parser.add_argument("--reports_file", type=str,
                       help="CSV file with reports")
    parser.add_argument("--view_labels_file", type=str,
                       help="CSV file with view labels (optional)")
    
    # For video processing
    parser.add_argument("--videos_dir", type=str,
                       help="Directory containing video files")
    parser.add_argument("--metadata_file", type=str,
                       help="CSV file with video metadata")
    
    # For sample data
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of sample data entries to create")
    
    # Data split parameters
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Proportion of data for testing")
    parser.add_argument("--val_size", type=float, default=0.1,
                       help="Proportion of data for validation")
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = EchoDataPreprocessor(
        args.input_dir,
        args.output_dir,
        args.test_size,
        args.val_size
    )
    
    # Process data based on type
    if args.data_type == "dicom":
        if not args.dicom_dir or not args.reports_file:
            logger.error("DICOM processing requires --dicom_dir and --reports_file")
            return
        preprocessor.prepare_from_dicom(
            args.dicom_dir,
            args.reports_file,
            args.view_labels_file
        )
    
    elif args.data_type == "videos":
        if not args.videos_dir or not args.metadata_file:
            logger.error("Video processing requires --videos_dir and --metadata_file")
            return
        preprocessor.prepare_from_videos(
            args.videos_dir,
            args.metadata_file
        )
    
    elif args.data_type == "sample":
        preprocessor.create_sample_data(args.num_samples)
    
    # Validate the prepared data
    preprocessor.validate_data()
    
    logger.info("Data preparation completed!")
    logger.info(f"Processed data saved to: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review the generated train/val/test CSV files")
    logger.info("2. Adjust any labels or metadata as needed")
    logger.info("3. Run the fine-tuning script with:")
    logger.info(f"   python echoprime_finetune.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()