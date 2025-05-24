#!/usr/bin/env python
"""
Script for running inference with the trained echo quality model.
This script processes each folder in the data/ directory independently and provides a summary.
"""

import os
import torch
import numpy as np
import glob
import pydicom
import cv2
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from torchvision.models.video import r2plus1d_18
import matplotlib.pyplot as plt

# Import from our modules
from inference.EchoPrime_qc import mask_outside_ultrasound, crop_and_scale, get_quality_issues

# Optional import for GradCAM
try:
    from training.echo_model_evaluation import visualize_gradcam
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: GradCAM visualization not available. Install seaborn to enable GradCAM features.")

# Constants for video processing (same as in EchoPrime_qc.py)
frames_to_take = 32
frame_stride = 2
video_size = 112
mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

class EchoQualityInference:
    def __init__(self, model_path="weights/video_quality_model.pt", device=None):
        """
        Initialize EchoQuality inference pipeline.
        
        Args:
            model_path (str): Path to model weights
            device (torch.device): Device to run inference on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the echo quality model."""
        print(f"Loading model from {self.model_path}...")
        self.model = r2plus1d_18(num_classes=1)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def process_dicom(self, dicom_path, save_mask_images=False):
        """
        Process a single DICOM file and prepare it for the model.
        
        Args:
            dicom_path (str): Path to the DICOM file
            save_mask_images (bool): Whether to save mask images
            
        Returns:
            torch.Tensor: Processed video tensor
        """
        try:
            # Read DICOM file
            dcm = pydicom.dcmread(dicom_path)
            pixels = dcm.pixel_array
            
            # Handle different dimensions
            if pixels.ndim < 3 or pixels.shape[2] == 3:
                print(f"Skipping {dicom_path}: Invalid dimensions {pixels.shape}")
                return None
            
            # If single channel, repeat to 3 channels
            if pixels.ndim == 3:
                pixels = np.repeat(pixels[..., None], 3, axis=3)
            
            # Mask everything outside ultrasound region
            filename = os.path.basename(dicom_path)
            pixels = mask_outside_ultrasound(pixels, filename if save_mask_images else None, save_mask_images)
            
            # Model specific preprocessing
            x = np.zeros((len(pixels), video_size, video_size, 3))
            for i in range(len(x)):
                x[i] = crop_and_scale(pixels[i])
            
            # Convert to tensor and permute dimensions
            x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
            
            # Normalize
            x.sub_(mean).div_(std)
            
            # If not enough frames, add padding
            if x.shape[1] < frames_to_take:
                padding = torch.zeros(
                    (
                        3,
                        frames_to_take - x.shape[1],
                        video_size,
                        video_size,
                    ),
                    dtype=torch.float,
                )
                x = torch.cat((x, padding), dim=1)
            
            # Apply stride and take required frames
            start = 0
            x = x[:, start: (start + frames_to_take): frame_stride, :, :]
            
            return x
        
        except Exception as e:
            print(f"Error processing {dicom_path}: {str(e)}")
            return None

    def run_inference_on_folder(self, folder_path, threshold=0.3, save_dir=None, generate_gradcam=False):
        """
        Run inference on all DICOM files in a folder.
        
        Args:
            folder_path (str): Path to folder containing DICOM files
            threshold (float): Threshold for binary classification
            save_dir (str, optional): Directory to save results and visualizations
            generate_gradcam (bool): Whether to generate GradCAM visualizations
            
        Returns:
            dict: Dictionary of results
        """
        # Find DICOM files
        dicom_paths = glob.glob(f"{folder_path}/**/*", recursive=True)
        dicom_paths = [p for p in dicom_paths if p.lower().endswith('.dcm') or os.path.isfile(p)]
        
        if not dicom_paths:
            return {"error": "No DICOM files found", "results": {}}
        
        print(f"Processing {len(dicom_paths)} files from {folder_path}")
        
        results = {}
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            if generate_gradcam:
                os.makedirs(os.path.join(save_dir, "gradcam"), exist_ok=True)
        
        for dicom_path in tqdm(dicom_paths, desc="Processing"):
            filename = os.path.basename(dicom_path)
            
            # Process DICOM file
            video = self.process_dicom(dicom_path, save_mask_images=(save_dir is not None))
            
            if video is None:
                print(f"Skipping {filename}: Processing failed")
                continue
            
            # Run inference
            with torch.no_grad():
                video = video.unsqueeze(0).to(self.device)  # Add batch dimension
                output = self.model(video)
                probability = torch.sigmoid(output).item()
                prediction = 1 if probability >= threshold else 0
                status = "PASS" if prediction > 0 else "FAIL"
                assessment = get_quality_issues(probability)
            
            # Store results
            results[filename] = {
                "score": probability,
                "status": status,
                "assessment": assessment
            }
            
            # Generate GradCAM visualization if requested
            if generate_gradcam and save_dir and GRADCAM_AVAILABLE:
                try:
                    visualize_gradcam(
                        self.model, 
                        video.squeeze(0).cpu(), 
                        target_layer_name="layer4", 
                        save_path=os.path.join(save_dir, "gradcam", f"{filename.replace('.dcm', '')}_gradcam.png")
                    )
                except Exception as e:
                    print(f"Error generating GradCAM for {filename}: {str(e)}")
            elif generate_gradcam and save_dir and not GRADCAM_AVAILABLE:
                print(f"Skipping GradCAM for {filename}: GradCAM not available (install seaborn)")
        
        # Save results to JSON if save_dir is provided
        if save_dir and results:
            with open(os.path.join(save_dir, "inference_results.json"), "w") as f:
                json.dump(results, f, indent=2)
            
            # Create a summary plot
            self.create_summary_plot(results, save_dir)
        
        return {"results": results, "total_files": len(dicom_paths)}

    def create_summary_plot(self, results, save_dir):
        """
        Create a summary plot of the inference results.
        
        Args:
            results (dict): Dictionary of inference results
            save_dir (str): Directory to save the plot
        """
        if not results:
            return
            
        # Extract scores
        scores = [result["score"] for result in results.values()]
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, alpha=0.7, color='blue')
        plt.axvline(x=0.3, color='red', linestyle='--', label='Threshold (0.3)')
        plt.xlabel('Quality Score')
        plt.ylabel('Count')
        plt.title('Distribution of Echo Quality Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "score_distribution.png"))
        plt.close()
        
        # Create pie chart of pass/fail
        pass_count = sum(1 for result in results.values() if result["status"] == "PASS")
        fail_count = len(results) - pass_count
        
        plt.figure(figsize=(8, 8))
        plt.pie(
            [pass_count, fail_count], 
            labels=["PASS", "FAIL"], 
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'],
            explode=(0.1, 0)
        )
        plt.title('Pass/Fail Distribution')
        plt.savefig(os.path.join(save_dir, "pass_fail_distribution.png"))
        plt.close()

    def process_folder(self, folder_path, threshold=0.3, generate_gradcam=False, save_dir=None):
        """
        Process a single folder and return results.
        
        Args:
            folder_path (str): Path to folder containing DICOM files
            threshold (float): Threshold for binary classification
            generate_gradcam (bool): Whether to generate GradCAM visualizations
            save_dir (str, optional): Directory to save results
            
        Returns:
            dict: Processing results
        """
        folder_name = os.path.basename(folder_path)
        print(f"\nProcessing folder: {folder_name}")
        
        try:
            # Create folder-specific save directory
            folder_save_dir = None
            if save_dir:
                folder_save_dir = os.path.join(save_dir, folder_name)
                os.makedirs(folder_save_dir, exist_ok=True)
            
            # Run inference on folder
            inference_result = self.run_inference_on_folder(
                folder_path, 
                threshold=threshold, 
                save_dir=folder_save_dir,
                generate_gradcam=generate_gradcam
            )
            
            if "error" in inference_result:
                return {
                    "folder": folder_name,
                    "status": "failed",
                    "error": inference_result["error"],
                    "num_files": 0,
                    "pass_count": 0,
                    "fail_count": 0,
                    "pass_rate": 0.0
                }
            
            results = inference_result["results"]
            total_files = inference_result["total_files"]
            
            # Calculate statistics
            pass_count = sum(1 for result in results.values() if result["status"] == "PASS")
            fail_count = len(results) - pass_count
            pass_rate = (pass_count / len(results) * 100) if results else 0.0
            
            return {
                "folder": folder_name,
                "status": "success",
                "num_files": total_files,
                "num_processed": len(results),
                "pass_count": pass_count,
                "fail_count": fail_count,
                "pass_rate": pass_rate,
                "results": results
            }
            
        except Exception as e:
            return {
                "folder": folder_name,
                "status": "failed",
                "error": str(e),
                "num_files": 0,
                "pass_count": 0,
                "fail_count": 0,
                "pass_rate": 0.0
            }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run inference with the echo quality model on device folders.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing device folders")
    parser.add_argument("--model", type=str, default="weights/video_quality_model.pt", help="Path to model weights")
    parser.add_argument("--output", type=str, default="results/inference_output", help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for binary classification")
    parser.add_argument("--gradcam", action="store_true", help="Generate GradCAM visualizations")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to run inference on")
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize inference pipeline
    inference = EchoQualityInference(model_path=args.model, device=device)
    
    # Find all folders in data directory
    data_dir = Path(args.data_dir)
    folders = [f for f in data_dir.iterdir() if f.is_dir()]
    
    if not folders:
        print(f"No folders found in {args.data_dir}")
        return
    
    print(f"Found {len(folders)} folders to process: {[f.name for f in folders]}")
    
    # Process each folder
    all_results = []
    for folder in folders:
        result = inference.process_folder(
            str(folder), 
            threshold=args.threshold,
            generate_gradcam=args.gradcam,
            save_dir=args.output
        )
        all_results.append(result)
    
    # Save individual results
    for result in all_results:
        folder_output_dir = os.path.join(args.output, result["folder"])
        os.makedirs(folder_output_dir, exist_ok=True)
        
        with open(os.path.join(folder_output_dir, "folder_summary.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
    
    # Create overall summary
    total_folders = len(all_results)
    successful_folders = sum(1 for r in all_results if r["status"] == "success")
    failed_folders = total_folders - successful_folders
    total_files = sum(r.get("num_files", 0) for r in all_results)
    total_processed = sum(r.get("num_processed", 0) for r in all_results)
    total_pass = sum(r.get("pass_count", 0) for r in all_results)
    total_fail = sum(r.get("fail_count", 0) for r in all_results)
    overall_pass_rate = (total_pass / total_processed * 100) if total_processed > 0 else 0.0
    
    summary = {
        "total_folders": total_folders,
        "successful_folders": successful_folders,
        "failed_folders": failed_folders,
        "total_files": total_files,
        "total_processed": total_processed,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "overall_pass_rate": overall_pass_rate,
        "folder_results": all_results
    }
    
    # Save summary
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("ECHO QUALITY INFERENCE SUMMARY")
    print("="*80)
    print(f"Total folders processed: {total_folders}")
    print(f"Successful: {successful_folders}")
    print(f"Failed: {failed_folders}")
    print(f"Total files found: {total_files}")
    print(f"Total files processed: {total_processed}")
    print(f"Overall pass rate: {overall_pass_rate:.1f}% ({total_pass}/{total_processed})")
    print()
    
    for result in all_results:
        status_icon = "✓" if result["status"] == "success" else "✗"
        if result["status"] == "success":
            print(f"{status_icon} {result['folder']:<30} {result['num_processed']:>3}/{result['num_files']:<3} files  Pass: {result['pass_rate']:>5.1f}%")
        else:
            print(f"{status_icon} {result['folder']:<30} {result.get('num_files', 0):>3} files  Error: {result.get('error', 'Unknown error')}")
    
    print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
