#!/usr/bin/env python
"""
Script for running EchoPrime inference on multiple device folders.
This script processes each folder in the data/ directory independently and provides a summary.
"""

import os
import math
import glob
import json
import pickle
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pydicom

# Import from our modules
from tools.report_processing import phrase_decode, extract_section, COARSE_VIEWS
from tools.video_io import read_video
from preprocessors.ultrasound_masking import mask_outside_ultrasound
from preprocessors.image_scaling import crop_and_scale

# Constants for video processing
FRAMES_TO_TAKE = 32
FRAME_STRIDE = 2
VIDEO_SIZE = 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

class EchoPrimeInference:
    def __init__(self, weights_dir="weights", device=None):
        """
        Initialize EchoPrime inference pipeline.
        
        Args:
            weights_dir (str): Directory containing model weights
            device (torch.device): Device to run inference on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = Path(weights_dir)
        
        # Load models
        self._load_models()
        
        # Load candidate data
        self._load_candidate_data()
        
        # Load MIL weights
        self._load_mil_weights()
    
    def _load_models(self):
        """Load the EchoPrime encoder and view classifier models."""
        print("Loading EchoPrime models...")
        
        # Load echo encoder
        echo_checkpoint = torch.load(
            self.weights_dir / "echo_prime_encoder.pt", 
            map_location=self.device
        )
        self.echo_encoder = torchvision.models.video.mvit_v2_s()
        self.echo_encoder.head[-1] = torch.nn.Linear(
            self.echo_encoder.head[-1].in_features, 512
        )
        self.echo_encoder.load_state_dict(echo_checkpoint)
        self.echo_encoder.eval()
        self.echo_encoder.to(self.device)
        
        # Load view classifier
        vc_checkpoint = torch.load(
            self.weights_dir / "view_classifier.ckpt", 
            map_location=self.device
        )
        vc_state_dict = {key[6:]: value for key, value in vc_checkpoint['state_dict'].items()}
        self.view_classifier = torchvision.models.convnext_base()
        self.view_classifier.classifier[-1] = torch.nn.Linear(
            self.view_classifier.classifier[-1].in_features, 11
        )
        self.view_classifier.load_state_dict(vc_state_dict)
        self.view_classifier.to(self.device)
        self.view_classifier.eval()
        
        print("Models loaded successfully!")
    
    def _load_candidate_data(self):
        """Load candidate embeddings, reports, and labels."""
        print("Loading candidate data...")
        
        candidates_dir = self.weights_dir / "candidates_data"
        
        # Load candidate studies
        self.candidate_studies = list(
            pd.read_csv(candidates_dir / "candidate_studies.csv")['Study']
        )
        
        # Load candidate embeddings
        candidate_embeddings_p1 = torch.load(candidates_dir / "candidate_embeddings_p1.pt")
        candidate_embeddings_p2 = torch.load(candidates_dir / "candidate_embeddings_p2.pt")
        self.candidate_embeddings = torch.cat((candidate_embeddings_p1, candidate_embeddings_p2), dim=0)
        
        # Load candidate reports and labels
        self.candidate_reports = pd.read_pickle(candidates_dir / "candidate_reports.pkl")
        self.candidate_reports = [phrase_decode(vec_phr) for vec_phr in tqdm(self.candidate_reports, desc="Processing candidate reports")]
        self.candidate_labels = pd.read_pickle(candidates_dir / "candidate_labels.pkl")
        
        # Load section to phenotypes mapping
        self.section_to_phenotypes = pd.read_pickle("section_to_phenotypes.pkl")
        
        print("Candidate data loaded successfully!")
    
    def _load_mil_weights(self):
        """Load MIL weights per section."""
        print("Loading MIL weights...")
        
        mil_weights = pd.read_csv("./weights/MIL_weights.csv")
        self.non_empty_sections = mil_weights['Section']
        self.section_weights = mil_weights.iloc[:, 1:].to_numpy()
        
        print("MIL weights loaded successfully!")
    
    def process_dicoms(self, input_dir, save_extracted_images=False, extracted_images_dir=None):
        """
        Process DICOM files from a directory.
        
        Args:
            input_dir (str): Path to directory containing DICOM files
            save_extracted_images (bool): Whether to save extracted images to data directory
            extracted_images_dir (str): Directory to save extracted images
            
        Returns:
            tuple: (torch.Tensor, dict) - Processed video tensor and error statistics
        """
        dicom_paths = glob.glob(f'{input_dir}/**/*', recursive=True)
        stack_of_videos = []
        
        # Track error statistics
        error_stats = {
            "total_files": len(dicom_paths),
            "processed_files": 0,
            "successful_files": 0,
            "error_counts": {
                "not_dicom": 0,
                "empty_pixel_array": 0,
                "invalid_dimensions": 0,
                "masking_error": 0,
                "scaling_error": 0,
                "other_errors": 0
            },
            "error_files": {
                "not_dicom": [],
                "empty_pixel_array": [],
                "invalid_dimensions": [],
                "masking_error": [],
                "scaling_error": [],
                "other_errors": []
            }
        }
        
        print(f"Processing {len(dicom_paths)} DICOM files from {input_dir}")
        
        for dicom_path in tqdm(dicom_paths, desc="Processing DICOMs"):
            filename = os.path.basename(dicom_path)
            
            try:
                # Try to read DICOM file
                try:
                    dcm = pydicom.dcmread(dicom_path)
                    error_stats["processed_files"] += 1
                except Exception as e:
                    error_stats["error_counts"]["not_dicom"] += 1
                    error_stats["error_files"]["not_dicom"].append(dicom_path)
                    continue
                
                try:
                    pixels = dcm.pixel_array
                except Exception as e:
                    error_stats["error_counts"]["empty_pixel_array"] += 1
                    error_stats["error_files"]["empty_pixel_array"].append(dicom_path)
                    continue
                
                # Check dimensions
                if pixels.ndim < 3:
                    error_stats["error_counts"]["invalid_dimensions"] += 1
                    error_stats["error_files"]["invalid_dimensions"].append(dicom_path)
                    continue
                
                # Exclude images with invalid dimensions (original EchoPrime logic)
                if pixels.shape[2] == 3:
                    error_stats["error_counts"]["invalid_dimensions"] += 1
                    error_stats["error_files"]["invalid_dimensions"].append(dicom_path)
                    continue
                
                # If single channel, repeat to 3 channels
                if pixels.ndim == 3:
                    pixels = np.repeat(pixels[..., None], 3, axis=3)
                
                # Check if pixel array is valid before processing
                if pixels.size == 0 or pixels.shape[0] == 0:
                    error_stats["error_counts"]["empty_pixel_array"] += 1
                    error_stats["error_files"]["empty_pixel_array"].append(dicom_path)
                    continue
                
                # Mask everything outside ultrasound region
                try:
                    pixels = mask_outside_ultrasound(pixels)
                    
                    # Verify pixels are not empty after masking
                    if pixels is None or len(pixels) == 0 or pixels.size == 0:
                        error_stats["error_counts"]["masking_error"] += 1
                        error_stats["error_files"]["masking_error"].append(dicom_path)
                        continue
                except Exception as e:
                    error_stats["error_counts"]["masking_error"] += 1
                    error_stats["error_files"]["masking_error"].append(dicom_path)
                    continue
                
                # Save extracted images to data directory if requested
                if save_extracted_images and extracted_images_dir:
                    os.makedirs(extracted_images_dir, exist_ok=True)
                    base_filename = os.path.splitext(filename)[0]
                    
                    # Create subdirectories for different processing stages
                    raw_dir = os.path.join(extracted_images_dir, "1_raw_extracted")
                    masked_dir = os.path.join(extracted_images_dir, "2_masked")
                    scaled_dir = os.path.join(extracted_images_dir, "3_scaled_cropped")
                    os.makedirs(raw_dir, exist_ok=True)
                    os.makedirs(masked_dir, exist_ok=True)
                    os.makedirs(scaled_dir, exist_ok=True)
                    
                    # Save a few sample frames as images
                    sample_indices = np.linspace(0, len(pixels)-1, min(5, len(pixels)), dtype=int)
                    for idx_frame, frame_idx in enumerate(sample_indices):
                        try:
                            # Save raw frame (before masking) - use the original pixel array
                            raw_frame = dcm.pixel_array[frame_idx]
                            
                            # Validate frame dimensions
                            if raw_frame.size == 0 or raw_frame.shape[0] == 0 or raw_frame.shape[1] == 0:
                                continue
                                
                            # Handle different dimensions properly
                            if raw_frame.ndim == 2:
                                # Grayscale - convert to 3 channel
                                raw_frame = cv2.cvtColor(raw_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                            elif raw_frame.ndim == 3 and raw_frame.shape[2] == 1:
                                # Single channel - repeat to 3 channels
                                raw_frame = np.repeat(raw_frame, 3, axis=2)
                            elif raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
                                # Already 3 channels
                                pass
                            else:
                                # Skip invalid dimensions
                                continue
                            
                            # Ensure valid image dimensions for PNG
                            if raw_frame.shape[0] > 65535 or raw_frame.shape[1] > 65535:
                                continue
                                
                            # Convert to uint8 for saving
                            if raw_frame.max() > 255:
                                raw_frame_uint8 = np.clip(raw_frame, 0, 255).astype(np.uint8)
                            elif raw_frame.max() <= 1.0:
                                raw_frame_uint8 = (raw_frame * 255).astype(np.uint8)
                            else:
                                raw_frame_uint8 = raw_frame.astype(np.uint8)
                            
                            # Save raw extracted frame
                            raw_frame_path = os.path.join(raw_dir, f"{base_filename}_frame_{idx_frame:02d}.png")
                            cv2.imwrite(raw_frame_path, raw_frame_uint8)
                        except Exception as e:
                            # Skip this frame if there's an error saving it
                            continue
                        
                        # Save masked frame
                        masked_frame = pixels[frame_idx]
                        if masked_frame.max() > 1.0:
                            masked_frame_uint8 = np.clip(masked_frame, 0, 255).astype(np.uint8)
                        else:
                            masked_frame_uint8 = (masked_frame * 255).astype(np.uint8)
                        
                        masked_frame_path = os.path.join(masked_dir, f"{base_filename}_frame_{idx_frame:02d}.png")
                        cv2.imwrite(masked_frame_path, masked_frame_uint8)
                
                # Model specific preprocessing and save scaled images
                x = np.zeros((len(pixels), VIDEO_SIZE, VIDEO_SIZE, 3))
                valid_frames = True
                for i in range(len(x)):
                    # Check if the frame is valid before scaling
                    if pixels[i].size == 0 or pixels[i].shape[0] == 0 or pixels[i].shape[1] == 0:
                        error_stats["error_counts"]["invalid_dimensions"] += 1
                        error_stats["error_files"]["invalid_dimensions"].append(f"{dicom_path} (frame {i})")
                        valid_frames = False
                        break
                    try:
                        x[i] = crop_and_scale(pixels[i], res=(VIDEO_SIZE, VIDEO_SIZE))
                    except Exception as e:
                        error_stats["error_counts"]["scaling_error"] += 1
                        error_stats["error_files"]["scaling_error"].append(f"{dicom_path} (frame {i})")
                        valid_frames = False
                        break
                
                # Save scaled/cropped frames if processing was successful
                if valid_frames and save_extracted_images and extracted_images_dir:
                    # Save scaled frames for the same sample indices
                    sample_indices = np.linspace(0, len(pixels)-1, min(5, len(pixels)), dtype=int)
                    for idx_frame, frame_idx in enumerate(sample_indices):
                        if frame_idx < len(x):
                            scaled_frame = x[frame_idx]
                            # Convert to uint8 for saving
                            if scaled_frame.max() > 1.0:
                                scaled_frame_uint8 = np.clip(scaled_frame, 0, 255).astype(np.uint8)
                            else:
                                scaled_frame_uint8 = (scaled_frame * 255).astype(np.uint8)
                            
                            scaled_frame_path = os.path.join(scaled_dir, f"{base_filename}_frame_{idx_frame:02d}.png")
                            cv2.imwrite(scaled_frame_path, scaled_frame_uint8)
                
                # Skip if any frame processing failed
                if not valid_frames:
                    continue
                
                x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
                
                # Normalize
                x.sub_(MEAN).div_(STD)
                
                # If not enough frames, add padding
                if x.shape[1] < FRAMES_TO_TAKE:
                    padding = torch.zeros(
                        (3, FRAMES_TO_TAKE - x.shape[1], VIDEO_SIZE, VIDEO_SIZE),
                        dtype=torch.float,
                    )
                    x = torch.cat((x, padding), dim=1)
                
                start = 0
                stack_of_videos.append(x[:, start:(start + FRAMES_TO_TAKE):FRAME_STRIDE, :, :])
                error_stats["successful_files"] += 1
                
            except Exception as e:
                error_stats["error_counts"]["other_errors"] += 1
                error_stats["error_files"]["other_errors"].append(dicom_path)
                print(f"Corrupt file or unexpected error: {dicom_path}")
                print(str(e))
        
        if not stack_of_videos:
            return None, error_stats
            
        return torch.stack(stack_of_videos), error_stats
    
    def embed_videos(self, stack_of_videos):
        """
        Embed videos using the EchoPrime encoder.
        
        Args:
            stack_of_videos (torch.Tensor): Video tensor
            
        Returns:
            torch.Tensor: Video embeddings
        """
        bin_size = 50
        n_bins = math.ceil(stack_of_videos.shape[0] / bin_size)
        stack_of_features_list = []
        
        with torch.no_grad():
            for bin_idx in range(n_bins):
                start_idx = bin_idx * bin_size
                end_idx = min((bin_idx + 1) * bin_size, stack_of_videos.shape[0])
                bin_videos = stack_of_videos[start_idx:end_idx].to(self.device)
                bin_features = self.echo_encoder(bin_videos)
                stack_of_features_list.append(bin_features)
            
            stack_of_features = torch.cat(stack_of_features_list, dim=0)
        
        return stack_of_features
    
    def get_views(self, stack_of_videos):
        """
        Get view classifications for videos.
        
        Args:
            stack_of_videos (torch.Tensor): Video tensor
            
        Returns:
            torch.Tensor: View encodings
        """
        stack_of_first_frames = stack_of_videos[:, :, 0, :, :].to(self.device)
        
        with torch.no_grad():
            out_logits = self.view_classifier(stack_of_first_frames)
        
        out_views = torch.argmax(out_logits, dim=1)
        stack_of_view_encodings = torch.nn.functional.one_hot(out_views, 11).to(self.device)
        
        return stack_of_view_encodings
    
    def encode_study(self, input_dir, save_extracted_images=False, extracted_images_dir=None):
        """
        Encode an entire study.
        
        Args:
            input_dir (str): Path to study directory
            save_extracted_images (bool): Whether to save extracted images to data directory
            extracted_images_dir (str): Directory to save extracted images
            
        Returns:
            tuple: (torch.Tensor, dict) - Study encoding and error statistics
        """
        stack_of_videos, error_stats = self.process_dicoms(
            input_dir, 
            save_extracted_images=save_extracted_images, 
            extracted_images_dir=extracted_images_dir
        )
        
        if stack_of_videos is None:
            return None, error_stats
        
        stack_of_features = self.embed_videos(stack_of_videos)
        stack_of_view_encodings = self.get_views(stack_of_videos)
        encoded_study = torch.cat((stack_of_features, stack_of_view_encodings), dim=1)
        
        return encoded_study, error_stats
    
    def generate_report(self, study_embedding):
        """
        Generate a report from study embedding.
        
        Args:
            study_embedding (torch.Tensor): Study embedding
            
        Returns:
            str: Generated report
        """
        study_embedding = study_embedding.cpu()
        generated_report = ""
        
        for s_dx, sec in enumerate(self.non_empty_sections):
            cur_weights = [self.section_weights[s_dx][torch.where(ten == 1)[0]] 
                          for ten in study_embedding[:, 512:]]
            no_view_study_embedding = study_embedding[:, :512] * torch.tensor(cur_weights, dtype=torch.float).unsqueeze(1)
            no_view_study_embedding = torch.mean(no_view_study_embedding, dim=0)
            no_view_study_embedding = torch.nn.functional.normalize(no_view_study_embedding, dim=0)
            similarities = no_view_study_embedding @ self.candidate_embeddings.T
            
            extracted_section = "Section not found."
            while extracted_section == "Section not found.":
                max_id = torch.argmax(similarities)
                predicted_section = self.candidate_reports[max_id]
                extracted_section = extract_section(predicted_section, sec)
                if extracted_section != "Section not found.":
                    generated_report += extracted_section
                similarities[max_id] = float('-inf')
        
        return generated_report
    
    def predict_metrics(self, study_embedding, k=50):
        """
        Predict metrics from study embedding.
        
        Args:
            study_embedding (torch.Tensor): Study embedding
            k (int): Number of top candidates to consider
            
        Returns:
            dict: Predicted metrics
        """
        per_section_study_embedding = torch.zeros(len(self.non_empty_sections), 512)
        study_embedding = study_embedding.cpu()
        
        for s_dx, sec in enumerate(self.non_empty_sections):
            this_section_weights = [self.section_weights[s_dx][torch.where(view_encoding == 1)[0]]
                                   for view_encoding in study_embedding[:, 512:]]
            this_section_study_embedding = (study_embedding[:, :512] * 
                                           torch.tensor(this_section_weights, dtype=torch.float).unsqueeze(1))
            this_section_study_embedding = torch.sum(this_section_study_embedding, dim=0)
            per_section_study_embedding[s_dx] = this_section_study_embedding
        
        per_section_study_embedding = torch.nn.functional.normalize(per_section_study_embedding)
        similarities = per_section_study_embedding @ self.candidate_embeddings.T
        top_candidate_ids = torch.topk(similarities, k=k, dim=1).indices
        
        preds = {}
        for s_dx, section in enumerate(self.section_to_phenotypes.keys()):
            for pheno in self.section_to_phenotypes[section]:
                preds[pheno] = np.nanmean([self.candidate_labels[pheno][self.candidate_studies[c_ids]]
                                          for c_ids in top_candidate_ids[s_dx]
                                          if self.candidate_studies[c_ids] in self.candidate_labels[pheno]])
        
        return preds
    
    def save_failed_files_to_json(self, folder_name, error_stats, save_dir):
        """
        Save failed files information to a JSON file.
        
        Args:
            folder_name (str): Name of the folder being processed
            error_stats (dict): Error statistics dictionary
            save_dir (str): Directory to save the failed files JSON
        """
        if not save_dir:
            return
            
        # Create directory for failed files if it doesn't exist
        failed_files_dir = save_dir
        os.makedirs(failed_files_dir, exist_ok=True)
        
        # Create a structured dictionary of failed files with reasons
        failed_files = {}
        for error_type, file_list in error_stats["error_files"].items():
            for file_path in file_list:
                # Use the filename as the key
                filename = os.path.basename(file_path)
                # If the file path contains a frame index (for frame-specific errors)
                if " (frame " in file_path:
                    base_path, frame_info = file_path.split(" (frame ", 1)
                    filename = os.path.basename(base_path)
                    frame_num = frame_info.rstrip(")")
                    error_reason = f"{error_type} in frame {frame_num}"
                else:
                    error_reason = error_type
                
                # Add to the failed files dictionary
                if filename not in failed_files:
                    failed_files[filename] = {
                        "path": file_path.split(" (frame ")[0] if " (frame " in file_path else file_path,
                        "reasons": [error_reason]
                    }
                else:
                    # If this file already has other errors, add this reason
                    if error_reason not in failed_files[filename]["reasons"]:
                        failed_files[filename]["reasons"].append(error_reason)
        
        # Convert to a list format for easier processing
        failed_files_list = [
            {
                "filename": filename,
                "path": info["path"],
                "reasons": info["reasons"]
            }
            for filename, info in failed_files.items()
        ]
        
        # Save to a JSON file named after the folder
        output_file = os.path.join(failed_files_dir, f"{folder_name}_failed_files.json")
        with open(output_file, "w") as f:
            json.dump({
                "folder": folder_name,
                "total_failed_files": len(failed_files_list),
                "failed_files": failed_files_list
            }, f, indent=2)
        
        print(f"Failed files for {folder_name} saved to {output_file}")

    def print_error_summary(self, folder_name, error_stats):
        """
        Print a clean summary of errors for a folder.
        
        Args:
            folder_name (str): Name of the folder
            error_stats (dict): Error statistics dictionary
        """
        if not error_stats or not error_stats.get("error_counts"):
            return
            
        total_errors = sum(error_stats["error_counts"].values())
        if total_errors == 0:
            return
            
        print(f"\n📊 Error Summary for {folder_name}:")
        print(f"   Total files: {error_stats['total_files']}")
        print(f"   Successful: {error_stats['successful_files']}")
        print(f"   Failed: {total_errors}")
        print("   Error breakdown:")
        
        for error_type, count in error_stats["error_counts"].items():
            if count > 0:
                error_descriptions = {
                    "not_dicom": "Not valid DICOM files",
                    "empty_pixel_array": "Empty or missing pixel data",
                    "invalid_dimensions": "Invalid image dimensions",
                    "masking_error": "Ultrasound masking failed",
                    "scaling_error": "Frame scaling/cropping failed",
                    "other_errors": "Other processing errors"
                }
                description = error_descriptions.get(error_type, error_type)
                print(f"     • {description}: {count}")
    
    def process_folder(self, folder_path, save_dir=None):
        """
        Process a single folder and return results.
        
        Args:
            folder_path (str): Path to folder containing DICOM files
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
            
            # Create preprocessed_data directory for extracted images
            data_save_dir = os.path.join("preprocessed_data", folder_name)
            
            # Encode the study with real-time image saving
            encoded_study, error_stats = self.encode_study(
                folder_path,
                save_extracted_images=True,
                extracted_images_dir=data_save_dir
            )
            
            # Save failed files to JSON if there are any errors
            if folder_save_dir and error_stats:
                self.save_failed_files_to_json(folder_name, error_stats, folder_save_dir)
            
            # Print error summary for this folder
            self.print_error_summary(folder_name, error_stats)
            
            if encoded_study is None:
                return {
                    "folder": folder_name,
                    "status": "failed",
                    "error": "No valid DICOM files found",
                    "num_files": error_stats.get("total_files", 0),
                    "num_processed": error_stats.get("successful_files", 0),
                    "num_videos": 0,
                    "error_stats": error_stats
                }
            
            # Generate report and metrics
            report = self.generate_report(encoded_study)
            metrics = self.predict_metrics(encoded_study)
            
            # Save results to JSON if save_dir is provided
            result = {
                "folder": folder_name,
                "status": "success",
                "num_files": error_stats.get("total_files", 0),
                "num_processed": error_stats.get("successful_files", 0),
                "num_videos": encoded_study.shape[0],
                "report": report,
                "metrics": metrics,
                "error_stats": error_stats
            }
            
            if folder_save_dir:
                with open(os.path.join(folder_save_dir, "results.json"), "w") as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Results for {folder_name} saved to {folder_save_dir}")
            
            return result
            
        except Exception as e:
            error_result = {
                "folder": folder_name,
                "status": "failed",
                "error": str(e),
                "num_files": 0,
                "num_processed": 0,
                "num_videos": 0
            }
            
            # Save error result if save_dir is provided
            if save_dir:
                folder_save_dir = os.path.join(save_dir, folder_name)
                os.makedirs(folder_save_dir, exist_ok=True)
                with open(os.path.join(folder_save_dir, "results.json"), "w") as f:
                    json.dump(error_result, f, indent=2, default=str)
            
            return error_result

def clean_output_directories():
    """
    Clean the data and results directories to ensure fresh output for each run.
    """
    directories_to_clean = ["preprocessed_data", "results"]
    
    for directory in directories_to_clean:
        if os.path.exists(directory):
            print(f"🧹 Cleaning {directory}/ directory...")
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created fresh {directory}/ directory")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run EchoPrime inference on device folders.")
    parser.add_argument("--data_dir", type=str, default="raw_data", help="Directory containing device folders")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory containing model weights")
    parser.add_argument("--output", type=str, default="results/inference_output", help="Directory to save results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to run inference on")
    args = parser.parse_args()
    
    # Clean output directories before starting
    clean_output_directories()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize inference pipeline
    inference = EchoPrimeInference(weights_dir=args.weights_dir, device=device)
    
    # Find all folders in data directory
    data_dir = Path(args.data_dir)
    folders = [f for f in data_dir.iterdir() if f.is_dir()]
    
    if not folders:
        print(f"No folders found in {args.data_dir}")
        return
    
    print(f"Found {len(folders)} folders to process: {[f.name for f in folders]}")
    
    # Process each folder with real-time saving
    all_results = []
    for folder in folders:
        result = inference.process_folder(str(folder), save_dir=args.output)
        all_results.append(result)
    
    # Save individual results
    for result in all_results:
        folder_output_dir = os.path.join(args.output, result["folder"])
        os.makedirs(folder_output_dir, exist_ok=True)
        
        with open(os.path.join(folder_output_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
    
    # Create summary
    summary = {
        "total_folders": len(all_results),
        "successful": sum(1 for r in all_results if r["status"] == "success"),
        "failed": sum(1 for r in all_results if r["status"] == "failed"),
        "total_videos": sum(r.get("num_videos", 0) for r in all_results),
        "results": all_results
    }
    
    # Save summary
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("ECHOPRIME INFERENCE SUMMARY")
    print("="*80)
    print(f"Total folders processed: {summary['total_folders']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total videos processed: {summary['total_videos']}")
    print()
    
    for result in all_results:
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"{status_icon} {result['folder']:<30} {result.get('num_videos', 0):>3} videos")
        if result["status"] == "failed":
            print(f"    Error: {result.get('error', 'Unknown error')}")
    
    print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
