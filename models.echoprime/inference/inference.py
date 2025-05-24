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
        
        mil_weights = pd.read_csv("MIL_weights.csv")
        self.non_empty_sections = mil_weights['Section']
        self.section_weights = mil_weights.iloc[:, 1:].to_numpy()
        
        print("MIL weights loaded successfully!")
    
    def process_dicoms(self, input_dir):
        """
        Process DICOM files from a directory.
        
        Args:
            input_dir (str): Path to directory containing DICOM files
            
        Returns:
            torch.Tensor: Processed video tensor
        """
        dicom_paths = glob.glob(f'{input_dir}/**/*', recursive=True)
        stack_of_videos = []
        
        print(f"Processing {len(dicom_paths)} DICOM files from {input_dir}")
        
        for dicom_path in tqdm(dicom_paths, desc="Processing DICOMs"):
            try:
                # Read DICOM file
                dcm = pydicom.dcmread(dicom_path)
                pixels = dcm.pixel_array
                
                # Exclude images with invalid dimensions
                if pixels.ndim < 3 or pixels.shape[2] == 3:
                    continue
                
                # If single channel, repeat to 3 channels
                if pixels.ndim == 3:
                    pixels = np.repeat(pixels[..., None], 3, axis=3)
                
                # Mask everything outside ultrasound region
                pixels = mask_outside_ultrasound(pixels)
                
                # Model specific preprocessing
                x = np.zeros((len(pixels), VIDEO_SIZE, VIDEO_SIZE, 3))
                for i in range(len(x)):
                    x[i] = crop_and_scale(pixels[i], res=(VIDEO_SIZE, VIDEO_SIZE))
                
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
                
            except Exception as e:
                print(f"Error processing {dicom_path}: {str(e)}")
                continue
        
        if not stack_of_videos:
            return None
            
        return torch.stack(stack_of_videos)
    
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
    
    def encode_study(self, input_dir):
        """
        Encode an entire study.
        
        Args:
            input_dir (str): Path to study directory
            
        Returns:
            torch.Tensor: Study encoding
        """
        stack_of_videos = self.process_dicoms(input_dir)
        
        if stack_of_videos is None:
            return None
        
        stack_of_features = self.embed_videos(stack_of_videos)
        stack_of_view_encodings = self.get_views(stack_of_videos)
        encoded_study = torch.cat((stack_of_features, stack_of_view_encodings), dim=1)
        
        return encoded_study
    
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
    
    def process_folder(self, folder_path):
        """
        Process a single folder and return results.
        
        Args:
            folder_path (str): Path to folder containing DICOM files
            
        Returns:
            dict: Processing results
        """
        folder_name = os.path.basename(folder_path)
        print(f"\nProcessing folder: {folder_name}")
        
        try:
            # Encode the study
            encoded_study = self.encode_study(folder_path)
            
            if encoded_study is None:
                return {
                    "folder": folder_name,
                    "status": "failed",
                    "error": "No valid DICOM files found",
                    "num_videos": 0
                }
            
            # Generate report and metrics
            report = self.generate_report(encoded_study)
            metrics = self.predict_metrics(encoded_study)
            
            return {
                "folder": folder_name,
                "status": "success",
                "num_videos": encoded_study.shape[0],
                "report": report,
                "metrics": metrics
            }
            
        except Exception as e:
            return {
                "folder": folder_name,
                "status": "failed",
                "error": str(e),
                "num_videos": 0
            }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run EchoPrime inference on device folders.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing device folders")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory containing model weights")
    parser.add_argument("--output", type=str, default="results/inference_output", help="Directory to save results")
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
    inference = EchoPrimeInference(weights_dir=args.weights_dir, device=device)
    
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
        result = inference.process_folder(str(folder))
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
