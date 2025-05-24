#!/usr/bin/env python
"""
Enhanced EchoPrime inference script for individual patient processing.
This script processes each DICOM file as a separate patient and provides comprehensive
device-specific analysis to help optimize capture protocols.
"""

import os
import math
import glob
import json
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import statistics

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

# View classification mapping
VIEW_NAMES = {
    0: "parasternal_long",
    1: "parasternal_short",
    2: "apical_4chamber",
    3: "apical_2chamber", 
    4: "apical_3chamber",
    5: "subcostal_4chamber",
    6: "subcostal_ivc",
    7: "suprasternal",
    8: "other",
    9: "poor_quality",
    10: "off_axis"
}

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
    
    def process_single_dicom(self, dicom_path):
        """
        Process a single DICOM file.
        
        Args:
            dicom_path (str): Path to DICOM file
            
        Returns:
            tuple: (processed_video_tensor, view_classifications, processing_info)
        """
        processing_info = {
            "original_frames": 0,
            "processed_frames": 0,
            "processing_errors": []
        }
        
        try:
            # Read DICOM file
            dcm = pydicom.dcmread(dicom_path)
            pixels = dcm.pixel_array
            processing_info["original_frames"] = len(pixels) if pixels.ndim > 2 else 1
            
            # Exclude images with invalid dimensions
            if pixels.ndim < 3 or pixels.shape[2] == 3:
                processing_info["processing_errors"].append("Invalid pixel dimensions")
                return None, None, processing_info
            
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
            processed_video = x[:, start:(start + FRAMES_TO_TAKE):FRAME_STRIDE, :, :]
            processing_info["processed_frames"] = processed_video.shape[1]
            
            # Get view classification
            first_frame = processed_video[:, 0:1, :, :].to(self.device)
            with torch.no_grad():
                view_logits = self.view_classifier(first_frame)
                view_probs = torch.softmax(view_logits, dim=1)
                view_class = torch.argmax(view_logits, dim=1).item()
                view_confidence = view_probs.max().item()
            
            view_info = {
                "view_class": view_class,
                "view_name": VIEW_NAMES.get(view_class, "unknown"),
                "confidence": view_confidence,
                "all_probabilities": view_probs.cpu().numpy().tolist()[0]
            }
            
            return processed_video.unsqueeze(0), view_info, processing_info
            
        except Exception as e:
            processing_info["processing_errors"].append(str(e))
            return None, None, processing_info
    
    def embed_videos(self, stack_of_videos):
        """
        Embed videos using the EchoPrime encoder.
        
        Args:
            stack_of_videos (torch.Tensor): Video tensor
            
        Returns:
            torch.Tensor: Video embeddings
        """
        with torch.no_grad():
            stack_of_videos = stack_of_videos.to(self.device)
            features = self.echo_encoder(stack_of_videos)
        
        return features
    
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
    
    def process_patient(self, dicom_path, device_name):
        """
        Process a single patient (DICOM file).
        
        Args:
            dicom_path (str): Path to DICOM file
            device_name (str): Name of the device
            
        Returns:
            dict: Patient processing results
        """
        patient_id = Path(dicom_path).stem
        
        try:
            # Process the DICOM file
            processed_video, view_info, processing_info = self.process_single_dicom(dicom_path)
            
            if processed_video is None:
                return {
                    "patient_id": patient_id,
                    "device": device_name,
                    "status": "failed",
                    "error": "; ".join(processing_info["processing_errors"]),
                    "processing_info": processing_info
                }
            
            # Embed the video
            features = self.embed_videos(processed_video)
            view_encodings = self.get_views(processed_video)
            study_embedding = torch.cat((features, view_encodings), dim=1)
            
            # Generate report and metrics
            report = self.generate_report(study_embedding)
            metrics = self.predict_metrics(study_embedding)
            
            # Analyze report quality
            quality_indicators = self._analyze_report_quality(report)
            
            return {
                "patient_id": patient_id,
                "device": device_name,
                "status": "success",
                "report": report,
                "metrics": metrics,
                "view_info": view_info,
                "processing_info": processing_info,
                "quality_indicators": quality_indicators
            }
            
        except Exception as e:
            return {
                "patient_id": patient_id,
                "device": device_name,
                "status": "failed",
                "error": str(e),
                "processing_info": {"processing_errors": [str(e)]}
            }
    
    def _analyze_report_quality(self, report):
        """
        Analyze the quality of the generated report.
        
        Args:
            report (str): Generated clinical report
            
        Returns:
            dict: Quality indicators
        """
        quality_indicators = {
            "well_visualized_count": report.lower().count("well visualized"),
            "not_well_visualized_count": report.lower().count("not well visualized"),
            "normal_count": report.lower().count("normal"),
            "abnormal_findings_count": 0,
            "total_sections": len([s for s in report.split("[SEP]") if s.strip()]),
            "report_length": len(report)
        }
        
        # Count abnormal findings
        abnormal_terms = ["dilated", "hypertrophy", "regurgitation", "stenosis", "dysfunction", 
                         "abnormal", "elevated", "depressed", "thickened", "calcification"]
        for term in abnormal_terms:
            quality_indicators["abnormal_findings_count"] += report.lower().count(term)
        
        # Calculate visualization rate
        total_viz_mentions = quality_indicators["well_visualized_count"] + quality_indicators["not_well_visualized_count"]
        if total_viz_mentions > 0:
            quality_indicators["visualization_rate"] = quality_indicators["well_visualized_count"] / total_viz_mentions
        else:
            quality_indicators["visualization_rate"] = 1.0  # Assume good if not mentioned
        
        return quality_indicators

def analyze_device_performance(all_results):
    """
    Analyze performance across different devices.
    
    Args:
        all_results (list): List of all patient results
        
    Returns:
        dict: Device analysis
    """
    device_stats = defaultdict(lambda: {
        "total_patients": 0,
        "successful_patients": 0,
        "failed_patients": 0,
        "metrics": defaultdict(list),
        "view_distribution": defaultdict(int),
        "quality_indicators": defaultdict(list),
        "processing_info": defaultdict(list)
    })
    
    # Collect statistics per device
    for result in all_results:
        device = result["device"]
        stats = device_stats[device]
        
        stats["total_patients"] += 1
        
        if result["status"] == "success":
            stats["successful_patients"] += 1
            
            # Collect metrics
            for metric, value in result["metrics"].items():
                if not np.isnan(value):
                    stats["metrics"][metric].append(value)
            
            # Collect view information
            if "view_info" in result:
                view_name = result["view_info"]["view_name"]
                stats["view_distribution"][view_name] += 1
            
            # Collect quality indicators
            if "quality_indicators" in result:
                for indicator, value in result["quality_indicators"].items():
                    stats["quality_indicators"][indicator].append(value)
            
            # Collect processing info
            if "processing_info" in result:
                for key, value in result["processing_info"].items():
                    if isinstance(value, (int, float)):
                        stats["processing_info"][key].append(value)
        else:
            stats["failed_patients"] += 1
    
    # Calculate summary statistics
    device_analysis = {}
    for device, stats in device_stats.items():
        analysis = {
            "total_patients": stats["total_patients"],
            "success_rate": stats["successful_patients"] / stats["total_patients"] if stats["total_patients"] > 0 else 0,
            "failed_patients": stats["failed_patients"]
        }
        
        # Calculate metric statistics
        analysis["metrics_summary"] = {}
        for metric, values in stats["metrics"].items():
            if values:
                analysis["metrics_summary"][metric] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        # Calculate view distribution percentages
        total_views = sum(stats["view_distribution"].values())
        analysis["view_distribution"] = {}
        for view, count in stats["view_distribution"].items():
            analysis["view_distribution"][view] = {
                "count": count,
                "percentage": count / total_views if total_views > 0 else 0
            }
        
        # Calculate quality indicators
        analysis["quality_summary"] = {}
        for indicator, values in stats["quality_indicators"].items():
            if values:
                analysis["quality_summary"][indicator] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values)
                }
        
        # Calculate processing statistics
        analysis["processing_summary"] = {}
        for key, values in stats["processing_info"].items():
            if values:
                analysis["processing_summary"][key] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values)
                }
        
        device_analysis[device] = analysis
    
    return device_analysis

def generate_recommendations(device_analysis):
    """
    Generate actionable recommendations based on device analysis.
    
    Args:
        device_analysis (dict): Device performance analysis
        
    Returns:
        dict: Recommendations per device
    """
    recommendations = {}
    
    for device, analysis in device_analysis.items():
        device_recommendations = []
        
        # Success rate recommendations
        if analysis["success_rate"] < 0.9:
            device_recommendations.append(f"Low success rate ({analysis['success_rate']:.1%}). Check DICOM file quality and acquisition settings.")
        
        # View distribution recommendations
        view_dist = analysis.get("view_distribution", {})
        total_views = sum(v["count"] for v in view_dist.values())
        
        if total_views > 0:
            # Check for poor quality views
            poor_quality_pct = view_dist.get("poor_quality", {}).get("percentage", 0)
            if poor_quality_pct > 0.2:
                device_recommendations.append(f"High poor quality rate ({poor_quality_pct:.1%}). Review image acquisition technique.")
            
            # Check for off-axis views
            off_axis_pct = view_dist.get("off_axis", {}).get("percentage", 0)
            if off_axis_pct > 0.15:
                device_recommendations.append(f"High off-axis rate ({off_axis_pct:.1%}). Focus on probe positioning and angulation.")
            
            # Check for view diversity
            standard_views = ["parasternal_long", "parasternal_short", "apical_4chamber", "apical_2chamber"]
            standard_view_count = sum(1 for view in standard_views if view in view_dist and view_dist[view]["count"] > 0)
            if standard_view_count < 3:
                device_recommendations.append("Limited view diversity. Ensure comprehensive echocardiographic examination.")
        
        # Quality indicators
        quality = analysis.get("quality_summary", {})
        if "visualization_rate" in quality:
            viz_rate = quality["visualization_rate"]["mean"]
            if viz_rate < 0.7:
                device_recommendations.append(f"Low visualization rate ({viz_rate:.1%}). Optimize gain, depth, and focus settings.")
        
        # Ejection fraction analysis
        ef_stats = analysis.get("metrics_summary", {}).get("ejection_fraction", {})
        if ef_stats:
            if ef_stats["std"] > 15:
                device_recommendations.append(f"High EF variability (std: {ef_stats['std']:.1f}%). Standardize acquisition protocol.")
            if ef_stats["count"] < analysis["total_patients"] * 0.8:
                device_recommendations.append("Low EF measurement success rate. Focus on optimal LV visualization.")
        
        # Processing efficiency
        processing = analysis.get("processing_summary", {})
        if "processed_frames" in processing:
            avg_frames = processing["processed_frames"]["mean"]
            if avg_frames < 20:
                device_recommendations.append(f"Low frame count ({avg_frames:.1f}). Consider longer acquisition times.")
        
        recommendations[device] = device_recommendations if device_recommendations else ["No specific recommendations - performance looks good!"]
    
    return recommendations

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run EchoPrime inference on individual patients.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing device folders")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory containing model weights")
    parser.add_argument("--output", type=str, default="results/inference_output_per_patient", help="Directory to save results")
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
    
    # Find all DICOM files across all device folders
    data_dir = Path(args.data_dir)
    all_dicom_files = []
    
    for device_folder in data_dir.iterdir():
        if device_folder.is_dir():
            device_name = device_folder.name
            
            # Find DICOM files in this device folder
            dicom_files = []
            for file_path in device_folder.iterdir():
                if file_path.is_file():
                    # Check if it's a DICOM file (by extension or content)
                    dicom_files.append((str(file_path), device_name))
            
            all_dicom_files.extend(dicom_files)
            print(f"Found {len(dicom_files)} DICOM files in {device_name}")
    
    if not all_dicom_files:
        print(f"No DICOM files found in {args.data_dir}")
        return
    
    print(f"\nTotal DICOM files to process: {len(all_dicom_files)}")
    
    # Process each DICOM file individually
    all_results = []
    device_folders = defaultdict(list)
    
    for dicom_path, device_name in tqdm(all_dicom_files, desc="Processing patients"):
        result = inference.process_patient(dicom_path, device_name)
        all_results.append(result)
        device_folders[device_name].append(result)
    
    # Save individual patient results by device
    for device_name, device_results in device_folders.items():
        device_output_dir = os.path.join(args.output, device_name)
        os.makedirs(device_output_dir, exist_ok=True)
        
        # Save individual patient files
        for result in device_results:
            patient_file = os.path.join(device_output_dir, f"patient_{result['patient_id']}.json")
            with open(patient_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
        
        # Save device summary
        device_summary = {
            "device": device_name,
            "total_patients": len(device_results),
            "successful": sum(1 for r in device_results if r["status"] == "success"),
            "failed": sum(1 for r in device_results if r["status"] == "failed"),
            "patients": device_results
        }
        
        with open(os.path.join(device_output_dir, "device_summary.json"), "w") as f:
            json.dump(device_summary, f, indent=2, default=str)
    
    # Perform device analysis
    print("\nAnalyzing device performance...")
    device_analysis = analyze_device_performance(all_results)
    
    # Generate recommendations
    recommendations = generate_recommendations(device_analysis)
    
    # Create comprehensive summary
    summary = {
        "total_patients": len(all_results),
        "successful": sum(1 for r in all_results if r["status"] == "success"),
        "failed": sum(1 for r in all_results if r["status"] == "failed"),
        "devices_analyzed": list(device_analysis.keys()),
        "device_analysis": device_analysis,
        "recommendations": recommendations
    }
    
    # Save comprehensive analysis
    with open(os.path.join(args.output, "comprehensive_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save simple summary for compatibility
    simple_summary = {
        "total_patients": len(all_results),
        "successful": sum(1 for r in all_results if r["status"] == "success"),
        "failed": sum(1 for r in all_results if r["status"] == "failed"),
        "results_by_device": {device: len(results) for device, results in device_folders.items()}
    }
    
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(simple_summary, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("ECHOPRIME PER-PATIENT INFERENCE SUMMARY")
    print("="*80)
    print(f"Total patients processed: {summary['total_patients']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print()
    
    # Print device-specific results
    for device_name, analysis in device_analysis.items():
        print(f"📱 {device_name}:")
        print(f"   Patients: {analysis['total_patients']}")
        print(f"   Success rate: {analysis['success_rate']:.1%}")
        
        # Print top metrics if available
        if "metrics_summary" in analysis and "ejection_fraction" in analysis["metrics_summary"]:
            ef_stats = analysis["metrics_summary"]["ejection_fraction"]
            print(f"   Avg EF: {ef_stats['mean']:.1f}% (±{ef_stats['std']:.1f})")
        
        # Print top view
        if "view_distribution" in analysis:
            top_view = max(analysis["view_distribution"].items(), key=lambda x: x[1]["count"], default=(None, None))
            if top_view[0]:
                print(f"   Top view: {top_view[0]} ({top_view[1]['percentage']:.1%})")
        
        print()
    
    # Print recommendations
    print("🔧 DEVICE OPTIMIZATION RECOMMENDATIONS:")
    print("="*80)
    for device, recs in recommendations.items():
        print(f"\n{device}:")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")
    
    print(f"\nDetailed analysis saved to: {args.output}/comprehensive_analysis.json")
    print(f"Individual patient results saved in device-specific folders")

if __name__ == "__main__":
    main()
