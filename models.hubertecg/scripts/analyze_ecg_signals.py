#!/usr/bin/env python3
"""
Script to analyze ECG signals and extract basic clinical details.

This script:
1. Reads JSON files from the raw_data directory (AliveCor format)
2. Extracts basic ECG parameters and signal quality metrics
3. Performs heart rate variability analysis
4. Saves analysis results as JSON files in the results directory

Usage:
    python scripts/analyze_ecg_signals.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False
    print("Warning: neurokit2 not available. Some advanced analysis features will be disabled.")


class ECGProcessor:
    """ECG signal processor for extracting clinical parameters."""
    
    def __init__(self, sampling_rate=300):
        self.sampling_rate = sampling_rate
    
    def extract_ecg_from_json(self, json_path, data_type='enhanced'):
        """Extract ECG data from AliveCor JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is an AliveCor file
            if 'data' not in data or data_type not in data.get('data', {}):
                return None, None
            
            ecg_data_section = data['data'][data_type]
            sampling_rate = ecg_data_section.get('frequency', 300)
            amplitude_resolution = ecg_data_section.get('amplitudeResolution', 500)
            
            # Extract lead data
            samples = ecg_data_section.get('samples', {})
            lead_names = ['leadI', 'leadII', 'leadIII', 'AVR', 'AVL', 'AVF']
            
            ecg_leads = {}
            for lead_name in lead_names:
                if lead_name in samples and samples[lead_name]:
                    lead_data = np.array(samples[lead_name], dtype=np.float64)
                    # Apply amplitude resolution scaling
                    lead_data = lead_data / amplitude_resolution
                    ecg_leads[lead_name] = lead_data
            
            return ecg_leads, sampling_rate
            
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            return None, None
    
    def calculate_heart_rate(self, ecg_signal, sampling_rate):
        """Calculate heart rate from ECG signal."""
        try:
            if HAS_NEUROKIT:
                # Use neurokit2 for R-peak detection
                signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
                rpeaks = info['ECG_R_Peaks']
                
                if len(rpeaks) > 1:
                    # Calculate RR intervals
                    rr_intervals = np.diff(rpeaks) / sampling_rate  # in seconds
                    heart_rate = 60 / np.mean(rr_intervals)  # beats per minute
                    return heart_rate, rr_intervals, rpeaks
                else:
                    return None, None, None
            else:
                # Simple peak detection fallback
                # Find peaks using scipy
                peaks, _ = signal.find_peaks(ecg_signal, height=np.std(ecg_signal), distance=int(0.6*sampling_rate))
                
                if len(peaks) > 1:
                    rr_intervals = np.diff(peaks) / sampling_rate
                    heart_rate = 60 / np.mean(rr_intervals)
                    return heart_rate, rr_intervals, peaks
                else:
                    return None, None, None
                    
        except Exception as e:
            print(f"Error calculating heart rate: {e}")
            return None, None, None
    
    def calculate_signal_quality(self, ecg_signal, sampling_rate):
        """Calculate signal quality metrics."""
        try:
            # Signal-to-noise ratio estimation
            # Use high-frequency content as noise estimate
            b, a = signal.butter(4, [1, 40], btype='band', fs=sampling_rate)
            filtered_signal = signal.filtfilt(b, a, ecg_signal)
            
            noise = ecg_signal - filtered_signal
            signal_power = np.var(filtered_signal)
            noise_power = np.var(noise)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = float('inf')
            
            # Baseline wander estimation (low frequency content)
            b_low, a_low = signal.butter(4, 0.5, btype='low', fs=sampling_rate)
            baseline = signal.filtfilt(b_low, a_low, ecg_signal)
            baseline_wander = np.std(baseline)
            
            # Motion artifact estimation (high frequency content)
            b_high, a_high = signal.butter(4, 40, btype='high', fs=sampling_rate)
            high_freq = signal.filtfilt(b_high, a_high, ecg_signal)
            motion_artifact = np.std(high_freq)
            
            return {
                'signal_noise_ratio': float(snr),
                'baseline_wander_score': float(baseline_wander),
                'motion_artifact_score': float(motion_artifact)
            }
            
        except Exception as e:
            print(f"Error calculating signal quality: {e}")
            return {
                'signal_noise_ratio': None,
                'baseline_wander_score': None,
                'motion_artifact_score': None
            }
    
    def calculate_hrv_metrics(self, rr_intervals):
        """Calculate heart rate variability metrics."""
        try:
            if rr_intervals is None or len(rr_intervals) < 2:
                return {}
            
            # Convert to milliseconds
            rr_ms = rr_intervals * 1000
            
            # Time domain metrics
            sdnn = np.std(rr_ms)  # Standard deviation of NN intervals
            rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))  # Root mean square of successive differences
            
            # pNN50: percentage of successive RR intervals that differ by more than 50ms
            diff_rr = np.abs(np.diff(rr_ms))
            pnn50 = (np.sum(diff_rr > 50) / len(diff_rr)) * 100
            
            hrv_metrics = {
                'hrv_sdnn': float(sdnn),
                'hrv_rmssd': float(rmssd),
                'hrv_pnn50': float(pnn50),
                'rr_interval_mean': float(np.mean(rr_ms)),
                'rr_interval_stddev': float(np.std(rr_ms)),
                'rr_interval_consistency': float(1.0 / (1.0 + np.std(rr_ms) / np.mean(rr_ms)))
            }
            
            # Frequency domain metrics (if enough data)
            if len(rr_intervals) > 10:
                try:
                    # Interpolate RR intervals for frequency analysis
                    time_rr = np.cumsum(rr_intervals)
                    time_interp = np.arange(0, time_rr[-1], 1.0)  # 1 Hz interpolation
                    rr_interp = np.interp(time_interp, time_rr[:-1], rr_ms)
                    
                    # Power spectral density
                    freqs, psd = signal.welch(rr_interp, fs=1.0, nperseg=min(256, len(rr_interp)//2))
                    
                    # Frequency bands
                    vlf_band = (freqs >= 0.003) & (freqs < 0.04)
                    lf_band = (freqs >= 0.04) & (freqs < 0.15)
                    hf_band = (freqs >= 0.15) & (freqs < 0.4)
                    
                    vlf_power = np.trapz(psd[vlf_band], freqs[vlf_band])
                    lf_power = np.trapz(psd[lf_band], freqs[lf_band])
                    hf_power = np.trapz(psd[hf_band], freqs[hf_band])
                    
                    hrv_metrics.update({
                        'hrv_lf': float(lf_power),
                        'hrv_hf': float(hf_power),
                        'hrv_lf_hf_ratio': float(lf_power / hf_power) if hf_power > 0 else None,
                        'frequency_peak': float(freqs[np.argmax(psd)]),
                        'frequency_power_vlf': float(vlf_power),
                        'frequency_power_lf': float(lf_power),
                        'frequency_power_hf': float(hf_power)
                    })
                except Exception as e:
                    print(f"Error in frequency domain analysis: {e}")
            
            return hrv_metrics
            
        except Exception as e:
            print(f"Error calculating HRV metrics: {e}")
            return {}
    
    def analyze_rhythm(self, rr_intervals, heart_rate):
        """Basic rhythm analysis."""
        try:
            if rr_intervals is None or len(rr_intervals) < 3:
                return "insufficient_data"
            
            # Convert to milliseconds
            rr_ms = rr_intervals * 1000
            
            # Check for regular rhythm
            rr_variability = np.std(rr_ms) / np.mean(rr_ms)
            
            if rr_variability < 0.1:
                rhythm = "regular"
            elif rr_variability < 0.2:
                rhythm = "slightly_irregular"
            else:
                rhythm = "irregular"
            
            # Basic rate classification
            if heart_rate is not None:
                if heart_rate < 60:
                    rate_class = "bradycardia"
                elif heart_rate > 100:
                    rate_class = "tachycardia"
                else:
                    rate_class = "normal_rate"
            else:
                rate_class = "unknown"
            
            return f"{rhythm}_{rate_class}"
            
        except Exception as e:
            print(f"Error analyzing rhythm: {e}")
            return "analysis_error"
    
    def detect_abnormalities(self, ecg_leads, heart_rate, rr_intervals):
        """Detect basic ECG abnormalities."""
        abnormalities = []
        
        try:
            # Heart rate abnormalities
            if heart_rate is not None:
                if heart_rate < 50:
                    abnormalities.append("severe_bradycardia")
                elif heart_rate < 60:
                    abnormalities.append("bradycardia")
                elif heart_rate > 120:
                    abnormalities.append("severe_tachycardia")
                elif heart_rate > 100:
                    abnormalities.append("tachycardia")
            
            # RR interval variability
            if rr_intervals is not None and len(rr_intervals) > 2:
                rr_ms = rr_intervals * 1000
                rr_variability = np.std(rr_ms) / np.mean(rr_ms)
                
                if rr_variability > 0.3:
                    abnormalities.append("high_rr_variability")
                elif rr_variability < 0.05:
                    abnormalities.append("low_rr_variability")
            
            # Signal quality issues
            for lead_name, lead_data in ecg_leads.items():
                if len(lead_data) > 0:
                    # Check for flat line
                    if np.std(lead_data) < 0.01:
                        abnormalities.append(f"flat_line_{lead_name}")
                    
                    # Check for extreme values
                    z_scores = np.abs(zscore(lead_data))
                    if np.any(z_scores > 5):
                        abnormalities.append(f"extreme_values_{lead_name}")
            
        except Exception as e:
            print(f"Error detecting abnormalities: {e}")
            abnormalities.append("analysis_error")
        
        return abnormalities
    
    def analyze_ecg(self, json_data_or_path):
        """Main analysis function."""
        try:
            # If it's a file path, load the JSON
            if isinstance(json_data_or_path, str):
                ecg_leads, sampling_rate = self.extract_ecg_from_json(json_data_or_path)
            else:
                # Assume it's already loaded JSON data
                ecg_leads, sampling_rate = self.extract_ecg_from_json_data(json_data_or_path)
            
            if ecg_leads is None:
                return {"error": "Could not extract ECG data"}
            
            # Use Lead II for primary analysis (most common for rhythm analysis)
            primary_lead = None
            for lead_name in ['leadII', 'leadI', 'leadIII']:
                if lead_name in ecg_leads:
                    primary_lead = ecg_leads[lead_name]
                    break
            
            if primary_lead is None:
                return {"error": "No suitable lead found for analysis"}
            
            # Calculate heart rate and RR intervals
            heart_rate, rr_intervals, peaks = self.calculate_heart_rate(primary_lead, sampling_rate)
            
            # Calculate signal quality
            signal_quality = self.calculate_signal_quality(primary_lead, sampling_rate)
            
            # Calculate HRV metrics
            hrv_metrics = self.calculate_hrv_metrics(rr_intervals)
            
            # Analyze rhythm
            rhythm = self.analyze_rhythm(rr_intervals, heart_rate)
            
            # Detect abnormalities
            abnormalities = self.detect_abnormalities(ecg_leads, heart_rate, rr_intervals)
            
            # Compile results
            results = {
                "heart_rate": heart_rate,
                "qrs_count": len(peaks) if peaks is not None else None,
                "qrs_detection_confidence": 0.8 if peaks is not None and len(peaks) > 0 else 0.0,
                "has_missing_leads": len(ecg_leads) < 6,
                "available_leads": list(ecg_leads.keys()),
                "analysis": {
                    "rhythm": rhythm,
                    "abnormalities": abnormalities
                },
                "signal_quality": signal_quality,
                "hrv_metrics": hrv_metrics,
                "sampling_rate": sampling_rate,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}


def process_patient_json_files(patient_dir, output_dir):
    """Process all JSON files in a patient directory."""
    patient_name = os.path.basename(patient_dir)
    print(f"\nAnalyzing ECG signals for patient: {patient_name}")
    
    # Create patient-specific output directory
    patient_output_dir = os.path.join(output_dir, patient_name)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = []
    for file in os.listdir(patient_dir):
        if file.lower().endswith('.json'):
            json_files.append(os.path.join(patient_dir, file))
    
    if not json_files:
        print(f"  No JSON files found in {patient_dir}")
        return []
    
    print(f"  Found {len(json_files)} JSON files")
    
    processor = ECGProcessor()
    results = []
    
    for json_path in json_files:
        try:
            print(f"    Analyzing: {os.path.basename(json_path)}")
            
            # Analyze ECG
            analysis_results = processor.analyze_ecg(json_path)
            
            # Save individual analysis results
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            analysis_file = os.path.join(patient_output_dir, f"{base_name}_analysis.json")
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # Store summary for CSV
            summary = {
                'patient': patient_name,
                'filename': os.path.basename(json_path),
                'heart_rate': analysis_results.get('heart_rate'),
                'rhythm': analysis_results.get('analysis', {}).get('rhythm'),
                'abnormalities_count': len(analysis_results.get('analysis', {}).get('abnormalities', [])),
                'signal_noise_ratio': analysis_results.get('signal_quality', {}).get('signal_noise_ratio'),
                'has_missing_leads': analysis_results.get('has_missing_leads'),
                'qrs_count': analysis_results.get('qrs_count'),
                'hrv_sdnn': analysis_results.get('hrv_metrics', {}).get('hrv_sdnn'),
                'analysis_file': f"{patient_name}/{base_name}_analysis.json",
                'status': 'success' if 'error' not in analysis_results else 'error'
            }
            
            results.append(summary)
            print(f"      Heart rate: {analysis_results.get('heart_rate', 'N/A')} bpm")
            print(f"      Rhythm: {analysis_results.get('analysis', {}).get('rhythm', 'N/A')}")
            print(f"      Abnormalities: {len(analysis_results.get('analysis', {}).get('abnormalities', []))}")
            
        except Exception as e:
            print(f"      Error analyzing {os.path.basename(json_path)}: {e}")
            results.append({
                'patient': patient_name,
                'filename': os.path.basename(json_path),
                'heart_rate': None,
                'rhythm': None,
                'abnormalities_count': None,
                'signal_noise_ratio': None,
                'has_missing_leads': None,
                'qrs_count': None,
                'hrv_sdnn': None,
                'analysis_file': None,
                'status': f'error: {str(e)}'
            })
    
    successful = len([r for r in results if r['status'] == 'success'])
    print(f"  Patient {patient_name}: {successful}/{len(json_files)} files analyzed successfully")
    return results


def process_all_patients(input_dir, output_dir):
    """Process all patient directories containing JSON files."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find AliveCor patient directories
    alivecor_dirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and item.startswith('PATIENT-ALIVECOR'):
            alivecor_dirs.append(item_path)
    
    if not alivecor_dirs:
        print(f"No AliveCor patient directories found in {input_dir}")
        return
    
    alivecor_dirs.sort()
    print(f"Found {len(alivecor_dirs)} AliveCor directories: {[os.path.basename(d) for d in alivecor_dirs]}")
    
    all_results = []
    total_successful = 0
    total_files = 0
    
    for patient_dir in alivecor_dirs:
        patient_results = process_patient_json_files(patient_dir, output_dir)
        all_results.extend(patient_results)
        
        successful = len([r for r in patient_results if r['status'] == 'success'])
        total_successful += successful
        total_files += len(patient_results)
    
    # Save combined results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv_path = os.path.join(output_dir, 'ecg_analysis_summary.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nAnalysis summary saved to: {results_csv_path}")
    
    # Print summary
    print(f"\n=== ECG ANALYSIS SUMMARY ===")
    print(f"Total patients processed: {len(alivecor_dirs)}")
    print(f"Total JSON files: {total_files}")
    print(f"Successfully analyzed: {total_successful}")
    print(f"Success rate: {total_successful/total_files*100:.1f}%" if total_files > 0 else "No files to process")


def main():
    parser = argparse.ArgumentParser(description='Analyze ECG signals and extract clinical parameters.')
    parser.add_argument('--input_dir', type=str, default='raw_data',
                        help='Directory containing patient folders with JSON files (default: raw_data)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save analysis results (default: results)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    print(f"Analyzing ECG signals from {args.input_dir}")
    print(f"Results will be saved to {args.output_dir}")
    
    if not HAS_NEUROKIT:
        print("\nNote: neurokit2 is not installed. Using basic signal processing methods.")
        print("For more advanced ECG analysis, install neurokit2: pip install neurokit2")
    
    process_all_patients(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
