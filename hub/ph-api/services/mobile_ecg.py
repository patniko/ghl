import json
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from loguru import logger
import neurokit2 as nk


class ECGProcessor:
    """Class for processing mobile ECG data from AliveCor devices"""

    @staticmethod
    def load_ecg_data(file_path: str) -> Dict[str, Any]:
        """
        Load ECG data from an AliveCor JSON file

        Args:
            file_path: Path to the AliveCor JSON file

        Returns:
            Dictionary containing the parsed ECG data
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading ECG data: {str(e)}")
            raise ValueError(f"Failed to load ECG data: {str(e)}")

    @staticmethod
    def extract_metadata(ecg_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from AliveCor ECG data

        Args:
            ecg_data: Dictionary containing the ECG data

        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            "id": ecg_data.get("id"),
            "patientID": ecg_data.get("patientID"),
            "duration": ecg_data.get("duration"),
            "recordedAt": ecg_data.get("recordedAt"),
            "deviceInfo": ecg_data.get("deviceInfo", {}),
        }

        # Add raw data properties
        if "data" in ecg_data and "raw" in ecg_data["data"]:
            raw_data = ecg_data["data"]["raw"]
            metadata.update(
                {
                    "frequency": raw_data.get("frequency"),
                    "mainsFrequency": raw_data.get("mainsFrequency"),
                    "amplitudeResolution": raw_data.get("amplitudeResolution"),
                    "numLeads": raw_data.get("numLeads"),
                }
            )

        return metadata

    @staticmethod
    def extract_lead_data(
        ecg_data: Dict[str, Any], lead_name: str = "leadI"
    ) -> List[int]:
        """
        Extract data for a specific lead from AliveCor ECG data

        Args:
            ecg_data: Dictionary containing the ECG data
            lead_name: Name of the lead to extract (default: leadI)

        Returns:
            List of ECG samples for the specified lead
        """
        try:
            if (
                "data" in ecg_data
                and "raw" in ecg_data["data"]
                and "samples" in ecg_data["data"]["raw"]
            ):
                samples = ecg_data["data"]["raw"]["samples"]
                if lead_name in samples:
                    return samples[lead_name]
            return []
        except Exception as e:
            logger.error(f"Error extracting lead data: {str(e)}")
            return []

    @staticmethod
    def calculate_heart_rate(lead_data: List[int], sampling_rate: int = 300) -> float:
        """
        Calculate heart rate from ECG lead data using R-peak detection

        Args:
            lead_data: List of ECG samples
            sampling_rate: Sampling rate in Hz (default: 300)

        Returns:
            Estimated heart rate in BPM
        """
        if not lead_data or len(lead_data) < sampling_rate * 2:
            return 0.0

        # Convert to numpy array
        signal = np.array(lead_data)

        # Simple R-peak detection (threshold-based)
        # In a real implementation, more sophisticated algorithms would be used
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        threshold = signal_mean + 2 * signal_std

        # Find peaks above threshold
        peaks = []
        for i in range(1, len(signal) - 1):
            if (
                signal[i] > threshold
                and signal[i] > signal[i - 1]
                and signal[i] > signal[i + 1]
            ):
                peaks.append(i)

        # Calculate RR intervals and heart rate
        if len(peaks) < 2:
            return 60.0  # Default to 60 BPM if not enough peaks

        rr_intervals = np.diff(peaks) / sampling_rate  # in seconds
        mean_rr = np.mean(rr_intervals)

        if mean_rr == 0:
            return 0.0

        heart_rate = 60.0 / mean_rr  # in BPM

        return heart_rate

    @staticmethod
    def detect_arrhythmias(
        lead_data: List[int], sampling_rate: int = 300
    ) -> Dict[str, Any]:
        """
        Basic arrhythmia detection from ECG lead data

        Args:
            lead_data: List of ECG samples
            sampling_rate: Sampling rate in Hz (default: 300)

        Returns:
            Dictionary with arrhythmia detection results
        """
        # This is a simplified placeholder implementation
        # In a real application, this would use more sophisticated algorithms

        if not lead_data or len(lead_data) < sampling_rate * 5:
            return {"error": "Insufficient data for analysis"}

        # Convert to numpy array
        signal = np.array(lead_data)

        # Calculate heart rate
        heart_rate = ECGProcessor.calculate_heart_rate(lead_data, sampling_rate)

        # Simple arrhythmia detection based on heart rate
        # This is highly simplified and for demonstration purposes only
        results = {
            "heart_rate": heart_rate,
            "rhythm": "Normal",
            "confidence": 0.0,
            "abnormalities": [],
        }

        # Calculate RR interval variability (simplified)
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        threshold = signal_mean + 2 * signal_std

        # Find peaks above threshold
        peaks = []
        for i in range(1, len(signal) - 1):
            if (
                signal[i] > threshold
                and signal[i] > signal[i - 1]
                and signal[i] > signal[i + 1]
            ):
                peaks.append(i)

        # Check for irregular rhythm
        irregular_rhythm = False
        if len(peaks) >= 3:
            rr_intervals = np.diff(peaks) / sampling_rate
            rr_std = np.std(rr_intervals)

            # High RR interval variability might indicate arrhythmia
            if rr_std > 0.15:
                irregular_rhythm = True
                results["abnormalities"].append("Irregular rhythm")

        # Determine rhythm type - prioritize heart rate based conditions over irregularity
        if heart_rate < 60:
            results["rhythm"] = "Bradycardia"
            results["confidence"] = 0.7
            results["abnormalities"].append("Low heart rate")
        elif heart_rate > 100:
            results["rhythm"] = "Tachycardia"
            results["confidence"] = 0.7
            results["abnormalities"].append("Elevated heart rate")
        elif irregular_rhythm:
            results["rhythm"] = "Possible Arrhythmia"
            results["confidence"] = 0.6
        else:
            results["rhythm"] = "Normal"
            results["confidence"] = 0.8

        return results

    @staticmethod
    def analyze_ecg_with_neurokit(ecg_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive ECG analysis using neurokit2

        Args:
            ecg_data: Dictionary containing the ECG data

        Returns:
            Dictionary with detailed analysis results for ECGAnalysis table
        """
        # Extract metadata
        metadata = ECGProcessor.extract_metadata(ecg_data)
        sampling_rate = metadata.get("frequency", 300)

        # Extract lead data
        lead_i_data = ECGProcessor.extract_lead_data(ecg_data, "leadI")

        # Check if we have enough data
        if not lead_i_data or len(lead_i_data) < sampling_rate * 2:
            return {
                "has_missing_leads": True,
                "signal_noise_ratio": 0.0,
                "baseline_wander_score": 0.0,
                "motion_artifact_score": 0.0,
                "rr_interval_mean": 0.0,
                "rr_interval_stddev": 0.0,
                "rr_interval_consistency": 0.0,
                "qrs_count": 0,
                "qrs_detection_confidence": 0.0,
                "hrv_sdnn": 0.0,
                "hrv_rmssd": 0.0,
                "hrv_pnn50": 0.0,
                "hrv_lf": 0.0,
                "hrv_hf": 0.0,
                "hrv_lf_hf_ratio": 0.0,
                "frequency_peak": 0.0,
                "frequency_power_vlf": 0.0,
                "frequency_power_lf": 0.0,
                "frequency_power_hf": 0.0,
            }

        # Convert to numpy array
        signal = np.array(lead_i_data)

        try:
            # Clean the signal
            ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)

            # Find R-peaks
            r_peaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            r_peaks = r_peaks_info[0]["ECG_R_Peaks"]

            # Check if we have enough R-peaks for analysis
            if len(r_peaks) < 3:
                logger.warning(
                    f"Not enough R-peaks detected for detailed analysis: {len(r_peaks)}"
                )
                return {
                    "has_missing_leads": True,
                    "signal_noise_ratio": 0.0,
                    "baseline_wander_score": 0.0,
                    "motion_artifact_score": 0.0,
                    "rr_interval_mean": 0.0,
                    "rr_interval_stddev": 0.0,
                    "rr_interval_consistency": 0.0,
                    "qrs_count": len(r_peaks),
                    "qrs_detection_confidence": 0.0,
                    "hrv_sdnn": 0.0,
                    "hrv_rmssd": 0.0,
                    "hrv_pnn50": 0.0,
                    "hrv_lf": 0.0,
                    "hrv_hf": 0.0,
                    "hrv_lf_hf_ratio": 0.0,
                    "frequency_peak": 0.0,
                    "frequency_power_vlf": 0.0,
                    "frequency_power_lf": 0.0,
                    "frequency_power_hf": 0.0,
                }

            # Convert r_peaks to numpy array if it's not already
            if not isinstance(r_peaks, np.ndarray):
                r_peaks = np.array(r_peaks)

            # Skip delineation which is causing issues
            # waves_info = nk.ecg_delineate(ecg_cleaned, r_peaks, sampling_rate=sampling_rate)

            # Extract heart rate variability (HRV) metrics
            hrv_time = pd.DataFrame(
                {"HRV_SDNN": [0], "HRV_RMSSD": [0], "HRV_pNN50": [0]}
            )
            hrv_freq = pd.DataFrame({"HRV_LF": [0], "HRV_HF": [0], "HRV_LFHF": [0]})

            try:
                # Only attempt HRV analysis if we have enough R-peaks
                if len(r_peaks) >= 3:
                    hrv_time = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)
                    hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate)

                    # Check for LF/HF ratio column - it might have different names in different versions
                    if (
                        "HRV_LF/HF" in hrv_freq.columns
                        and "HRV_LFHF" not in hrv_freq.columns
                    ):
                        # Copy the column with the standardized name
                        hrv_freq["HRV_LFHF"] = hrv_freq["HRV_LF/HF"]
            except Exception as e:
                logger.error(f"Error in HRV analysis: {str(e)}")
                logger.error(traceback.format_exc())

            # Signal quality assessment
            has_missing_leads = len(lead_i_data) < (
                sampling_rate * 5
            )  # Less than 5 seconds of data

            # Calculate signal-to-noise ratio
            if len(ecg_cleaned) > 0:
                signal_power = np.mean(np.square(ecg_cleaned))
                noise = signal - ecg_cleaned
                noise_power = np.mean(np.square(noise))
                snr = (
                    10 * np.log10(signal_power / noise_power)
                    if noise_power > 0
                    else 100
                )
            else:
                snr = 0

            # Baseline wander assessment
            try:
                baseline = nk.signal_filter(
                    signal, lowcut=0.1, highcut=0.5, sampling_rate=sampling_rate
                )
                baseline_wander_score = (
                    np.std(baseline) / np.std(signal) if np.std(signal) > 0 else 0
                )
            except Exception as e:
                logger.error(f"Error in baseline wander assessment: {str(e)}")
                baseline_wander_score = 0

            # Motion artifact detection
            try:
                high_freq = nk.signal_filter(
                    signal, lowcut=20, highcut=None, sampling_rate=sampling_rate
                )
                motion_artifact_score = (
                    np.std(high_freq) / np.std(signal) if np.std(signal) > 0 else 0
                )
            except Exception as e:
                logger.error(f"Error in motion artifact detection: {str(e)}")
                motion_artifact_score = 0

            # R-R interval analysis
            try:
                rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # in ms
                rr_mean = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
                rr_std = np.std(rr_intervals) if len(rr_intervals) > 0 else 0
                rr_consistency = 1 - (rr_std / rr_mean) if rr_mean > 0 else 0
            except Exception as e:
                logger.error(f"Error in R-R interval analysis: {str(e)}")
                rr_mean = 0
                rr_std = 0
                rr_consistency = 0

            # Frequency analysis
            try:
                frequency_analysis = nk.signal_psd(
                    ecg_cleaned, sampling_rate=sampling_rate
                )

                # Handle the frequency analysis result which is a DataFrame with 'Frequency' and 'Power' columns
                if (
                    isinstance(frequency_analysis, tuple)
                    and len(frequency_analysis) >= 2
                ):
                    # Old style return format (tuple)
                    frequencies, power = frequency_analysis
                    max_power_idx = np.argmax(power)
                    peak_frequency = (
                        frequencies[max_power_idx]
                        if max_power_idx < len(frequencies)
                        else 0
                    )
                elif (
                    hasattr(frequency_analysis, "columns")
                    and "Frequency" in frequency_analysis.columns
                    and "Power" in frequency_analysis.columns
                ):
                    # New style return format (DataFrame)
                    max_power_idx = (
                        frequency_analysis["Power"].idxmax()
                        if not frequency_analysis.empty
                        else 0
                    )
                    peak_frequency = (
                        frequency_analysis.loc[max_power_idx, "Frequency"]
                        if max_power_idx in frequency_analysis.index
                        else 0
                    )
                else:
                    # Fallback
                    peak_frequency = 0
                    logger.warning("Unexpected format for frequency_analysis")

                # Get power in different frequency bands
                vlf_power_df = nk.signal_power(
                    ecg_cleaned,
                    frequency_band=[0.003, 0.04],
                    sampling_rate=sampling_rate,
                )
                lf_power_df = nk.signal_power(
                    ecg_cleaned,
                    frequency_band=[0.04, 0.15],
                    sampling_rate=sampling_rate,
                )
                hf_power_df = nk.signal_power(
                    ecg_cleaned, frequency_band=[0.15, 0.4], sampling_rate=sampling_rate
                )

                # Extract scalar values from the DataFrames
                vlf_power = (
                    vlf_power_df.iloc[0, 0]
                    if hasattr(vlf_power_df, "iloc") and vlf_power_df.shape[0] > 0
                    else 0
                )
                lf_power = (
                    lf_power_df.iloc[0, 0]
                    if hasattr(lf_power_df, "iloc") and lf_power_df.shape[0] > 0
                    else 0
                )
                hf_power = (
                    hf_power_df.iloc[0, 0]
                    if hasattr(hf_power_df, "iloc") and hf_power_df.shape[0] > 0
                    else 0
                )
            except Exception as e:
                logger.error(f"Error in frequency analysis: {str(e)}")
                logger.error(traceback.format_exc())
                peak_frequency = 0
                vlf_power = 0
                lf_power = 0
                hf_power = 0

            # Compile results
            analysis_results = {
                # Signal quality metrics
                "has_missing_leads": has_missing_leads,
                "signal_noise_ratio": float(snr),
                "baseline_wander_score": float(baseline_wander_score),
                "motion_artifact_score": float(motion_artifact_score),
                # R-R interval metrics
                "rr_interval_mean": float(rr_mean),
                "rr_interval_stddev": float(rr_std),
                "rr_interval_consistency": float(rr_consistency),
                # QRS detection
                "qrs_count": len(r_peaks),
                "qrs_detection_confidence": 1.0 if len(r_peaks) > 0 else 0.0,
                # HRV metrics
                "hrv_sdnn": float(hrv_time["HRV_SDNN"].iloc[0])
                if not hrv_time.empty and "HRV_SDNN" in hrv_time.columns
                else 0,
                "hrv_rmssd": float(hrv_time["HRV_RMSSD"].iloc[0])
                if not hrv_time.empty and "HRV_RMSSD" in hrv_time.columns
                else 0,
                "hrv_pnn50": float(hrv_time["HRV_pNN50"].iloc[0])
                if not hrv_time.empty and "HRV_pNN50" in hrv_time.columns
                else 0,
                "hrv_lf": float(hrv_freq["HRV_LF"].iloc[0])
                if not hrv_freq.empty and "HRV_LF" in hrv_freq.columns
                else 0,
                "hrv_hf": float(hrv_freq["HRV_HF"].iloc[0])
                if not hrv_freq.empty and "HRV_HF" in hrv_freq.columns
                else 0,
                "hrv_lf_hf_ratio": float(hrv_freq["HRV_LFHF"].iloc[0])
                if not hrv_freq.empty and "HRV_LFHF" in hrv_freq.columns
                else 0,
                # Frequency content analysis
                "frequency_peak": float(peak_frequency),
                "frequency_power_vlf": float(vlf_power),
                "frequency_power_lf": float(lf_power),
                "frequency_power_hf": float(hf_power),
            }

            return analysis_results

        except Exception as e:
            logger.error(f"Error in neurokit ECG analysis: {str(e)}")
            logger.error(traceback.format_exc())

            # Return default values in case of error
            return {
                "has_missing_leads": True,
                "signal_noise_ratio": 0.0,
                "baseline_wander_score": 0.0,
                "motion_artifact_score": 0.0,
                "rr_interval_mean": 0.0,
                "rr_interval_stddev": 0.0,
                "rr_interval_consistency": 0.0,
                "qrs_count": 0,
                "qrs_detection_confidence": 0.0,
                "hrv_sdnn": 0.0,
                "hrv_rmssd": 0.0,
                "hrv_pnn50": 0.0,
                "hrv_lf": 0.0,
                "hrv_hf": 0.0,
                "hrv_lf_hf_ratio": 0.0,
                "frequency_peak": 0.0,
                "frequency_power_vlf": 0.0,
                "frequency_power_lf": 0.0,
                "frequency_power_hf": 0.0,
            }

    @staticmethod
    def analyze_ecg(ecg_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on ECG data

        Args:
            ecg_data: Dictionary containing the ECG data

        Returns:
            Dictionary with analysis results
        """
        # Extract metadata
        metadata = ECGProcessor.extract_metadata(ecg_data)

        # Get sampling rate
        sampling_rate = metadata.get("frequency", 300)

        # Extract lead data
        lead_i_data = ECGProcessor.extract_lead_data(ecg_data, "leadI")
        lead_ii_data = ECGProcessor.extract_lead_data(ecg_data, "leadII")

        # Calculate heart rate from lead I
        heart_rate = ECGProcessor.calculate_heart_rate(lead_i_data, sampling_rate)

        # Detect arrhythmias
        arrhythmia_results = ECGProcessor.detect_arrhythmias(lead_i_data, sampling_rate)

        # Combine results
        results = {
            "metadata": metadata,
            "heart_rate": heart_rate,
            "analysis": {
                "rhythm": arrhythmia_results.get("rhythm", "Unknown"),
                "confidence": arrhythmia_results.get("confidence", 0.0),
                "abnormalities": arrhythmia_results.get("abnormalities", []),
            },
            "lead_stats": {
                "leadI": {
                    "samples": len(lead_i_data),
                    "min": min(lead_i_data) if lead_i_data else None,
                    "max": max(lead_i_data) if lead_i_data else None,
                    "mean": np.mean(lead_i_data) if lead_i_data else None,
                    "std": np.std(lead_i_data) if lead_i_data else None,
                },
                "leadII": {
                    "samples": len(lead_ii_data),
                    "min": min(lead_ii_data) if lead_ii_data else None,
                    "max": max(lead_ii_data) if lead_ii_data else None,
                    "mean": np.mean(lead_ii_data) if lead_ii_data else None,
                    "std": np.std(lead_ii_data) if lead_ii_data else None,
                },
            },
        }

        return results
