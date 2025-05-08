import json
import pytest
from unittest.mock import patch
import pathlib

from services.mobile_ecg import ECGProcessor

# Path to sample AliveCor files
SAMPLES_DIR = pathlib.Path(__file__).parent.parent / "samples"
ALIVECOR_SAMPLES = [SAMPLES_DIR / f"alivecor-{i}.json" for i in range(1, 11)]


# Load a sample file for testing
def load_sample_ecg_data(sample_index=1):
    """Load a sample AliveCor ECG file for testing"""
    sample_path = SAMPLES_DIR / f"alivecor-{sample_index}.json"
    try:
        with open(sample_path, "r") as f:
            return json.load(f)
    except Exception as e:
        pytest.skip(f"Could not load sample ECG data: {str(e)}")


class TestECGProcessor:
    """Test the ECGProcessor class using real AliveCor sample files"""

    def test_extract_metadata(self):
        """Test extracting metadata from ECG data"""
        # Load sample data
        sample_data = load_sample_ecg_data(1)

        # Extract metadata
        metadata = ECGProcessor.extract_metadata(sample_data)

        # Verify basic metadata fields
        assert "id" in metadata
        assert "patientID" in metadata
        assert "duration" in metadata
        assert "recordedAt" in metadata
        assert "deviceInfo" in metadata

        # Verify raw data properties if available
        if "frequency" in metadata:
            assert metadata["frequency"] == 300  # AliveCor typically uses 300Hz
        if "mainsFrequency" in metadata:
            assert metadata["mainsFrequency"] in [50, 60]  # Depends on region

    def test_extract_lead_data(self):
        """Test extracting lead data from ECG data"""
        # Load sample data
        sample_data = load_sample_ecg_data(1)

        # Extract lead data
        lead_i_data = ECGProcessor.extract_lead_data(sample_data, "leadI")

        # Verify lead data
        assert len(lead_i_data) > 0
        assert isinstance(lead_i_data, list)
        assert all(isinstance(x, (int, float)) for x in lead_i_data)

    def test_calculate_heart_rate(self):
        """Test heart rate calculation from ECG data"""
        # Create a simple ECG signal with clear R peaks
        # Simulate a heart rate of 60 BPM (1 beat per second)
        sampling_rate = 300  # 300 Hz
        duration = 10  # 10 seconds
        signal = []

        for i in range(duration * sampling_rate):
            # Add baseline
            value = 3000

            # Add R peaks every second (60 BPM)
            if i % sampling_rate < 10:  # 10 samples wide peak
                value += 1000  # R peak amplitude

            signal.append(value)

        heart_rate = ECGProcessor.calculate_heart_rate(signal, sampling_rate)

        # Allow some tolerance due to the simple peak detection algorithm
        assert 55 <= heart_rate <= 65

        # Also test with real data
        sample_data = load_sample_ecg_data(1)
        lead_i_data = ECGProcessor.extract_lead_data(sample_data, "leadI")

        if lead_i_data:
            heart_rate = ECGProcessor.calculate_heart_rate(lead_i_data, 300)
            # Just verify it returns a reasonable heart rate (40-200 BPM)
            assert 40 <= heart_rate <= 200 or heart_rate == 60.0  # 60 is the default

    def test_detect_arrhythmias(self):
        """Test arrhythmia detection with real data"""
        # Load sample data
        sample_data = load_sample_ecg_data(1)
        lead_i_data = ECGProcessor.extract_lead_data(sample_data, "leadI")

        if lead_i_data:
            # Test normal rhythm detection
            results = ECGProcessor.detect_arrhythmias(lead_i_data, 300)

            assert "rhythm" in results
            assert "confidence" in results
            assert "abnormalities" in results
            assert results["rhythm"] in [
                "Normal",
                "Bradycardia",
                "Tachycardia",
                "Possible Arrhythmia",
            ]

            # Test bradycardia detection (mocked)
            with patch.object(ECGProcessor, "calculate_heart_rate", return_value=50.0):
                brady_results = ECGProcessor.detect_arrhythmias(lead_i_data, 300)
                assert brady_results["rhythm"] == "Bradycardia"
                assert "Low heart rate" in brady_results["abnormalities"]

            # Test tachycardia detection (mocked)
            with patch.object(ECGProcessor, "calculate_heart_rate", return_value=120.0):
                tachy_results = ECGProcessor.detect_arrhythmias(lead_i_data, 300)
                assert tachy_results["rhythm"] == "Tachycardia"
                assert "Elevated heart rate" in tachy_results["abnormalities"]

    def test_analyze_ecg(self):
        """Test comprehensive ECG analysis with real data"""
        # Test with multiple sample files
        for i in range(1, 4):  # Test first 3 samples
            sample_data = load_sample_ecg_data(i)
            results = ECGProcessor.analyze_ecg(sample_data)

            # Verify result structure
            assert "metadata" in results
            assert "heart_rate" in results
            assert "analysis" in results
            assert "lead_stats" in results

            # Verify lead stats
            if "leadI" in results["lead_stats"]:
                lead_stats = results["lead_stats"]["leadI"]
                assert "samples" in lead_stats
                assert "min" in lead_stats
                assert "max" in lead_stats
                assert "mean" in lead_stats
                assert "std" in lead_stats

            # Verify analysis results
            assert "rhythm" in results["analysis"]
            assert "confidence" in results["analysis"]
            assert "abnormalities" in results["analysis"]
