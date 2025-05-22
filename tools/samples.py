#!/usr/bin/env python3
"""
samples.py - Tool for generating sample data for the GHL project

This tool can generate sample data for various types of data used in the GHL project,
including questionnaire data, blood results, mobile measures, consent data, echo data,
and ECG data. It can use existing samples in the sample folder or generate its own data
dynamically.
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import csv
import datetime
import uuid
from pathlib import Path
import shutil
import zipfile
from typing import List, Dict, Tuple, Optional, Union, Any

# Optional imports - these might not be installed
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. Echo sample processing will be limited.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not installed. Image processing will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not installed. Some processing features will be limited.")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define constants
SAMPLE_DIR = Path(__file__).parent.parent / "samples"
OUTPUT_DIR = SAMPLE_DIR

# Define data schemas based on pipeline.md
QUESTIONNAIRE_SCHEMA = {
    "Questionnaire Identifier": str,
    "Patient ID": str,
    "Age": int,
    "Biological Sex": str,
    "Location": str,
    "Setting": str,
    "Smoking Years": int,
    "Daily Tobacco": int,
    "Alcohol Consumption": int
}

BLOOD_RESULTS_SCHEMA = {
    "lab_result_id": str,
    "patient_id": str,
    "collection_date": str,
    "collection_time": str,
    "lab_facility": str,
    "ordering_provider": str,
    "result_status": str,
    "ldl": float,
    "hdl": float,
    "triglycerides": float,
    "total_cholesterol": float,
    "hba1c": float,
    "creatinine": float,
    "ldl_flag": str,
    "hdl_flag": str,
    "triglycerides_flag": str,
    "total_cholesterol_flag": str,
    "hba1c_flag": str,
    "creatinine_flag": str,
    "fasting_status": str,
    "specimen_type": str,
    "laboratory_method_ldl": str,
    "laboratory_method_hdl": str,
    "laboratory_method_triglycerides": str,
    "laboratory_method_total_cholesterol": str,
    "laboratory_method_hba1c": str,
    "laboratory_method_creatinine": str
}

MOBILE_MEASURES_SCHEMA = {
    "Measure Identifier": str,
    "Patient ID": str,
    "Measurement Date": str,
    "Measurement Time": str,
    "Blood Pressure (Systolic)": int,
    "Blood Pressure (Diastolic)": int,
    "Pulse Rate": int,
    "SpO2": int,
    "Respiratory Rate": int,
    "Weight": float,
    "Height": float
}

CONSENT_SCHEMA = {
    "Patient Identifier": str
}

# Helper functions for data generation
def generate_patient_id() -> str:
    """Generate a unique patient identifier."""
    return f"P{uuid.uuid4().hex[:8].upper()}"

def generate_date(start_date="2024-01-01", end_date="2025-01-01") -> str:
    """Generate a random date between start_date and end_date."""
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_date = start + datetime.timedelta(days=random_days)
    return random_date.strftime("%Y-%m-%d")

def generate_time() -> str:
    """Generate a random time in HH:MM format."""
    hour = random.randint(8, 17)  # 8 AM to 5 PM
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"

def get_flag_for_value(value: float, normal_range: Tuple[float, float]) -> str:
    """Get the appropriate flag for a value based on its normal range."""
    if value < normal_range[0] * 0.8:
        return "CRITICAL_LOW"
    elif value < normal_range[0]:
        return "LOW"
    elif value > normal_range[1] * 1.2:
        return "CRITICAL_HIGH"
    elif value > normal_range[1]:
        return "HIGH"
    else:
        return "NORMAL"

# Data generation functions
def generate_questionnaire_data(num_samples: int = 10, patient_ids: List[str] = None) -> pd.DataFrame:
    """
    Generate sample questionnaire data.
    
    Args:
        num_samples: Number of samples to generate
        patient_ids: List of patient IDs to use (if None, will generate new IDs)
        
    Returns:
        DataFrame containing the generated questionnaire data
    """
    if patient_ids is None:
        patient_ids = [generate_patient_id() for _ in range(num_samples)]
    
    data = []
    for i, patient_id in enumerate(patient_ids[:num_samples]):
        questionnaire_id = f"Q{uuid.uuid4().hex[:8].upper()}"
        age = random.randint(30, 99)
        sex = random.choice(["Female", "Male", "Other"])
        location = random.choice(["Urban-LCECU Hospital", "CHAD Hospital", "CMC Cardiology"])
        setting = random.choice(["Outpatient", "Inpatient"])
        
        # Generate smoking and alcohol data with some correlation to age
        smoking_years = 0
        daily_tobacco = 0
        alcohol_consumption = 0
        
        # 30% chance of being a smoker
        if random.random() < 0.3:
            smoking_years = random.randint(1, max(1, age - 18))  # Started after 18
            daily_tobacco = random.randint(1, 50)
        
        # 40% chance of consuming alcohol
        if random.random() < 0.4:
            alcohol_consumption = random.randint(1, 10)
        
        data.append({
            "Questionnaire Identifier": questionnaire_id,
            "Patient ID": patient_id,
            "Age": age,
            "Biological Sex": sex,
            "Location": location,
            "Setting": setting,
            "Smoking Years": smoking_years,
            "Daily Tobacco": daily_tobacco,
            "Alcohol Consumption": alcohol_consumption
        })
    
    return pd.DataFrame(data)

def generate_blood_results(num_samples: int = 10, patient_ids: List[str] = None) -> pd.DataFrame:
    """
    Generate sample blood results data.
    
    Args:
        num_samples: Number of samples to generate
        patient_ids: List of patient IDs to use (if None, will generate new IDs)
        
    Returns:
        DataFrame containing the generated blood results data
    """
    if patient_ids is None:
        patient_ids = [generate_patient_id() for _ in range(num_samples)]
    
    # Define normal ranges for lab values
    normal_ranges = {
        "ldl": (50, 130),
        "hdl": (40, 60),
        "triglycerides": (50, 150),
        "total_cholesterol": (125, 200),
        "hba1c": (4.0, 5.7),
        "creatinine": (0.6, 1.2)
    }
    
    # Define lab facilities and methods
    lab_facilities = ["Central Lab", "Regional Medical Center", "University Hospital Lab"]
    
    ldl_methods = ["Direct Measurement", "Friedewald Equation", "Martin-Hopkins Equation"]
    hdl_methods = ["Precipitation Method", "Direct Enzymatic Method", "Ultracentrifugation"]
    triglycerides_methods = ["Enzymatic Method", "Colorimetric Method", "GPO-PAP Method"]
    cholesterol_methods = ["Enzymatic Method", "Colorimetric Method", "CHOD-PAP Method"]
    hba1c_methods = ["HPLC", "Immunoassay", "Boronate Affinity Chromatography"]
    creatinine_methods = ["Jaffe Method", "Enzymatic Method", "Mass Spectrometry"]
    
    specimen_types = ["SERUM", "PLASMA", "WHOLE_BLOOD", "URINE"]
    fasting_statuses = ["FASTING", "NON_FASTING", "UNKNOWN"]
    result_statuses = ["FINAL", "PRELIMINARY", "CORRECTED", "CANCELLED"]
    
    data = []
    for i, patient_id in enumerate(patient_ids[:num_samples]):
        # Generate 1-3 lab results per patient
        num_results = random.randint(1, 3)
        
        for j in range(num_results):
            lab_result_id = f"LR{uuid.uuid4().hex[:8].upper()}"
            collection_date = generate_date()
            collection_time = generate_time()
            lab_facility = random.choice(lab_facilities)
            ordering_provider = f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson'])}"
            result_status = random.choice(result_statuses)
            
            # Generate lab values with some abnormal results
            ldl = round(random.uniform(normal_ranges["ldl"][0] * 0.7, normal_ranges["ldl"][1] * 1.3), 1)
            hdl = round(random.uniform(normal_ranges["hdl"][0] * 0.7, normal_ranges["hdl"][1] * 1.3), 1)
            triglycerides = round(random.uniform(normal_ranges["triglycerides"][0] * 0.7, normal_ranges["triglycerides"][1] * 1.3), 1)
            total_cholesterol = round(random.uniform(normal_ranges["total_cholesterol"][0] * 0.7, normal_ranges["total_cholesterol"][1] * 1.3), 1)
            hba1c = round(random.uniform(normal_ranges["hba1c"][0] * 0.7, normal_ranges["hba1c"][1] * 1.3), 1)
            creatinine = round(random.uniform(normal_ranges["creatinine"][0] * 0.7, normal_ranges["creatinine"][1] * 1.3), 2)
            
            # Generate flags based on values
            ldl_flag = get_flag_for_value(ldl, normal_ranges["ldl"])
            hdl_flag = get_flag_for_value(hdl, normal_ranges["hdl"])
            triglycerides_flag = get_flag_for_value(triglycerides, normal_ranges["triglycerides"])
            total_cholesterol_flag = get_flag_for_value(total_cholesterol, normal_ranges["total_cholesterol"])
            hba1c_flag = get_flag_for_value(hba1c, normal_ranges["hba1c"])
            creatinine_flag = get_flag_for_value(creatinine, normal_ranges["creatinine"])
            
            fasting_status = random.choice(fasting_statuses)
            specimen_type = random.choice(specimen_types)
            
            data.append({
                "lab_result_id": lab_result_id,
                "patient_id": patient_id,
                "collection_date": collection_date,
                "collection_time": collection_time,
                "lab_facility": lab_facility,
                "ordering_provider": ordering_provider,
                "result_status": result_status,
                "ldl": ldl,
                "hdl": hdl,
                "triglycerides": triglycerides,
                "total_cholesterol": total_cholesterol,
                "hba1c": hba1c,
                "creatinine": creatinine,
                "ldl_flag": ldl_flag,
                "hdl_flag": hdl_flag,
                "triglycerides_flag": triglycerides_flag,
                "total_cholesterol_flag": total_cholesterol_flag,
                "hba1c_flag": hba1c_flag,
                "creatinine_flag": creatinine_flag,
                "fasting_status": fasting_status,
                "specimen_type": specimen_type,
                "laboratory_method_ldl": random.choice(ldl_methods),
                "laboratory_method_hdl": random.choice(hdl_methods),
                "laboratory_method_triglycerides": random.choice(triglycerides_methods),
                "laboratory_method_total_cholesterol": random.choice(cholesterol_methods),
                "laboratory_method_hba1c": random.choice(hba1c_methods),
                "laboratory_method_creatinine": random.choice(creatinine_methods)
            })
    
    return pd.DataFrame(data)

def generate_mobile_measures(num_samples: int = 10, patient_ids: List[str] = None) -> pd.DataFrame:
    """
    Generate sample mobile measures data.
    
    Args:
        num_samples: Number of samples to generate
        patient_ids: List of patient IDs to use (if None, will generate new IDs)
        
    Returns:
        DataFrame containing the generated mobile measures data
    """
    if patient_ids is None:
        patient_ids = [generate_patient_id() for _ in range(num_samples)]
    
    data = []
    for i, patient_id in enumerate(patient_ids[:num_samples]):
        # Generate 1-5 measurements per patient
        num_measurements = random.randint(1, 5)
        
        # Generate base values for this patient to ensure consistency
        base_systolic = random.randint(100, 140)
        base_diastolic = random.randint(60, 90)
        base_pulse = random.randint(60, 90)
        base_spo2 = random.randint(95, 99)
        base_resp_rate = random.randint(12, 18)
        base_weight = round(random.uniform(50.0, 90.0), 1)
        base_height = round(random.uniform(150.0, 185.0), 1)
        
        # Generate measurements with small variations
        for j in range(num_measurements):
            measure_id = f"MM{uuid.uuid4().hex[:8].upper()}"
            measurement_date = generate_date()
            measurement_time = generate_time()
            
            # Add small variations to base values
            systolic = max(80, min(150, base_systolic + random.randint(-5, 5)))
            diastolic = max(50, min(110, base_diastolic + random.randint(-5, 5)))
            pulse = max(50, min(100, base_pulse + random.randint(-5, 5)))
            spo2 = max(90, min(100, base_spo2 + random.randint(-2, 1)))
            resp_rate = max(8, min(20, base_resp_rate + random.randint(-2, 2)))
            weight = round(max(40.0, min(100.0, base_weight + random.uniform(-0.5, 0.5))), 1)
            height = base_height  # Height typically doesn't change
            
            data.append({
                "Measure Identifier": measure_id,
                "Patient ID": patient_id,
                "Measurement Date": measurement_date,
                "Measurement Time": measurement_time,
                "Blood Pressure (Systolic)": systolic,
                "Blood Pressure (Diastolic)": diastolic,
                "Pulse Rate": pulse,
                "SpO2": spo2,
                "Respiratory Rate": resp_rate,
                "Weight": weight,
                "Height": height
            })
    
    return pd.DataFrame(data)

def generate_consent_data(num_samples: int = 10, patient_ids: List[str] = None) -> pd.DataFrame:
    """
    Generate sample consent data.
    
    Args:
        num_samples: Number of samples to generate
        patient_ids: List of patient IDs to use (if None, will generate new IDs)
        
    Returns:
        DataFrame containing the generated consent data
    """
    if patient_ids is None:
        patient_ids = [generate_patient_id() for _ in range(num_samples)]
    
    data = []
    for patient_id in patient_ids[:num_samples]:
        data.append({
            "Patient Identifier": patient_id
        })
    
    return pd.DataFrame(data)

def process_echo_samples(output_dir: Path = None) -> List[str]:
    """
    Process echo samples from the samples directory.
    
    Args:
        output_dir: Directory to save processed samples (if None, will use samples/echo)
        
    Returns:
        List of paths to the processed echo samples
    """
    echo_dir = SAMPLE_DIR / "echo"
    if output_dir is None or output_dir == echo_dir:
        # If no output directory is specified or it's the same as the input directory,
        # just process the files in place
        output_dir = echo_dir
        copy_files = False
    else:
        # Otherwise, copy the files to the output directory
        copy_files = True
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of DICOM files
    dicom_files = list(echo_dir.glob("*.dcm"))
    
    processed_files = []
    for dicom_file in dicom_files:
        try:
            # If pydicom is available, read the DICOM file
            if PYDICOM_AVAILABLE:
                ds = pydicom.dcmread(dicom_file)
                # In a real scenario, we would process the DICOM file here
            
            # Copy the file to the output directory if needed
            if copy_files:
                output_file = output_dir / dicom_file.name
                shutil.copy(dicom_file, output_file)
            else:
                output_file = dicom_file
            
            processed_files.append(str(output_file))
        except Exception as e:
            print(f"Error processing {dicom_file}: {e}")
    
    return processed_files

def process_ecg_samples(output_dir: Path = None) -> List[str]:
    """
    Process ECG samples from the samples directory.
    
    Args:
        output_dir: Directory to save processed samples (if None, will use samples/ecg/normalized)
        
    Returns:
        List of paths to the processed ECG samples
    """
    ecg_dir = SAMPLE_DIR / "ecg" / "normalized"
    if output_dir is None or output_dir == ecg_dir:
        # If no output directory is specified or it's the same as the input directory,
        # just process the files in place
        output_dir = ecg_dir
        copy_files = False
    else:
        # Otherwise, copy the files to the output directory
        copy_files = True
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of NPY files
    npy_files = list(ecg_dir.glob("*.npy"))
    
    processed_files = []
    for npy_file in npy_files:
        try:
            # Load the ECG data
            ecg_data = np.load(npy_file)
            
            # If torch is available, we could apply transformations here
            if TORCH_AVAILABLE:
                # In a real scenario, we would process the ECG data here
                # For example, using the preprocess_ecg function from fixed_hubert_ecg.py
                pass
            
            # Copy the file to the output directory if needed
            if copy_files:
                output_file = output_dir / npy_file.name
                shutil.copy(npy_file, output_file)
            else:
                output_file = npy_file
            
            processed_files.append(str(output_file))
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")
    
    return processed_files

def export_to_csv(data: pd.DataFrame, output_file: Path) -> str:
    """
    Export data to a CSV file.
    
    Args:
        data: DataFrame containing the data to export
        output_file: Path to the output file
        
    Returns:
        Path to the exported CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_file.parent, exist_ok=True)
    
    # Export data to CSV
    data.to_csv(output_file, index=False)
    
    return str(output_file)

def create_zip_archive(directory: Path, output_file: Path = None) -> str:
    """
    Create a zip archive of the specified directory.
    
    Args:
        directory: Directory to zip
        output_file: Path to the output zip file (if None, will use directory name + .zip)
        
    Returns:
        Path to the created zip file
    """
    if output_file is None:
        output_file = directory.parent / f"{directory.name}.zip"
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(directory.parent)
                zipf.write(file_path, arcname)
    
    return str(output_file)

def main():
    """Main function for the samples.py tool."""
    parser = argparse.ArgumentParser(description="Generate sample data for the GHL project")
    parser.add_argument("--type", choices=["questionnaire", "blood", "mobile", "consent", "echo", "ecg", "all"],
                        default="all", help="Type of data to generate")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save generated samples")
    parser.add_argument("--use-existing", action="store_true", help="Use existing samples when available")
    parser.add_argument("--create-zip", action="store_true", help="Create a zip archive of the generated samples")
    parser.add_argument("--zip-file", type=str, default=None, help="Path to the output zip file (if --create-zip is specified)")
    parser.add_argument("--include-partials", action="store_true", 
                        help="Include partial records to simulate real-world data (some patients will be missing certain data types)")
    parser.add_argument("--partial-rate", type=float, default=0.3, 
                        help="Percentage of patients with partial records (0.0-1.0, default: 0.3)")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    # Generate patient IDs to use across all data types
    patient_ids = [generate_patient_id() for _ in range(args.num_samples)]
    
    # If including partial records, determine which patients will have which data types
    if args.include_partials and args.partial_rate > 0:
        # Ensure at least one patient has complete records
        complete_patients = max(1, int(args.num_samples * (1 - args.partial_rate)))
        partial_patients = args.num_samples - complete_patients
        
        # For partial patients, randomly decide which data types they'll have
        patient_data_types = {}
        for i, patient_id in enumerate(patient_ids):
            if i < complete_patients:
                # Complete patients have all data types
                patient_data_types[patient_id] = ["questionnaire", "blood", "mobile", "consent", "echo", "ecg"]
            else:
                # Partial patients have a random subset of data types (but always include consent)
                available_types = ["questionnaire", "blood", "mobile", "echo", "ecg"]
                # Select a random number of data types (at least 1, at most all)
                num_types = random.randint(1, len(available_types))
                selected_types = random.sample(available_types, num_types)
                # Always include consent data
                selected_types.append("consent")
                patient_data_types[patient_id] = selected_types
    else:
        # All patients have all data types
        patient_data_types = {patient_id: ["questionnaire", "blood", "mobile", "consent", "echo", "ecg"] 
                             for patient_id in patient_ids}
    
    # Create CSV output directory
    csv_dir = output_dir / "csv"
    os.makedirs(csv_dir, exist_ok=True)
    
    # Generate data based on type and patient data types
    if args.type in ["questionnaire", "all"]:
        # Filter patient IDs based on data types
        questionnaire_patient_ids = [pid for pid in patient_ids if "questionnaire" in patient_data_types[pid]]
        if questionnaire_patient_ids:
            questionnaire_data = generate_questionnaire_data(len(questionnaire_patient_ids), questionnaire_patient_ids)
            questionnaire_file = csv_dir / "questionnaire.csv"
            export_to_csv(questionnaire_data, questionnaire_file)
            print(f"Generated {len(questionnaire_data)} questionnaire samples: {questionnaire_file}")
        else:
            print("No patients selected for questionnaire data")
    
    if args.type in ["blood", "all"]:
        # Filter patient IDs based on data types
        blood_patient_ids = [pid for pid in patient_ids if "blood" in patient_data_types[pid]]
        if blood_patient_ids:
            blood_data = generate_blood_results(len(blood_patient_ids), blood_patient_ids)
            blood_file = csv_dir / "blood_results.csv"
            export_to_csv(blood_data, blood_file)
            print(f"Generated {len(blood_data)} blood result samples: {blood_file}")
        else:
            print("No patients selected for blood data")
    
    if args.type in ["mobile", "all"]:
        # Filter patient IDs based on data types
        mobile_patient_ids = [pid for pid in patient_ids if "mobile" in patient_data_types[pid]]
        if mobile_patient_ids:
            mobile_data = generate_mobile_measures(len(mobile_patient_ids), mobile_patient_ids)
            mobile_file = csv_dir / "mobile_measures.csv"
            export_to_csv(mobile_data, mobile_file)
            print(f"Generated {len(mobile_data)} mobile measure samples: {mobile_file}")
        else:
            print("No patients selected for mobile data")
    
    if args.type in ["consent", "all"]:
        # Filter patient IDs based on data types (all patients should have consent)
        consent_patient_ids = [pid for pid in patient_ids if "consent" in patient_data_types[pid]]
        if consent_patient_ids:
            consent_data = generate_consent_data(len(consent_patient_ids), consent_patient_ids)
            consent_file = csv_dir / "consent.csv"
            export_to_csv(consent_data, consent_file)
            print(f"Generated {len(consent_data)} consent samples: {consent_file}")
        else:
            print("No patients selected for consent data")
    
    if args.type in ["echo", "all"]:
        # Filter patient IDs based on data types
        echo_patient_ids = [pid for pid in patient_ids if "echo" in patient_data_types[pid]]
        if echo_patient_ids:
            echo_output_dir = output_dir / "echo" if args.output_dir else None
            echo_files = process_echo_samples(echo_output_dir)
            print(f"Processed {len(echo_files)} echo samples")
        else:
            print("No patients selected for echo data")
    
    if args.type in ["ecg", "all"]:
        # Filter patient IDs based on data types
        ecg_patient_ids = [pid for pid in patient_ids if "ecg" in patient_data_types[pid]]
        if ecg_patient_ids:
            ecg_output_dir = output_dir / "ecg" / "normalized" if args.output_dir else None
            ecg_files = process_ecg_samples(ecg_output_dir)
            print(f"Processed {len(ecg_files)} ECG samples")
        else:
            print("No patients selected for ECG data")
    
    # Create zip archive if requested
    if args.create_zip:
        zip_file = Path(args.zip_file) if args.zip_file else None
        zip_path = create_zip_archive(output_dir, zip_file)
        print(f"Created zip archive: {zip_path}")

if __name__ == "__main__":
    main()
