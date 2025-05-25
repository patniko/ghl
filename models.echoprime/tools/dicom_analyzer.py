#!/usr/bin/env python3
"""
DICOM File Analyzer Tool

This script analyzes DICOM files in the data/ directory and its subdirectories,
extracting metadata to provide a summary of:
- Total number of files
- Number of files per device
- Number of unique patients across all devices
- Number of unique patients per device
"""

import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import pydicom
from pydicom.errors import InvalidDicomError
import json


def read_dicom_metadata(file_path):
    """
    Read DICOM file and extract relevant metadata.
    
    Args:
        file_path (str): Path to the DICOM file
        
    Returns:
        dict: Dictionary containing patient ID, device info, and other metadata
    """
    try:
        ds = pydicom.dcmread(file_path, force=True)
        
        # Extract patient ID
        patient_id = getattr(ds, 'PatientID', 'Unknown')
        
        # Extract device/manufacturer information
        manufacturer = getattr(ds, 'Manufacturer', 'Unknown')
        manufacturer_model = getattr(ds, 'ManufacturerModelName', 'Unknown')
        station_name = getattr(ds, 'StationName', 'Unknown')
        
        # Extract study information
        study_instance_uid = getattr(ds, 'StudyInstanceUID', 'Unknown')
        series_instance_uid = getattr(ds, 'SeriesInstanceUID', 'Unknown')
        
        # Extract additional metadata
        study_date = getattr(ds, 'StudyDate', 'Unknown')
        modality = getattr(ds, 'Modality', 'Unknown')
        
        return {
            'patient_id': patient_id,
            'manufacturer': manufacturer,
            'manufacturer_model': manufacturer_model,
            'station_name': station_name,
            'study_instance_uid': study_instance_uid,
            'series_instance_uid': series_instance_uid,
            'study_date': study_date,
            'modality': modality,
            'file_path': file_path
        }
        
    except InvalidDicomError:
        print(f"Warning: {file_path} is not a valid DICOM file")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None


def analyze_dicom_directory(data_dir="data"):
    """
    Analyze all DICOM files in the specified directory and subdirectories.
    
    Args:
        data_dir (str): Path to the data directory
        
    Returns:
        dict: Analysis results
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return None
    
    # Initialize counters and collections
    total_files = 0
    valid_dicom_files = 0
    files_per_device = defaultdict(int)
    patients_per_device = defaultdict(set)
    all_patients = set()
    device_info = defaultdict(list)
    
    print(f"Scanning directory: {data_dir}")
    print("=" * 50)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_dir):
        if not files:
            continue
            
        # Determine device name from directory structure
        device_name = os.path.basename(root)
        if device_name == "data":
            continue
            
        print(f"Processing device: {device_name}")
        
        for file in files:
            if file.startswith('.'):  # Skip hidden files
                continue
                
            file_path = os.path.join(root, file)
            total_files += 1
            
            # Read DICOM metadata
            metadata = read_dicom_metadata(file_path)
            
            if metadata:
                valid_dicom_files += 1
                patient_id = metadata['patient_id']
                
                # Count files per device
                files_per_device[device_name] += 1
                
                # Track patients per device
                patients_per_device[device_name].add(patient_id)
                
                # Track all unique patients
                all_patients.add(patient_id)
                
                # Store device information
                device_info[device_name].append({
                    'manufacturer': metadata['manufacturer'],
                    'model': metadata['manufacturer_model'],
                    'station_name': metadata['station_name']
                })
    
    # Calculate summary statistics
    unique_patients_total = len(all_patients)
    unique_patients_per_device = {device: len(patients) 
                                 for device, patients in patients_per_device.items()}
    
    # Get unique device information per device
    unique_device_info = {}
    for device, info_list in device_info.items():
        # Get most common manufacturer and model for each device
        manufacturers = [info['manufacturer'] for info in info_list if info['manufacturer'] != 'Unknown']
        models = [info['model'] for info in info_list if info['model'] != 'Unknown']
        stations = [info['station_name'] for info in info_list if info['station_name'] != 'Unknown']
        
        unique_device_info[device] = {
            'manufacturer': Counter(manufacturers).most_common(1)[0][0] if manufacturers else 'Unknown',
            'model': Counter(models).most_common(1)[0][0] if models else 'Unknown',
            'station_name': Counter(stations).most_common(1)[0][0] if stations else 'Unknown'
        }
    
    return {
        'summary': {
            'total_files_scanned': total_files,
            'valid_dicom_files': valid_dicom_files,
            'total_devices': len(files_per_device),
            'unique_patients_total': unique_patients_total
        },
        'files_per_device': dict(files_per_device),
        'unique_patients_per_device': unique_patients_per_device,
        'device_information': unique_device_info,
        'patient_lists_per_device': {device: list(patients) 
                                   for device, patients in patients_per_device.items()}
    }


def print_summary(results):
    """
    Print a formatted summary of the analysis results.
    
    Args:
        results (dict): Analysis results from analyze_dicom_directory
    """
    if not results:
        return
    
    summary = results['summary']
    files_per_device = results['files_per_device']
    patients_per_device = results['unique_patients_per_device']
    device_info = results['device_information']
    
    print("\n" + "=" * 60)
    print("DICOM FILE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total files scanned: {summary['total_files_scanned']}")
    print(f"  Valid DICOM files: {summary['valid_dicom_files']}")
    print(f"  Total devices: {summary['total_devices']}")
    print(f"  Unique patients (across all devices): {summary['unique_patients_total']}")
    
    print(f"\nFILES PER DEVICE:")
    for device, count in sorted(files_per_device.items()):
        print(f"  {device}: {count} files")
    
    print(f"\nUNIQUE PATIENTS PER DEVICE:")
    for device, count in sorted(patients_per_device.items()):
        print(f"  {device}: {count} patients")
    
    print(f"\nDEVICE INFORMATION:")
    for device, info in sorted(device_info.items()):
        print(f"  {device}:")
        print(f"    Manufacturer: {info['manufacturer']}")
        print(f"    Model: {info['model']}")
        print(f"    Station Name: {info['station_name']}")


def save_results_to_json(results, output_file="dicom_analysis_results.json"):
    """
    Save analysis results to a JSON file.
    
    Args:
        results (dict): Analysis results
        output_file (str): Output file path
    """
    if results:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main function to run the DICOM analysis."""
    # Set default data directory
    data_dir = "data"
    
    # Allow command line argument for data directory
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    print("DICOM File Analyzer")
    print("=" * 30)
    
    # Run analysis
    results = analyze_dicom_directory(data_dir)
    
    if results:
        # Print summary
        print_summary(results)
        
        # Save detailed results
        save_results_to_json(results)
        
        print("\nAnalysis complete!")
    else:
        print("Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
