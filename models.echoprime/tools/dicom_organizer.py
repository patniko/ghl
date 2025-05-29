#!/usr/bin/env python3
"""
DICOM File Organizer Tool

This script reorganizes DICOM files from the data/ directory structure into
patient-specific directories with the format: devicename-patientid

Original structure: data/device/file1, data/device/file2, ...
New structure: data_organized/devicename-patientid/file1, data_organized/devicename-patientid/file2, ...
"""

import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict
import pydicom
from pydicom.errors import InvalidDicomError
import json


def read_dicom_patient_info(file_path):
    """
    Read DICOM file and extract patient ID and device information.
    
    Args:
        file_path (str): Path to the DICOM file
        
    Returns:
        tuple: (patient_id, device_name) or (None, None) if error
    """
    try:
        ds = pydicom.dcmread(file_path, force=True)
        patient_id = getattr(ds, 'PatientID', 'Unknown')
        return patient_id
        
    except InvalidDicomError:
        print(f"Warning: {file_path} is not a valid DICOM file")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None


def organize_dicom_files(source_dir="data", target_dir="data_organized", copy_files=True):
    """
    Organize DICOM files into patient-specific directories.
    
    Args:
        source_dir (str): Source directory containing DICOM files
        target_dir (str): Target directory for organized files
        copy_files (bool): If True, copy files; if False, move files
        
    Returns:
        dict: Organization results and statistics
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist")
        return None
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Track organization statistics
    stats = {
        'total_files_processed': 0,
        'files_organized': 0,
        'files_failed': 0,
        'patients_created': 0,
        'devices_processed': set(),
        'patient_directories': [],
        'failed_files': []
    }
    
    # Track patient directories created
    patient_dirs_created = set()
    
    print(f"Organizing DICOM files from {source_dir} to {target_dir}")
    print("=" * 60)
    
    # Walk through all subdirectories in source
    for root, dirs, files in os.walk(source_dir):
        if not files:
            continue
            
        # Determine device name from directory structure
        device_name = os.path.basename(root)
        
        # If we're processing the source directory itself (not a subdirectory),
        # use the directory name as the device name
        if root == source_dir:
            device_name = os.path.basename(source_dir)
        
        # Skip if this is the parent data directory with no device name
        if device_name == "data" and root != source_dir:
            continue
            
        print(f"Processing device: {device_name}")
        stats['devices_processed'].add(device_name)
        
        for file in files:
            if file.startswith('.'):  # Skip hidden files
                continue
                
            source_file_path = os.path.join(root, file)
            stats['total_files_processed'] += 1
            
            # Read patient ID from DICOM file
            patient_id = read_dicom_patient_info(source_file_path)
            
            if patient_id is None:
                stats['files_failed'] += 1
                stats['failed_files'].append(source_file_path)
                continue
            
            # Create patient directory name
            patient_dir_name = f"{device_name}-{patient_id}"
            patient_dir_path = os.path.join(target_dir, patient_dir_name)
            
            # Create patient directory if it doesn't exist
            if patient_dir_name not in patient_dirs_created:
                os.makedirs(patient_dir_path, exist_ok=True)
                patient_dirs_created.add(patient_dir_name)
                stats['patient_directories'].append(patient_dir_name)
                print(f"  Created directory: {patient_dir_name}")
            
            # Determine target file path
            target_file_path = os.path.join(patient_dir_path, file)
            
            # Copy or move the file
            try:
                if copy_files:
                    shutil.copy2(source_file_path, target_file_path)
                else:
                    shutil.move(source_file_path, target_file_path)
                
                stats['files_organized'] += 1
                
            except Exception as e:
                print(f"  Error processing {source_file_path}: {str(e)}")
                stats['files_failed'] += 1
                stats['failed_files'].append(source_file_path)
    
    # Update final statistics
    stats['patients_created'] = len(patient_dirs_created)
    stats['devices_processed'] = list(stats['devices_processed'])
    
    return stats


def print_organization_summary(stats):
    """
    Print a formatted summary of the organization results.
    
    Args:
        stats (dict): Organization statistics
    """
    if not stats:
        return
    
    print("\n" + "=" * 60)
    print("DICOM FILE ORGANIZATION SUMMARY")
    print("=" * 60)
    
    print(f"\nFILE STATISTICS:")
    print(f"  Total files processed: {stats['total_files_processed']}")
    print(f"  Files successfully organized: {stats['files_organized']}")
    print(f"  Files failed: {stats['files_failed']}")
    
    print(f"\nORGANIZATION STATISTICS:")
    print(f"  Devices processed: {len(stats['devices_processed'])}")
    print(f"  Patient directories created: {stats['patients_created']}")
    
    print(f"\nDEVICES PROCESSED:")
    for device in sorted(stats['devices_processed']):
        print(f"  - {device}")
    
    if stats['failed_files']:
        print(f"\nFAILED FILES ({len(stats['failed_files'])}):")
        for failed_file in stats['failed_files'][:10]:  # Show first 10
            print(f"  - {failed_file}")
        if len(stats['failed_files']) > 10:
            print(f"  ... and {len(stats['failed_files']) - 10} more")
    
    print(f"\nPATIENT DIRECTORIES CREATED ({len(stats['patient_directories'])}):")
    for patient_dir in sorted(stats['patient_directories'])[:20]:  # Show first 20
        print(f"  - {patient_dir}")
    if len(stats['patient_directories']) > 20:
        print(f"  ... and {len(stats['patient_directories']) - 20} more")


def save_organization_log(stats, log_file="dicom_organization_log.json"):
    """
    Save organization results to a JSON log file.
    
    Args:
        stats (dict): Organization statistics
        log_file (str): Output log file path
    """
    if stats:
        with open(log_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\nOrganization log saved to: {log_file}")


def verify_organization(target_dir="data_organized"):
    """
    Verify the organization by checking the created directory structure.
    
    Args:
        target_dir (str): Target directory to verify
        
    Returns:
        dict: Verification results
    """
    if not os.path.exists(target_dir):
        print(f"Target directory {target_dir} does not exist")
        return None
    
    verification = {
        'patient_directories': [],
        'files_per_patient': {},
        'total_files': 0
    }
    
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):
            verification['patient_directories'].append(item)
            
            # Count files in this patient directory
            file_count = len([f for f in os.listdir(item_path) 
                            if os.path.isfile(os.path.join(item_path, f)) and not f.startswith('.')])
            verification['files_per_patient'][item] = file_count
            verification['total_files'] += file_count
    
    return verification


def main():
    """Main function to run the DICOM organization."""
    # Parse command line arguments
    source_dir = "data"
    target_dir = "data_organized"
    copy_files = True  # Default to copy (safer)
    
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
    if len(sys.argv) > 2:
        target_dir = sys.argv[2]
    if len(sys.argv) > 3:
        copy_files = sys.argv[3].lower() in ['true', '1', 'yes', 'copy']
    
    print("DICOM File Organizer")
    print("=" * 30)
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Operation: {'Copy' if copy_files else 'Move'} files")
    print()
    
    # Ask for confirmation if moving files
    if not copy_files:
        response = input("WARNING: This will MOVE files from the source directory. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return
    
    # Run organization
    stats = organize_dicom_files(source_dir, target_dir, copy_files)
    
    if stats:
        # Print summary
        print_organization_summary(stats)
        
        # Save log
        save_organization_log(stats)
        
        # Verify organization
        print("\nVerifying organization...")
        verification = verify_organization(target_dir)
        if verification:
            print(f"Verification complete:")
            print(f"  Patient directories: {len(verification['patient_directories'])}")
            print(f"  Total files organized: {verification['total_files']}")
        
        print("\nOrganization complete!")
    else:
        print("Organization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
