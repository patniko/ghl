#!/usr/bin/env python3
"""
Script to dump comprehensive DICOM metadata from ECG files.

This script extracts and saves all available metadata from DICOM files
including patient information, acquisition parameters, and technical details.

Usage:
    python scripts/dump_dicom_metadata.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import json
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm
from datetime import datetime


def extract_dicom_metadata(dicom_path):
    """
    Extract comprehensive metadata from a DICOM file.
    
    Args:
        dicom_path (str): Path to the DICOM file
        
    Returns:
        dict: Dictionary containing all extractable metadata
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        
        metadata = {
            'filename': os.path.basename(dicom_path),
            'file_size_bytes': os.path.getsize(dicom_path),
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # Basic DICOM information
        metadata.update({
            'sop_class_uid': getattr(ds, 'SOPClassUID', 'N/A'),
            'sop_instance_uid': getattr(ds, 'SOPInstanceUID', 'N/A'),
            'study_instance_uid': getattr(ds, 'StudyInstanceUID', 'N/A'),
            'series_instance_uid': getattr(ds, 'SeriesInstanceUID', 'N/A'),
            'modality': getattr(ds, 'Modality', 'N/A'),
            'transfer_syntax_uid': getattr(ds, 'file_meta', {}).get('TransferSyntaxUID', 'N/A')
        })
        
        # Patient information
        metadata.update({
            'patient_id': getattr(ds, 'PatientID', 'N/A'),
            'patient_name': str(getattr(ds, 'PatientName', 'N/A')),
            'patient_birth_date': getattr(ds, 'PatientBirthDate', 'N/A'),
            'patient_sex': getattr(ds, 'PatientSex', 'N/A'),
            'patient_age': getattr(ds, 'PatientAge', 'N/A'),
            'patient_weight': getattr(ds, 'PatientWeight', 'N/A'),
            'patient_size': getattr(ds, 'PatientSize', 'N/A')
        })
        
        # Study information
        metadata.update({
            'study_date': getattr(ds, 'StudyDate', 'N/A'),
            'study_time': getattr(ds, 'StudyTime', 'N/A'),
            'study_description': getattr(ds, 'StudyDescription', 'N/A'),
            'study_id': getattr(ds, 'StudyID', 'N/A'),
            'accession_number': getattr(ds, 'AccessionNumber', 'N/A')
        })
        
        # Series information
        metadata.update({
            'series_date': getattr(ds, 'SeriesDate', 'N/A'),
            'series_time': getattr(ds, 'SeriesTime', 'N/A'),
            'series_description': getattr(ds, 'SeriesDescription', 'N/A'),
            'series_number': getattr(ds, 'SeriesNumber', 'N/A'),
            'protocol_name': getattr(ds, 'ProtocolName', 'N/A')
        })
        
        # Equipment information
        metadata.update({
            'manufacturer': getattr(ds, 'Manufacturer', 'N/A'),
            'manufacturer_model_name': getattr(ds, 'ManufacturerModelName', 'N/A'),
            'device_serial_number': getattr(ds, 'DeviceSerialNumber', 'N/A'),
            'software_versions': getattr(ds, 'SoftwareVersions', 'N/A'),
            'institution_name': getattr(ds, 'InstitutionName', 'N/A'),
            'station_name': getattr(ds, 'StationName', 'N/A')
        })
        
        # ECG-specific information
        if hasattr(ds, 'WaveformSequence') and ds.WaveformSequence:
            waveform = ds.WaveformSequence[0]
            metadata.update({
                'sampling_frequency': getattr(waveform, 'SamplingFrequency', 'N/A'),
                'number_of_waveform_channels': getattr(waveform, 'NumberOfWaveformChannels', 'N/A'),
                'number_of_waveform_samples': getattr(waveform, 'NumberOfWaveformSamples', 'N/A'),
                'waveform_bits_allocated': getattr(waveform, 'WaveformBitsAllocated', 'N/A'),
                'waveform_sample_interpretation': getattr(waveform, 'WaveformSampleInterpretation', 'N/A')
            })
            
            # Channel information
            if hasattr(waveform, 'ChannelDefinitionSequence'):
                channels = []
                for i, channel in enumerate(waveform.ChannelDefinitionSequence):
                    channel_info = {
                        'channel_number': i + 1,
                        'channel_label': getattr(channel, 'ChannelLabel', f'Channel_{i+1}'),
                        'channel_status': getattr(channel, 'ChannelStatus', 'N/A'),
                        'channel_source_sequence': str(getattr(channel, 'ChannelSourceSequence', 'N/A')),
                        'channel_sensitivity': getattr(channel, 'ChannelSensitivity', 'N/A'),
                        'channel_sensitivity_units': getattr(channel, 'ChannelSensitivityUnitsSequence', 'N/A')
                    }
                    channels.append(channel_info)
                metadata['channels'] = channels
                metadata['total_channels'] = len(channels)
        
        # Image information (if present)
        metadata.update({
            'rows': getattr(ds, 'Rows', 'N/A'),
            'columns': getattr(ds, 'Columns', 'N/A'),
            'bits_allocated': getattr(ds, 'BitsAllocated', 'N/A'),
            'bits_stored': getattr(ds, 'BitsStored', 'N/A'),
            'high_bit': getattr(ds, 'HighBit', 'N/A'),
            'pixel_representation': getattr(ds, 'PixelRepresentation', 'N/A'),
            'photometric_interpretation': getattr(ds, 'PhotometricInterpretation', 'N/A')
        })
        
        # Additional technical parameters
        metadata.update({
            'acquisition_date': getattr(ds, 'AcquisitionDate', 'N/A'),
            'acquisition_time': getattr(ds, 'AcquisitionTime', 'N/A'),
            'content_date': getattr(ds, 'ContentDate', 'N/A'),
            'content_time': getattr(ds, 'ContentTime', 'N/A'),
            'instance_creation_date': getattr(ds, 'InstanceCreationDate', 'N/A'),
            'instance_creation_time': getattr(ds, 'InstanceCreationTime', 'N/A')
        })
        
        # Extract all other DICOM tags as a comprehensive dump
        all_tags = {}
        for elem in ds:
            if elem.tag != (0x7fe0, 0x0010):  # Skip pixel data
                try:
                    tag_name = elem.name if hasattr(elem, 'name') else str(elem.tag)
                    tag_value = str(elem.value) if hasattr(elem, 'value') else 'N/A'
                    # Limit very long values
                    if len(tag_value) > 1000:
                        tag_value = tag_value[:1000] + "... [truncated]"
                    all_tags[f"{elem.tag}_{tag_name}"] = tag_value
                except:
                    all_tags[f"{elem.tag}_unknown"] = "Error reading value"
        
        metadata['all_dicom_tags'] = all_tags
        
        return metadata
        
    except InvalidDicomError as e:
        return {
            'filename': os.path.basename(dicom_path),
            'error': f'Invalid DICOM file: {str(e)}',
            'extraction_timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'filename': os.path.basename(dicom_path),
            'error': f'Error reading DICOM: {str(e)}',
            'extraction_timestamp': datetime.now().isoformat()
        }


def process_dicom_files(input_dir, output_dir):
    """
    Process all DICOM files in the input directory and extract metadata.
    
    Args:
        input_dir (str): Directory containing DICOM files
        output_dir (str): Directory to save metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all DICOM files
    dicom_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.dcm', '.dicom')) or not '.' in file:
            dicom_files.append(os.path.join(input_dir, file))
    
    if not dicom_files:
        print(f"No DICOM files found in {input_dir}")
        return
    
    print(f"Found {len(dicom_files)} DICOM files")
    
    all_metadata = []
    
    for dicom_path in tqdm(dicom_files, desc="Extracting metadata"):
        print(f"\nProcessing: {os.path.basename(dicom_path)}")
        
        metadata = extract_dicom_metadata(dicom_path)
        all_metadata.append(metadata)
        
        # Save individual metadata file
        base_name = os.path.splitext(os.path.basename(dicom_path))[0]
        individual_json_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        with open(individual_json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"  Saved metadata to: {individual_json_path}")
        
        # Print key information
        if 'error' not in metadata:
            print(f"  Modality: {metadata.get('modality', 'N/A')}")
            print(f"  Manufacturer: {metadata.get('manufacturer', 'N/A')}")
            print(f"  Model: {metadata.get('manufacturer_model_name', 'N/A')}")
            print(f"  Sampling Frequency: {metadata.get('sampling_frequency', 'N/A')} Hz")
            print(f"  Channels: {metadata.get('number_of_waveform_channels', 'N/A')}")
            print(f"  Samples: {metadata.get('number_of_waveform_samples', 'N/A')}")
        else:
            print(f"  Error: {metadata['error']}")
    
    # Save combined metadata
    combined_json_path = os.path.join(output_dir, 'all_metadata.json')
    with open(combined_json_path, 'w') as f:
        json.dump(all_metadata, f, indent=2, default=str)
    
    # Create a summary CSV
    summary_data = []
    for metadata in all_metadata:
        if 'error' not in metadata:
            summary_data.append({
                'filename': metadata.get('filename', 'N/A'),
                'modality': metadata.get('modality', 'N/A'),
                'manufacturer': metadata.get('manufacturer', 'N/A'),
                'model': metadata.get('manufacturer_model_name', 'N/A'),
                'patient_id': metadata.get('patient_id', 'N/A'),
                'study_date': metadata.get('study_date', 'N/A'),
                'sampling_frequency': metadata.get('sampling_frequency', 'N/A'),
                'channels': metadata.get('number_of_waveform_channels', 'N/A'),
                'samples': metadata.get('number_of_waveform_samples', 'N/A'),
                'file_size_mb': round(metadata.get('file_size_bytes', 0) / (1024*1024), 2)
            })
        else:
            summary_data.append({
                'filename': metadata.get('filename', 'N/A'),
                'error': metadata.get('error', 'N/A')
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, 'metadata_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"\nSummary:")
    print(f"  Combined metadata saved to: {combined_json_path}")
    print(f"  Summary CSV saved to: {summary_csv_path}")
    print(f"  Individual metadata files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract comprehensive metadata from DICOM ECG files.')
    parser.add_argument('--input_dir', type=str, default='data/12L',
                        help='Directory containing DICOM files (default: data/12L)')
    parser.add_argument('--output_dir', type=str, default='data/12L/metadata',
                        help='Directory to save metadata (default: data/12L/metadata)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    print(f"Extracting metadata from DICOM files in {args.input_dir}")
    print(f"Metadata will be saved to {args.output_dir}")
    
    process_dicom_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
