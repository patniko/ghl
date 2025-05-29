#!/usr/bin/env python3
"""
DICOM File Unpacker Tool

This script unpacks DICOM files from the raw_data directory, extracting:
- Image data (as PNG files)
- Video data (as MP4 files for multi-frame DICOM)
- Metadata (as JSON files)
- Audio data (if present)

The script creates an organized directory structure with extracted content.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import pydicom
from pydicom.errors import InvalidDicomError
import cv2
from PIL import Image
import argparse


def extract_dicom_content(dicom_file_path, output_dir):
    """
    Extract content from a single DICOM file.
    
    Args:
        dicom_file_path (str): Path to the DICOM file
        output_dir (str): Output directory for extracted content
        
    Returns:
        dict: Information about extracted content
    """
    try:
        # Read DICOM file
        ds = pydicom.dcmread(dicom_file_path, force=True)
        
        # Extract metadata
        metadata = extract_metadata(ds)
        
        # Create output subdirectory based on patient and study
        patient_id = getattr(ds, 'PatientID', 'Unknown')
        study_uid = getattr(ds, 'StudyInstanceUID', 'Unknown')
        series_uid = getattr(ds, 'SeriesInstanceUID', 'Unknown')
        
        # Create a clean directory name
        safe_patient_id = "".join(c for c in patient_id if c.isalnum() or c in ('-', '_'))[:50]
        safe_study_uid = study_uid.split('.')[-1][:20]  # Use last part of UID
        safe_series_uid = series_uid.split('.')[-1][:20]
        
        content_dir = os.path.join(output_dir, f"{safe_patient_id}_{safe_study_uid}_{safe_series_uid}")
        os.makedirs(content_dir, exist_ok=True)
        
        # Save metadata
        metadata_file = os.path.join(content_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        extracted_files = [metadata_file]
        
        # Check if DICOM contains pixel data
        if hasattr(ds, 'pixel_array'):
            pixel_array = ds.pixel_array
            
            # Handle different types of pixel data
            if len(pixel_array.shape) == 2:
                # Single frame image
                image_file = extract_single_frame(pixel_array, content_dir, "image.png")
                if image_file:
                    extracted_files.append(image_file)
                    
            elif len(pixel_array.shape) == 3:
                # Multi-frame image or video
                if pixel_array.shape[0] > 1:
                    # Multiple frames - treat as video
                    video_file = extract_video_frames(pixel_array, content_dir, "video.mp4")
                    if video_file:
                        extracted_files.append(video_file)
                    
                    # Also extract individual frames
                    frames_dir = os.path.join(content_dir, "frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    for i, frame in enumerate(pixel_array):
                        frame_file = extract_single_frame(frame, frames_dir, f"frame_{i:04d}.png")
                        if frame_file:
                            extracted_files.append(frame_file)
                else:
                    # Single frame with color channels
                    image_file = extract_single_frame(pixel_array[0], content_dir, "image.png")
                    if image_file:
                        extracted_files.append(image_file)
            
            elif len(pixel_array.shape) == 4:
                # Multi-frame with color channels
                frames_dir = os.path.join(content_dir, "frames")
                os.makedirs(frames_dir, exist_ok=True)
                
                for i, frame in enumerate(pixel_array):
                    frame_file = extract_single_frame(frame, frames_dir, f"frame_{i:04d}.png")
                    if frame_file:
                        extracted_files.append(frame_file)
                
                # Try to create video from frames
                video_file = extract_video_frames(pixel_array, content_dir, "video.mp4")
                if video_file:
                    extracted_files.append(video_file)
        
        # Extract audio data if present
        if hasattr(ds, 'AudioSampleData'):
            audio_file = extract_audio_data(ds, content_dir)
            if audio_file:
                extracted_files.append(audio_file)
        
        return {
            'success': True,
            'patient_id': patient_id,
            'study_uid': study_uid,
            'series_uid': series_uid,
            'output_dir': content_dir,
            'extracted_files': extracted_files,
            'metadata': metadata
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_path': dicom_file_path
        }


def extract_metadata(ds):
    """
    Extract comprehensive metadata from DICOM dataset.
    
    Args:
        ds: pydicom Dataset object
        
    Returns:
        dict: Extracted metadata
    """
    metadata = {}
    
    # Patient information
    metadata['patient'] = {
        'id': getattr(ds, 'PatientID', 'Unknown'),
        'name': str(getattr(ds, 'PatientName', 'Unknown')),
        'birth_date': getattr(ds, 'PatientBirthDate', 'Unknown'),
        'sex': getattr(ds, 'PatientSex', 'Unknown'),
        'age': getattr(ds, 'PatientAge', 'Unknown')
    }
    
    # Study information
    metadata['study'] = {
        'instance_uid': getattr(ds, 'StudyInstanceUID', 'Unknown'),
        'date': getattr(ds, 'StudyDate', 'Unknown'),
        'time': getattr(ds, 'StudyTime', 'Unknown'),
        'description': getattr(ds, 'StudyDescription', 'Unknown'),
        'id': getattr(ds, 'StudyID', 'Unknown')
    }
    
    # Series information
    metadata['series'] = {
        'instance_uid': getattr(ds, 'SeriesInstanceUID', 'Unknown'),
        'number': getattr(ds, 'SeriesNumber', 'Unknown'),
        'description': getattr(ds, 'SeriesDescription', 'Unknown'),
        'date': getattr(ds, 'SeriesDate', 'Unknown'),
        'time': getattr(ds, 'SeriesTime', 'Unknown')
    }
    
    # Equipment information
    metadata['equipment'] = {
        'manufacturer': getattr(ds, 'Manufacturer', 'Unknown'),
        'model': getattr(ds, 'ManufacturerModelName', 'Unknown'),
        'station_name': getattr(ds, 'StationName', 'Unknown'),
        'software_version': getattr(ds, 'SoftwareVersions', 'Unknown')
    }
    
    # Image information
    if hasattr(ds, 'pixel_array'):
        pixel_array = ds.pixel_array
        metadata['image'] = {
            'modality': getattr(ds, 'Modality', 'Unknown'),
            'rows': getattr(ds, 'Rows', 'Unknown'),
            'columns': getattr(ds, 'Columns', 'Unknown'),
            'pixel_spacing': getattr(ds, 'PixelSpacing', 'Unknown'),
            'bits_allocated': getattr(ds, 'BitsAllocated', 'Unknown'),
            'bits_stored': getattr(ds, 'BitsStored', 'Unknown'),
            'photometric_interpretation': getattr(ds, 'PhotometricInterpretation', 'Unknown'),
            'samples_per_pixel': getattr(ds, 'SamplesPerPixel', 'Unknown'),
            'pixel_array_shape': pixel_array.shape,
            'pixel_array_dtype': str(pixel_array.dtype)
        }
        
        # Multi-frame specific information
        if len(pixel_array.shape) >= 3:
            metadata['image']['number_of_frames'] = getattr(ds, 'NumberOfFrames', pixel_array.shape[0])
            metadata['image']['frame_time'] = getattr(ds, 'FrameTime', 'Unknown')
            metadata['image']['cine_rate'] = getattr(ds, 'CineRate', 'Unknown')
    
    return metadata


def extract_single_frame(pixel_array, output_dir, filename):
    """
    Extract a single frame as PNG image.
    
    Args:
        pixel_array: numpy array containing pixel data
        output_dir: output directory
        filename: output filename
        
    Returns:
        str: path to saved image file or None if failed
    """
    try:
        # Normalize pixel values to 0-255 range
        if pixel_array.dtype != np.uint8:
            # Handle different bit depths
            if pixel_array.max() > 255:
                # Scale to 8-bit
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            else:
                pixel_array = pixel_array.astype(np.uint8)
        
        # Handle different image formats
        if len(pixel_array.shape) == 2:
            # Grayscale image
            image = Image.fromarray(pixel_array, mode='L')
        elif len(pixel_array.shape) == 3:
            if pixel_array.shape[2] == 3:
                # RGB image
                image = Image.fromarray(pixel_array, mode='RGB')
            elif pixel_array.shape[2] == 1:
                # Single channel image
                image = Image.fromarray(pixel_array[:, :, 0], mode='L')
            else:
                # Unknown format, try first channel
                image = Image.fromarray(pixel_array[:, :, 0], mode='L')
        else:
            return None
        
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None


def extract_video_frames(pixel_array, output_dir, filename):
    """
    Extract multi-frame data as MP4 video.
    
    Args:
        pixel_array: numpy array containing multi-frame pixel data
        output_dir: output directory
        filename: output filename
        
    Returns:
        str: path to saved video file or None if failed
    """
    try:
        output_path = os.path.join(output_dir, filename)
        
        # Get frame dimensions
        if len(pixel_array.shape) == 3:
            num_frames, height, width = pixel_array.shape
            channels = 1
        elif len(pixel_array.shape) == 4:
            num_frames, height, width, channels = pixel_array.shape
        else:
            return None
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Default frame rate
        
        if channels == 1:
            # Grayscale video
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
        else:
            # Color video
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        
        # Write frames
        for i in range(num_frames):
            frame = pixel_array[i]
            
            # Normalize frame
            if frame.dtype != np.uint8:
                if frame.max() > 255:
                    frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Handle different channel formats
            if len(frame.shape) == 2:
                # Grayscale
                video_writer.write(frame)
            elif len(frame.shape) == 3:
                if frame.shape[2] == 3:
                    # RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                else:
                    # Use first channel
                    video_writer.write(frame[:, :, 0])
        
        video_writer.release()
        return output_path
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return None


def extract_audio_data(ds, output_dir):
    """
    Extract audio data from DICOM file.
    
    Args:
        ds: pydicom Dataset object
        output_dir: output directory
        
    Returns:
        str: path to saved audio file or None if failed
    """
    try:
        if hasattr(ds, 'AudioSampleData'):
            audio_data = ds.AudioSampleData
            output_path = os.path.join(output_dir, "audio.wav")
            
            # Save raw audio data (this is a simplified approach)
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            return output_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
    
    return None


def unpack_dicom_directory(raw_data_dir, output_dir="unpacked_dicom"):
    """
    Unpack all DICOM files in the raw_data directory.
    
    Args:
        raw_data_dir (str): Path to raw_data directory
        output_dir (str): Output directory for unpacked content
        
    Returns:
        dict: Summary of unpacking results
    """
    if not os.path.exists(raw_data_dir):
        print(f"Error: Directory {raw_data_dir} does not exist")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize statistics
    stats = {
        'total_files': 0,
        'successful_extractions': 0,
        'failed_extractions': 0,
        'patients_processed': set(),
        'extracted_content': [],
        'errors': []
    }
    
    print(f"Unpacking DICOM files from: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Process all files in the directory
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.startswith('.'):  # Skip hidden files
                continue
            
            file_path = os.path.join(root, file)
            stats['total_files'] += 1
            
            print(f"Processing: {file}")
            
            # Extract content from DICOM file
            result = extract_dicom_content(file_path, output_dir)
            
            if result['success']:
                stats['successful_extractions'] += 1
                stats['patients_processed'].add(result['patient_id'])
                stats['extracted_content'].append({
                    'source_file': file_path,
                    'patient_id': result['patient_id'],
                    'output_dir': result['output_dir'],
                    'extracted_files': result['extracted_files']
                })
                print(f"  ✓ Extracted to: {result['output_dir']}")
                print(f"  ✓ Files created: {len(result['extracted_files'])}")
            else:
                stats['failed_extractions'] += 1
                stats['errors'].append(result)
                print(f"  ✗ Failed: {result['error']}")
    
    # Convert set to list for JSON serialization
    stats['patients_processed'] = list(stats['patients_processed'])
    
    return stats


def print_unpacking_summary(stats):
    """
    Print a formatted summary of the unpacking results.
    
    Args:
        stats (dict): Unpacking statistics
    """
    if not stats:
        return
    
    print("\n" + "=" * 60)
    print("DICOM UNPACKING SUMMARY")
    print("=" * 60)
    
    print(f"\nFILE STATISTICS:")
    print(f"  Total DICOM files processed: {stats['total_files']}")
    print(f"  Successful extractions: {stats['successful_extractions']}")
    print(f"  Failed extractions: {stats['failed_extractions']}")
    
    print(f"\nCONTENT STATISTICS:")
    print(f"  Unique patients processed: {len(stats['patients_processed'])}")
    print(f"  Total content directories created: {len(stats['extracted_content'])}")
    
    if stats['patients_processed']:
        print(f"\nPATIENTS PROCESSED:")
        for patient in sorted(stats['patients_processed'])[:10]:
            print(f"  - {patient}")
        if len(stats['patients_processed']) > 10:
            print(f"  ... and {len(stats['patients_processed']) - 10} more")
    
    if stats['errors']:
        print(f"\nERRORS ({len(stats['errors'])}):")
        for error in stats['errors'][:5]:
            print(f"  - {error.get('file_path', 'Unknown')}: {error.get('error', 'Unknown error')}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more errors")


def main():
    """Main function to run the DICOM unpacking."""
    parser = argparse.ArgumentParser(description='Unpack DICOM files and extract content')
    parser.add_argument('--raw-data-dir', default='raw_data', 
                       help='Path to raw_data directory (default: raw_data)')
    parser.add_argument('--output-dir', default='unpacked_dicom',
                       help='Output directory for unpacked content (default: unpacked_dicom)')
    parser.add_argument('--save-log', action='store_true',
                       help='Save detailed log to JSON file')
    
    args = parser.parse_args()
    
    print("DICOM File Unpacker")
    print("=" * 30)
    
    # Run unpacking
    stats = unpack_dicom_directory(args.raw_data_dir, args.output_dir)
    
    if stats:
        # Print summary
        print_unpacking_summary(stats)
        
        # Save log if requested
        if args.save_log:
            log_file = os.path.join(args.output_dir, "unpacking_log.json")
            with open(log_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"\nDetailed log saved to: {log_file}")
        
        print(f"\nUnpacking complete! Check the '{args.output_dir}' directory for extracted content.")
    else:
        print("Unpacking failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
