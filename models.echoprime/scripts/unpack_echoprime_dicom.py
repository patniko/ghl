#!/usr/bin/env python3
"""
Simple script to unpack DICOM files from the EchoPrime raw_data directory.
This script uses the unpack_dicom_files.py module to extract content from DICOM files.
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to the path so we can import the unpacker
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from unpack_dicom_files import unpack_dicom_directory, print_unpacking_summary
import json


def main():
    """Main function to unpack EchoPrime DICOM files."""
    
    # Set paths relative to the echoprime directory
    echoprime_dir = Path(__file__).parent.parent
    raw_data_dir = echoprime_dir / "raw_data"
    output_dir = echoprime_dir / "unpacked_data"
    
    print("EchoPrime DICOM Unpacker")
    print("=" * 40)
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if raw_data directory exists
    if not raw_data_dir.exists():
        print(f"Error: Raw data directory does not exist: {raw_data_dir}")
        sys.exit(1)
    
    # Run the unpacking
    stats = unpack_dicom_directory(str(raw_data_dir), str(output_dir))
    
    if stats:
        # Print summary
        print_unpacking_summary(stats)
        
        # Save a log file
        log_file = output_dir / "echoprime_unpacking_log.json"
        with open(log_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\nDetailed log saved to: {log_file}")
        
        print(f"\n🎉 Unpacking complete!")
        print(f"📁 Check the '{output_dir}' directory for extracted content.")
        print(f"📊 Processed {stats['total_files']} DICOM files")
        print(f"✅ Successfully extracted {stats['successful_extractions']} files")
        print(f"👥 Found {len(stats['patients_processed'])} unique patients")
        
        if stats['failed_extractions'] > 0:
            print(f"⚠️  {stats['failed_extractions']} files failed to extract")
    else:
        print("❌ Unpacking failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
