#!/usr/bin/env python3
"""
Script to run Jupyter notebook for HuBERT-ECG.

This script:
1. Sets up the environment for Jupyter
2. Starts the Jupyter notebook server
3. Provides a URL for accessing the notebook

Usage:
    python run_jupyter.py [--port PORT] [--no-browser]
"""

import argparse
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run Jupyter notebook for HuBERT-ECG')
    parser.add_argument('--port', type=int, default=8888,
                        help='Port to run Jupyter on (default: 8888)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open a browser window')
    
    args = parser.parse_args()
    
    print(f"Starting Jupyter notebook server on port {args.port}")
    
    # Build the command to run Jupyter
    cmd = [
        "jupyter", 
        "notebook", 
        "--port", str(args.port)
    ]
    
    if args.no_browser:
        cmd.append("--no-browser")
    
    # Start the Jupyter server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped")
    except subprocess.CalledProcessError as e:
        print(f"\nError running Jupyter: {e}")
        print("\nMake sure the dependencies are installed with:")
        print("  pip install -r requirements.txt")
        print("  or")
        print("  poetry install")

if __name__ == '__main__':
    main()
