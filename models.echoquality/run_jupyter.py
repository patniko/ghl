#!/usr/bin/env python
"""
Script to run Jupyter notebook with the EchoQuality environment.
This script ensures that Jupyter is run with the correct Python environment
and configuration.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from pathlib import Path

def main():
    """Run Jupyter notebook with the EchoQuality environment."""
    parser = argparse.ArgumentParser(description="Run Jupyter notebook with EchoQuality environment")
    parser.add_argument("--port", type=int, default=8888, help="Port to run Jupyter on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    args = parser.parse_args()
    
    # Ensure mask_images directory exists in results
    mask_dir = Path("results/mask_images")
    if not mask_dir.exists():
        os.makedirs(mask_dir / "original", exist_ok=True)
        os.makedirs(mask_dir / "before", exist_ok=True)
        os.makedirs(mask_dir / "after", exist_ok=True)
        print(f"Created mask_images directory at {mask_dir.absolute()}")
    
    # Ensure data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        os.makedirs(data_dir / "example_study", exist_ok=True)
        print(f"Created data directory at {data_dir.absolute()}")
    
    # Build command
    cmd = [
        sys.executable, "-m", "jupyter", "notebook",
        f"--port={args.port}",
        f"--ip={args.ip}"
    ]
    
    if args.no_browser:
        cmd.append("--no-browser")
    
    # Print info
    print(f"Starting Jupyter notebook on http://{args.ip}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    # Open browser if requested
    if not args.no_browser:
        webbrowser.open(f"http://{args.ip}:{args.port}/tree")
    
    # Run Jupyter
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
