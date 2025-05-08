#!/usr/bin/env python3
"""
Script to start the MLflow server.
Run this script to install dependencies and start the MLflow server.
"""
import os
import subprocess
import sys

def main():
    """Install dependencies and start the MLflow server."""
    print("Installing dependencies...")
    subprocess.run(["poetry", "install"], check=True)
    
    print("\nStarting MLflow server...")
    subprocess.run(["poetry", "run", "mlflow", "server", 
                   "--host", "127.0.0.1", "--port", "5000"], check=True)

if __name__ == "__main__":
    main()
