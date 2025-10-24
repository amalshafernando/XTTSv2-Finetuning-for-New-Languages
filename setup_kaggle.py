#!/usr/bin/env python3
"""
Kaggle setup script for XTTSv2 Sinhala finetuning
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages for Kaggle"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_kaggle.txt"])
    print("Requirements installed successfully!")

def setup_directories():
    """Create necessary directories"""
    dirs = [
        "checkpoints",
        "checkpoints/XTTS_v2.0_original_model_files",
        "datasets/sinhala/wavs",
        "outputs"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def download_model_files():
    """Download XTTSv2 model files"""
    print("Downloading XTTSv2 model files...")
    subprocess.check_call([sys.executable, "download_checkpoint.py", "--output_path", "checkpoints/"])
    print("Model files downloaded!")

if __name__ == "__main__":
    print("Setting up XTTSv2 for Kaggle...")
    setup_directories()
    install_requirements()
    download_model_files()
    print("Setup complete! Ready for training.")
