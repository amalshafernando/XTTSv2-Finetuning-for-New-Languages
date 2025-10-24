#!/usr/bin/env python3
"""
Kaggle Notebook Training Script for XTTSv2 Sinhala Finetuning
Run this in Kaggle notebook to train the model
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def install_requirements():
    """Install required packages in Kaggle"""
    print("Installing requirements...")
    
    # Install TTS package
    subprocess.run([sys.executable, "-m", "pip", "install", "TTS"], check=True)
    
    # Install other requirements
    requirements = [
        "transformers>=4.45.2",
        "tokenizers==0.20.1", 
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.11.2",
        "pandas>=1.4.0,<2.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.64.1",
        "einops>=0.6.0",
        "unidecode>=1.3.0",
        "inflect>=5.0.0",
        "phonemizer>=3.2.0",
        "espeak-ng>=1.50",
        "coqpit>=0.0.16",
        "trainer>=0.0.36"
    ]
    
    for req in requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {req}")
    
    print("✅ Requirements installation completed!")

def setup_dataset():
    """Setup dataset from Kaggle input"""
    print("Setting up dataset...")
    
    # Import the dataset setup script
    from kaggle_dataset_setup import setup_kaggle_dataset
    
    success = setup_kaggle_dataset()
    if not success:
        print("❌ Dataset setup failed!")
        return False
    
    print("✅ Dataset setup completed!")
    return True

def download_model_files():
    """Download XTTSv2 model files"""
    print("Downloading XTTSv2 model files...")
    
    # Create checkpoints directory
    checkpoints_dir = "checkpoints/XTTS_v2.0_original_model_files"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Download script
    download_script = """
import os
from TTS.utils.manage import ModelManager

# Define paths
CHECKPOINTS_OUT_PATH = "checkpoints/XTTS_v2.0_original_model_files/"

# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# XTTS files
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

# Download all files
all_links = [DVAE_CHECKPOINT_LINK, MEL_NORM_LINK, TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK, XTTS_CONFIG_LINK]
ModelManager._download_model_files(all_links, CHECKPOINTS_OUT_PATH, progress_bar=True)

print("✅ All model files downloaded!")
"""
    
    # Write and execute download script
    with open("download_models.py", "w") as f:
        f.write(download_script)
    
    subprocess.run([sys.executable, "download_models.py"], check=True)
    os.remove("download_models.py")
    
    print("✅ Model files downloaded!")

def train_dvae():
    """Train DVAE model"""
    print("Starting DVAE training...")
    
    cmd = [
        sys.executable, "train_dvae_xtts.py",
        "--output_path", "checkpoints",
        "--train_csv_path", "datasets/sinhala/metadata_train.csv",
        "--eval_csv_path", "datasets/sinhala/metadata_eval.csv", 
        "--language", "si",  # Sinhala language code
        "--lr", "5e-6",
        "--num_epochs", "5",
        "--batch_size", "512"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("✅ DVAE training completed!")

def train_gpt():
    """Train GPT model"""
    print("Starting GPT training...")
    
    cmd = [
        sys.executable, "train_gpt_xtts.py",
        "--output_path", "checkpoints",
        "--metadatas", "datasets/sinhala/metadata_train.csv,datasets/sinhala/metadata_eval.csv,si",
        "--num_epochs", "10",
        "--batch_size", "1",
        "--grad_acumm", "1",
        "--max_audio_length", "255995",
        "--max_text_length", "200",
        "--weight_decay", "1e-2",
        "--lr", "5e-6",
        "--save_step", "5000"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("✅ GPT training completed!")

def main():
    """Main training pipeline"""
    print("🚀 Starting XTTSv2 Sinhala Finetuning on Kaggle...")
    
    try:
        # Step 1: Install requirements
        install_requirements()
        
        # Step 2: Setup dataset
        if not setup_dataset():
            return
        
        # Step 3: Download model files
        download_model_files()
        
        # Step 4: Train DVAE (optional but recommended)
        print("\n" + "="*50)
        print("STEP 4: Training DVAE Model")
        print("="*50)
        train_dvae()
        
        # Step 5: Train GPT
        print("\n" + "="*50)
        print("STEP 5: Training GPT Model")
        print("="*50)
        train_gpt()
        
        print("\n🎉 Training completed successfully!")
        print("Your finetuned model is saved in the 'checkpoints' directory")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
