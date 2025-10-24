#!/usr/bin/env python3
"""
Kaggle Dataset Setup for XTTSv2 Sinhala Finetuning
This script prepares the dataset from Kaggle for XTTSv2 training
"""

import os
import pandas as pd
import shutil
from pathlib import Path

def setup_kaggle_dataset():
    """
    Setup dataset from Kaggle for XTTSv2 training
    """
    print("Setting up Kaggle dataset for XTTSv2 training...")
    
    # Define paths
    kaggle_dataset_path = "/kaggle/input/sinhala-tts-dataset"  # Kaggle input path
    target_dataset_path = "datasets/sinhala"
    
    # Create target directory structure
    os.makedirs(f"{target_dataset_path}/wavs", exist_ok=True)
    
    # Copy audio files
    print("Copying audio files...")
    if os.path.exists(f"{kaggle_dataset_path}/wavs"):
        shutil.copytree(f"{kaggle_dataset_path}/wavs", f"{target_dataset_path}/wavs", dirs_exist_ok=True)
        print(f"✓ Copied audio files to {target_dataset_path}/wavs")
    else:
        print("❌ Audio files not found in Kaggle dataset")
        return False
    
    # Convert CSV files to XTTSv2 format
    print("Converting CSV files to XTTSv2 format...")
    
    # Process training data
    if os.path.exists(f"{kaggle_dataset_path}/train.csv"):
        df_train = pd.read_csv(f"{kaggle_dataset_path}/train.csv", sep='|')
        df_train_xtts = pd.DataFrame()
        df_train_xtts['audio_file'] = df_train['audio_file_path'].apply(lambda x: x.replace('wav/', 'wavs/'))
        df_train_xtts['text'] = df_train['transcript']
        df_train_xtts['speaker_name'] = df_train['speaker_id']
        df_train_xtts.to_csv(f"{target_dataset_path}/metadata_train.csv", sep='|', index=False)
        print(f"✓ Converted training data: {len(df_train)} samples")
    else:
        print("❌ Training CSV not found")
        return False
    
    # Process validation data
    if os.path.exists(f"{kaggle_dataset_path}/val.csv"):
        df_val = pd.read_csv(f"{kaggle_dataset_path}/val.csv", sep='|')
        df_val_xtts = pd.DataFrame()
        df_val_xtts['audio_file'] = df_val['audio_file_path'].apply(lambda x: x.replace('wav/', 'wavs/'))
        df_val_xtts['text'] = df_val['transcript']
        df_val_xtts['speaker_name'] = df_val['speaker_id']
        df_val_xtts.to_csv(f"{target_dataset_path}/metadata_eval.csv", sep='|', index=False)
        print(f"✓ Converted validation data: {len(df_val)} samples")
    else:
        print("❌ Validation CSV not found")
        return False
    
    # Copy dictionary file if exists
    if os.path.exists(f"{kaggle_dataset_path}/dict.txt"):
        shutil.copy2(f"{kaggle_dataset_path}/dict.txt", f"{target_dataset_path}/dict.txt")
        print("✓ Copied dictionary file")
    
    print("✅ Dataset setup completed successfully!")
    return True

if __name__ == "__main__":
    setup_kaggle_dataset()
