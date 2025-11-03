import os
from pathlib import Path

# Base directory
base_dir = r"d:\finetune for nre lang\XTTSv2-Finetuning-for-New-Languages"

checkpoints_dir = os.path.join(base_dir, "checkpoints")

print("Searching for checkpoint files...\n")

# Find GPT training output folders
if os.path.exists(checkpoints_dir):
    print("=== GPT Training Folders ===")
    gpt_folders = [f for f in os.listdir(checkpoints_dir) if f.startswith("GPT_XTTS_FT-")]
    if gpt_folders:
        for folder in gpt_folders:
            folder_path = os.path.join(checkpoints_dir, folder)
            print(f"\nFolder: {folder}")
            
            # Find model checkpoints
            if os.path.isdir(folder_path):
                pth_files = [f for f in os.listdir(folder_path) if f.endswith(".pth")]
                if pth_files:
                    print("  Model checkpoints:")
                    for pth in sorted(pth_files):
                        print(f"    - {pth}")
                
                # Check for config.json
                config_path = os.path.join(folder_path, "config.json")
                if os.path.exists(config_path):
                    print(f"  ✓ config.json found")
    else:
        print("No GPT training folders found yet (you haven't trained yet)")
    
    print("\n=== Vocabulary Files ===")
    # Find vocab.json
    for root, dirs, files in os.walk(checkpoints_dir):
        for file in files:
            if file == "vocab.json":
                vocab_path = os.path.join(root, file)
                print(f"✓ Found: {vocab_path}")
    
    print("\n=== XTTS Original Model Files ===")
    original_model_path = os.path.join(checkpoints_dir, "XTTS_v2.0_original_model_files")
    if os.path.exists(original_model_path):
        print(f"✓ Original model folder exists: {original_model_path}")
        files = os.listdir(original_model_path)
        print(f"  Files: {', '.join(files[:10])}...")
    else:
        print(f"✗ Original model folder not found")
        
        # Check if XTTS-v2 folder exists
        xtts_v2_path = os.path.join(base_dir, "XTTS-v2")
        if os.path.exists(xtts_v2_path):
            print(f"\n  Found XTTS-v2 folder: {xtts_v2_path}")
            print("  You need to copy this to checkpoints/XTTS_v2.0_original_model_files/")
else:
    print(f"Checkpoints directory not found: {checkpoints_dir}")
    print("You may need to create it or run the setup steps first")