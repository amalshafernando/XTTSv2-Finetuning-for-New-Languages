import os
import urllib.request
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) / total_size)
            print(f"\rDownloading {filename}: {percent:.1f}%", end="")
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\nDownloaded: {filename}")
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        return False

def main():
    # Create checkpoints directory
    checkpoints_dir = "XTTSv2-Finetuning-for-New-Languages/checkpoints/XTTS_v2.0_original_model_files"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # URLs for XTTS v2.0 files
    files_to_download = [
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json", "vocab.json"),
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth", "model.pth"),
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json", "config.json"),
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth", "dvae.pth"),
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth", "mel_stats.pth")
    ]
    
    print("Downloading XTTS v2.0 model files...")
    
    for url, filename in files_to_download:
        filepath = os.path.join(checkpoints_dir, filename)
        if os.path.exists(filepath):
            print(f"File already exists: {filename}")
            continue
            
        print(f"Downloading {filename}...")
        success = download_file(url, filepath)
        if not success:
            print(f"Failed to download {filename}")
    
    print("Download process completed!")

if __name__ == "__main__":
    main()

