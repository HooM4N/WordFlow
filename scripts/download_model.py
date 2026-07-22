# downloads latest pre-trained checkpoint from hf
import os
from huggingface_hub import snapshot_download

def download_run(repo_id="hoom4n/WordFlow", save_dir="runs/latest"):
    print(f"Downloading {repo_id} to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=save_dir,
        allow_patterns=["*.pt", "*.json"],
        local_dir_use_symlinks=False
    )
    print("Download complete!")

if __name__ == "__main__":
    download_run()