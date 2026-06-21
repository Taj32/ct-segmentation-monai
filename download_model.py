#imports
from huggingface_hub import hf_hub_download
import os

print("starting model download...")
os.makedirs("checkpoints", exist_ok=True)

# docker container on huggingface spaces downloads the model
try:
    path = hf_hub_download(
        repo_id="Hipps/ct-segmenetation-spleen",
        filename="best_model_liver_6_19.pth",
        local_dir="checkpoints"
    )
    print(f"Downloaded to: {path}")
    print(f"File size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")
except Exception as e:
    print(f"Download failed: {e}")
    raise