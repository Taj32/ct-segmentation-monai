from huggingface_hub import hf_hub_download
import os

os.makedirs("checkpoints", exist_ok=True)
hf_hub_download(
    repo_id="Hipps/ct-segmenetation-spleen",
    filename="best_model.pth",
    local_dir="checkpoints"
)
print("Model downloaded successfully")