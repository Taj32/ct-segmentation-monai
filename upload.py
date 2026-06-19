
# imports
from huggingface_hub import login, upload_file, upload_file, upload_folder
import os
from dotenv import load_dotenv


load_dotenv()

# Login to the Hugging Face Hub
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

# Upload the best model to the Hugging Face Hub
upload_file(
    path_or_fileobj="checkpoints/best_model.pth",
    path_in_repo="best_model.pth",
    repo_id="Hipps/ct-segmenetation-spleen",
    repo_type="model",
    commit_message="Upload best_model.pth"
)