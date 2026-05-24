from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import torch
import tempfile
import os
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, EnsureTyped,
    KeepLargestConnectedComponent, AsDiscrete
)
from monai.inferers import sliding_window_inference
import matplotlib.patches as mpatches
import SimpleITK as sitk
from huggingface_hub import hf_hub_download


app = FastAPI(title="CT Segmentation API")


# -- Helper Functions --
def dicom_to_nifti(dicom_path: str, output_path: str) -> str:
    """Convert a DICOM file to NIfTI format."""
    image = sitk.ReadImage(dicom_path)
    sitk.WriteImage(image, output_path)
    return output_path

# download model from HF hub if not already present
model_path = "checkpoints/best_model.pth"
if not os.path.exists(model_path):
    os.makedirs("checkpoints", exist_ok=True)
    model_path = hf_hub_download(
        repo_id="Hipps/ct-segmentation-spleen",
        filename="best_model.pth",
        local_dir="checkpoints"
    )

# load model once at startup
device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3, # configure model to use 3D convolutions
    in_channels=1, # input: grayscale image
    out_channels=2, # output: spleen and background segmentation
    channels=(16, 32, 64, 128, 256), # Define number  f filters in each layer
    strides=(2, 2, 2, 2), # downsample
    num_res_units=2,
    norm=Norm.BATCH, # use batch normalization for better stability
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ── preprocessing transforms ──
preprocess = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0)),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"], a_min=-57, a_max=164,
        b_min=0.0, b_max=1.0, clip=True
    ),
    EnsureTyped(keys=["image"]),
])

post_pred = AsDiscrete(argmax=True, to_onehot=2)
keep_largest = KeepLargestConnectedComponent(applied_labels=[1])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    # validate file type
    if not file.filename.endswith((".nii", ".nii.gz", ".dcm")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii, .nii.gz, or .dcm files accepted"
        )

    is_dicom = file.filename.endswith(".dcm")
    suffix = ".dcm" if is_dicom else (
        ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode='wb') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        
        if is_dicom:
            nifti_path = tmp_path.replace(".dcm", ".nii.gz")
            dicom_to_nifti(tmp_path, nifti_path)
            os.unlink(tmp_path)  # delete original DICOM
            tmp_path = nifti_path
        
        # preprocess
        data = preprocess([{"image": tmp_path}])
        input_tensor = data[0]["image"].unsqueeze(0).to(device)

        # run inference
        with torch.no_grad():
            output = sliding_window_inference(
                input_tensor, (96, 96, 96), 4, model, overlap=0.75, mode="gaussian"
            )

        # post process
        output_discrete = torch.argmax(output, dim=1, keepdim=True)
        output_cleaned = keep_largest(output_discrete[0])

        # find spleen slices and pick middle one
        vol = output_cleaned[0].cpu().numpy()
        spleen_slices = [
            i for i in range(vol.shape[2]) if vol[:, :, i].sum() > 0
        ]

        if not spleen_slices:
            return {"message": "No spleen detected", "dice": None}

        slice_idx = spleen_slices[len(spleen_slices) // 2]

        # generate overlay PNG
        slice_ct = input_tensor[0, 0, :, :, slice_idx].cpu().numpy()
        slice_pred = vol[:, :, slice_idx]
        plt.suptitle("Spleen Segmentation Result", fontsize=14, fontweight="bold")
        
        


        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(slice_ct, cmap="gray")
        axes[0].set_title("CT Scan")
        axes[0].axis("off")
        
        # create a red RGBA overlay manually
        red_overlay = np.zeros((*slice_pred.shape, 4))  # RGBA
        red_overlay[slice_pred == 1] = [1, 0, 0, 0.5]  # bright red, 50% opacity

        axes[1].imshow(slice_ct, cmap="gray")
        axes[1].imshow(red_overlay)
        axes[1].set_title("Predicted Segmentation")
        axes[1].axis("off")
        red_patch = mpatches.Patch(color="red", alpha=0.7, label="Spleen Prediction")
        axes[1].legend(handles=[red_patch], loc="lower right")


        plt.tight_layout()
        overlay_path = tmp_path.replace(suffix, "_overlay.png")
        plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close()

        return FileResponse(
            overlay_path,
            media_type="image/png",
            filename="segmentation_overlay.png"
        )

    finally:
         if os.path.exists(tmp_path):
            os.unlink(tmp_path)