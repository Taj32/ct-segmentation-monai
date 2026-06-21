"""
app.py
======
FastAPI backend for the 3D CT Liver & Tumor Segmentation Platform.

Exposes three endpoints:
  GET  /          — health check (root)
  GET  /health    — detailed health check with model info
  POST /segment       — upload a CT scan, receive a segmentation overlay PNG
  POST /segment_data  — upload a CT scan, receive raw CT + mask as .npz for interactive viewing

The model is a 3D U-Net trained on the Medical Segmentation Decathlon Task03 Liver dataset.
It outputs a 3-class segmentation mask: 0=background, 1=liver, 2=tumor.

Inference uses MONAI's sliding window approach to handle full CT volumes that
exceed GPU memory. Deployed on Hugging Face Spaces (CPU inference).
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response, FileResponse
from contextlib import asynccontextmanager

import torch
import tempfile
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import SimpleITK as sitk

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, EnsureTyped,
    KeepLargestConnectedComponent, AsDiscrete
)
from monai.inferers import sliding_window_inference


# ── Lifespan Event ─────────────────────────────────────────────────────────────
# Runs startup/shutdown logic. Currently just prints a ready message.
# Can be extended to warm up the model or connect to a database on startup.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application ready")
    yield  # application runs here; anything after yield runs on shutdown


app = FastAPI(title="CT Segmentation API", lifespan=lifespan)


# ── Helper Functions ───────────────────────────────────────────────────────────

def dicom_to_nifti(dicom_path: str, output_path: str) -> str:
    """
    Convert a DICOM file to NIfTI format using SimpleITK.
    MONAI's preprocessing pipeline expects NIfTI (.nii.gz) input,
    so DICOM uploads are converted before inference.
    """
    image = sitk.ReadImage(dicom_path)
    sitk.WriteImage(image, output_path)
    return output_path


# ── Model Loading ──────────────────────────────────────────────────────────────
# Model is downloaded from HF model hub at Docker build time via download_model.py
# and stored at this path inside the container.
model_path = "checkpoints/best_model_liver_6_19.pth"

# Use GPU if available, otherwise fall back to CPU (HF free tier is CPU-only)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 3D U-Net architecture — same configuration used during training
# out_channels=3: background (0), liver (1), tumor (2)
model = UNet(
    spatial_dims=3,                    # 3D convolutions for volumetric CT data
    in_channels=1,                     # grayscale CT input (single channel)
    out_channels=3,                    # 3-class output: background / liver / tumor
    channels=(16, 32, 64, 128, 256),   # feature map sizes at each encoder level
    strides=(2, 2, 2, 2),              # downsampling stride between levels
    num_res_units=2,                   # residual units per level for better gradient flow
    norm=Norm.BATCH,                   # batch normalization for training stability
).to(device)

# load trained weights — map_location ensures compatibility between GPU-trained
# and CPU-deployed models
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # set to evaluation mode (disables dropout, fixes batch norm)


# ── Preprocessing Pipeline ─────────────────────────────────────────────────────
# MONAI Compose applies transforms sequentially to each uploaded CT volume.
# These match the transforms used during training to ensure consistent inference.
preprocess = Compose([
    LoadImaged(keys=["image"]),                    # read NIfTI file from disk
    EnsureChannelFirstd(keys=["image"]),           # add channel dim: (H,W,D) → (1,H,W,D)
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0)),  # resample to isotropic 1.5mm spacing
    Orientationd(keys=["image"], axcodes="RAS"),   # standardize orientation to RAS
    ScaleIntensityRanged(                          # window CT HU values to [0, 1]
        keys=["image"],
        a_min=-57, a_max=164,                      # soft tissue HU window
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    EnsureTyped(keys=["image"]),                   # ensure correct tensor dtype
])

# Post-processing transforms (applied after inference)
post_pred = AsDiscrete(argmax=True, to_onehot=3)   # convert logits to one-hot class labels
post_label = AsDiscrete(argmax=True, to_onehot=3)  # same for ground truth labels

# Remove isolated prediction artifacts — keeps only the largest connected region
# for each class (prevents small spurious predictions far from the main organ)
keep_largest = KeepLargestConnectedComponent(applied_labels=[1, 2])


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint — basic health check for HF Spaces container monitoring."""
    return {"status": "ok", "message": "CT Segmentation API is running"}


@app.get("/health")
def health():
    """
    Detailed health check returning model configuration.
    Useful for verifying which checkpoint is loaded after a deployment update.
    """
    return {
        "status": "ok",
        "model_path": model_path,
        "device": str(device),
        "out_channels": 3
    }


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Main segmentation endpoint.
    Accepts a NIfTI (.nii / .nii.gz) or DICOM (.dcm) CT scan upload.
    Returns a PNG image showing the CT scan with a color overlay:
      - Green = predicted liver region
      - Red   = predicted tumor region

    The overlay is generated on the middle axial slice containing liver/tumor.
    """
    # ── Input Validation ──
    if not file.filename.endswith((".nii", ".nii.gz", ".dcm")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii, .nii.gz, or .dcm files accepted"
        )

    is_dicom = file.filename.endswith(".dcm")
    suffix = ".dcm" if is_dicom else (
        ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    )

    # read entire file content into memory
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty — please try again"
        )

    # write to a temporary file so MONAI can read it from disk
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode='wb') as tmp:
        tmp.write(content)
        tmp.flush()       # push Python buffer to OS
        os.fsync(tmp.fileno())  # force OS to write to physical disk
        tmp_path = tmp.name

    try:
        # ── DICOM Conversion ──
        # Convert DICOM to NIfTI if needed before passing to MONAI pipeline
        if is_dicom:
            nifti_path = tmp_path.replace(".dcm", ".nii.gz")
            dicom_to_nifti(tmp_path, nifti_path)
            os.unlink(tmp_path)
            tmp_path = nifti_path

        # ── Preprocessing ──
        data = preprocess([{"image": tmp_path}])
        input_tensor = data[0]["image"].unsqueeze(0).to(device)  # add batch dim

        # ── Inference ──
        # Sliding window inference tiles the volume into 96³ overlapping patches,
        # runs each patch through the model, then stitches predictions together.
        # Gaussian mode weights center of each patch more heavily than edges.
        with torch.no_grad():
            output = sliding_window_inference(
                input_tensor, (96, 96, 96), 4, model, overlap=0.75, mode="gaussian"
            )

        # ── Post-processing ──
        # Use softmax probabilities with class-specific thresholds instead of argmax.
        # Lower tumor threshold (0.3 vs 0.5) increases tumor recall at the cost
        # of some precision — important since tumors are the minority class.
        probs = torch.softmax(output, dim=1)
        output_discrete = torch.zeros(
            (1, 1, *probs.shape[2:]),
            dtype=torch.long,
            device=device
        )
        output_discrete[probs[0:1, 1:2] > 0.5] = 1   # liver: standard threshold
        output_discrete[probs[0:1, 2:3] > 0.3] = 2   # tumor: lower threshold for higher recall

        # remove small disconnected prediction blobs
        output_cleaned = keep_largest(output_discrete[0])
        vol = output_cleaned[0].cpu().numpy()  # shape: (H, W, D)

        # find axial slices containing any liver or tumor prediction
        target_slices = [
            i for i in range(vol.shape[2]) if vol[:, :, i].sum() > 0
        ]

        if not target_slices:
            return {"message": "No liver or tumor detected", "dice": None}

        # pick the middle slice for visualization
        slice_idx = target_slices[len(target_slices) // 2]

        # ── Visualization ──
        slice_ct = input_tensor[0, 0, :, :, slice_idx].cpu().numpy()
        slice_pred = vol[:, :, slice_idx]

        # build RGBA overlay — transparent where background (class 0)
        color_overlay = np.zeros((*slice_pred.shape, 4))
        color_overlay[slice_pred == 1] = [0, 1, 0, 0.5]   # semi-transparent green = liver
        color_overlay[slice_pred == 2] = [1, 0, 0, 0.7]   # semi-transparent red = tumor

        plt.suptitle("Liver Tumor Segmentation Result", fontsize=14, fontweight="bold")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # left panel: raw CT scan
        axes[0].imshow(slice_ct, cmap="gray")
        axes[0].set_title("CT Scan")
        axes[0].axis("off")

        # right panel: CT with color segmentation overlay
        axes[1].imshow(slice_ct, cmap="gray")
        axes[1].imshow(color_overlay)
        axes[1].set_title("Predicted Segmentation")
        axes[1].axis("off")
        green_patch = mpatches.Patch(color="green", alpha=0.5, label="Liver")
        red_patch = mpatches.Patch(color="red", alpha=0.7, label="Tumor")
        axes[1].legend(handles=[green_patch, red_patch], loc="lower right")

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
        # always clean up temp files to prevent disk space accumulation
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/segment_data")
async def segment_data(file: UploadFile = File(...)):
    """
    Full volume segmentation endpoint for the interactive slice viewer.
    Accepts a NIfTI (.nii / .nii.gz) or DICOM (.dcm) CT scan upload.
    Returns a compressed numpy .npz file containing:
      - ct:           preprocessed CT volume (H, W, D)
      - mask:         segmentation mask (H, W, D), values: 0/1/2
      - spleen_slices: axial slice indices containing liver or tumor

    The Streamlit dashboard loads this .npz to power the slice navigator
    and 3D volume viewer without making additional API calls.
    """
    # ── Input Validation ──
    if not file.filename.endswith((".nii", ".nii.gz", ".dcm")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii, .nii.gz, or .dcm files accepted"
        )

    is_dicom = file.filename.endswith(".dcm")
    suffix = ".dcm" if is_dicom else (
        ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty — please try again"
        )

    # HF free tier has limited memory — reject files over 50MB
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large for free tier — please use a file under 50MB"
        )

    # write to temp file and verify it was written correctly
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode='wb') as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name

    written_size = os.path.getsize(tmp_path)
    print(f"Content received: {len(content)} bytes, Written to disk: {written_size} bytes")

    if written_size == 0:
        raise HTTPException(
            status_code=500,
            detail=f"File write failed — received {len(content)} bytes but wrote 0"
        )

    try:
        # ── DICOM Conversion ──
        if is_dicom:
            nifti_path = tmp_path.replace(".dcm", ".nii.gz")
            dicom_to_nifti(tmp_path, nifti_path)
            os.unlink(tmp_path)
            tmp_path = nifti_path

        # ── Preprocessing ──
        data = preprocess([{"image": tmp_path}])
        input_tensor = data[0]["image"].unsqueeze(0).to(device)

        # ── Inference ──
        with torch.no_grad():
            output = sliding_window_inference(
                input_tensor, (96, 96, 96), 4, model, overlap=0.75, mode="gaussian"
            )

        # ── Post-processing ──
        # Same thresholding strategy as /segment endpoint
        probs = torch.softmax(output, dim=1)
        output_discrete = torch.zeros(
            (1, 1, *probs.shape[2:]),
            dtype=torch.long,
            device=device
        )
        output_discrete[probs[0:1, 1:2] > 0.5] = 1   # liver
        output_discrete[probs[0:1, 2:3] > 0.3] = 2   # tumor (lower threshold)
        output_cleaned = keep_largest(output_discrete[0])

        vol = output_cleaned[0].cpu().numpy()   # segmentation mask
        ct = input_tensor[0, 0].cpu().numpy()   # preprocessed CT volume

        # find slices containing any predicted liver or tumor voxels
        target_slices = [
            i for i in range(vol.shape[2]) if vol[:, :, i].sum() > 0
        ]

        if not target_slices:
            return {"message": "No liver or tumor detected"}

        # ── Pack and Return ──
        # Compress CT + mask into a single .npz binary for efficient transfer.
        # Note: key named 'spleen_slices' for backwards compatibility with Streamlit.
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            ct=ct,
            mask=vol,
            spleen_slices=np.array(target_slices)
        )
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=segmentation.npz"}
        )

    finally:
        # clean up temp files regardless of success or failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)