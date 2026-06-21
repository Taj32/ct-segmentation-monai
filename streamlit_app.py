"""
streamlit_app.py
================
Interactive dashboard for the 3D CT Liver & Tumor Segmentation project.
Connects to the FastAPI backend (hosted on HF Spaces) to run inference
and display results in three ways:
  1. Single overlay PNG via /segment
  2. Interactive slice-by-slice viewer via /segment_data
  3. 3D volumetric mesh viewer (Plotly)

Also displays model training metrics, inference latency history,
and model architecture information.
"""

import time
import io

import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from skimage import measure
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter




# ── Session State Initialization ──────────────────────────────────────────────
# Streamlit reruns the entire script on every interaction, so persistent
# state (CT volume, mask, latency history) must be stored in session_state.
if "latency_history" not in st.session_state:
    st.session_state["latency_history"] = []

if "ct" not in st.session_state:
    st.session_state["ct"] = None      # raw CT volume (numpy array)

if "mask" not in st.session_state:
    st.session_state["mask"] = None    # segmentation mask (numpy array, values 0/1/2)

if "spleen_slices" not in st.session_state:
    st.session_state["spleen_slices"] = []  # slice indices containing liver/tumor


# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CT Segmentation Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 3D CT Segmentation Dashboard")
st.markdown(
    "Upload a CT scan to get an automated liver and tumor segmentation "
    "using a U-Net model trained on the Medical Segmentation Decathlon."
)


# ── Sidebar: API Configuration ─────────────────────────────────────────────────
# Allows switching between the deployed HF Space API and a local dev server
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input(
    "API Endpoint",
    value="https://Hipps-ct-segmentation-monai.hf.space"
)


# ── Section 1: Quick Segmentation (/segment endpoint) ─────────────────────────
# Uploads a NIfTI file and returns a single overlay PNG showing
# the middle liver/tumor slice with green=liver, red=tumor overlays.
st.header("Upload CT Scan")
uploaded_file = st.file_uploader(
    "Choose a NIfTI file",
    type=["nii", "nii.gz"],
    help="Upload a .nii or .nii.gz CT scan file"
)

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")

    if st.button("Run Segmentation", type="primary"):
        with st.spinner("Running inference... this may take a few minutes on CPU"):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{api_url}/segment",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    timeout=600
                )
                latency = time.time() - start_time
                st.session_state["latency_history"].append(latency)

                if response.status_code == 200:
                    # decode PNG response and display inline
                    image = Image.open(io.BytesIO(response.content))
                    st.header("Segmentation Result")
                    st.image(
                        image,
                        caption="CT Scan with Predicted Segmentation Overlay",
                        use_container_width=True
                    )
                    st.success("Segmentation complete!")

                    # allow user to save the overlay image
                    st.download_button(
                        label="Download Overlay PNG",
                        data=response.content,
                        file_name="segmentation_overlay.png",
                        mime="image/png"
                    )
                else:
                    st.error(f"API error: {response.status_code} — {response.text}")

            except requests.exceptions.Timeout:
                st.error("Request timed out — file may be too large for the free tier")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ── Section 2: Model Training Metrics ─────────────────────────────────────────
# Hardcoded metrics from the final training run.
# Future improvement: connect to a hosted MLflow server for live metrics.
st.header("📊 Model Training Metrics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Best Val Dice", "0.7023")  # Most Updated Model
with col2:
    st.metric("Dataset", "Task03 Liver")
with col3:
    st.metric("Target Dice", "0.70 ✓")

st.subheader("Training Configuration")
st.table({
    "Parameter": ["Model", "Epochs", "Batch Size", "Learning Rate", "Patch Size", "Loss Function", "Classes"],
    "Value":     ["3D U-Net", "700+", "4", "1e-4", "96³", "Tversky Loss + DiceCELoss", "Background / Liver / Tumor"]
})


# ── Section 3: Interactive Slice Viewer (/segment_data endpoint) ───────────────
# Uploads a NIfTI file and returns the full CT + mask volumes as a compressed
# numpy .npz file. The slider lets the user scrub through every axial slice.
st.header("🔬 Interactive Slice Viewer")
st.markdown("Upload a CT scan to explore segmentation results slice by slice.")
st.warning("⚠️ HF free tier has upload limits — use files under 50MB for best results. Larger files may time out.")

uploaded_file_viewer = st.file_uploader(
    "Choose a NIfTI file for slice viewer",
    type=["nii", "nii.gz"],
    key="slice_viewer",
    help="Upload a .nii or .nii.gz CT scan file"
)

if uploaded_file_viewer is not None:
    st.success(f"File uploaded: {uploaded_file_viewer.name} ({uploaded_file_viewer.size / 1024 / 1024:.1f} MB)")

    if st.button("Run Slice Analysis", type="primary", key="slice_btn"):
        with st.spinner("Running inference... please wait"):
            try:
                start_time = time.time()

                # wrap in BytesIO to ensure the buffer is at position 0 before sending
                file_bytes = uploaded_file_viewer.getvalue()
                response = requests.post(
                    f"{api_url}/segment_data",
                    files={"file": (uploaded_file_viewer.name, io.BytesIO(file_bytes), "application/octet-stream")},
                    timeout=600
                )
                latency = time.time() - start_time
                st.session_state["latency_history"].append(latency)

                if response.status_code == 200:
                    # decode compressed numpy binary returned by the API
                    buf = io.BytesIO(response.content)
                    data = np.load(buf)
                    ct = data["ct"]               # shape: (H, W, D)
                    mask = data["mask"]           # shape: (H, W, D), values: 0/1/2
                    spleen_slices = data["spleen_slices"].tolist()  # axial slice indices with liver/tumor

                    # store volumes in session state so the slider can reference them
                    st.session_state["ct"] = ct
                    st.session_state["mask"] = mask
                    st.session_state["spleen_slices"] = spleen_slices

                    
                else:
                    st.error(f"API error: {response.status_code}")

            except Exception as e:
                st.error(f"Error: {str(e)}")


# ── Slice Navigator ────────────────────────────────────────────────────────────
# Renders once CT/mask data is loaded into session state.
# Three-panel view: raw CT | color overlay | binary mask
if "ct" in st.session_state and st.session_state["ct"] is not None:
    ct = st.session_state["ct"]
    mask = st.session_state["mask"]
    spleen_slices = st.session_state["spleen_slices"]

    st.subheader("Slice Navigator")

    # slider spanning the full depth of the volume
    slice_idx = st.slider(
        "Select Slice",
        min_value=0,
        max_value=ct.shape[2] - 1,
        value=spleen_slices[len(spleen_slices) // 2],  # default to middle liver slice
        help=f"Liver/tumor visible on slices {spleen_slices[0]} to {spleen_slices[-1]}"
    )
    
    #st.success(f"Segmentation complete! Liver/tumor found on {len(spleen_slices)} slices.")
    
    st.success(
        f"Segmentation complete — liver/tumor found on {len(spleen_slices)} slices "
        f"(slices {spleen_slices[0]}–{spleen_slices[-1]})"
    )
    
    # identify which slices contain tumor (class 2) specifically
    tumor_slices = [
        i for i in range(mask.shape[2])
        if (mask[:, :, i] == 2).sum() > 0
    ]

    if tumor_slices:
        st.success(
            f"🔴 Tumor detected on {len(tumor_slices)} slices "
            f"(slices {tumor_slices[0]}–{tumor_slices[-1]})"
        )
    else:
        st.info("No tumor detected in this scan — only liver segmentation present.")

    # indicator showing whether the selected slice contains liver/tumor
    if slice_idx in spleen_slices:
        st.success(f"✓ Liver/tumor detected on slice {slice_idx}")
    else:
        st.info(f"No liver/tumor on slice {slice_idx} — visible on slices {spleen_slices[0]}–{spleen_slices[-1]}")

    # extract 2D slices for the selected axial index
    slice_ct = ct[:, :, slice_idx]
    slice_mask = mask[:, :, slice_idx]

    # build RGBA color overlay: green=liver (class 1), red=tumor (class 2)
    color_overlay = np.zeros((*slice_mask.shape, 4))
    color_overlay[slice_mask == 1] = [0, 1, 0, 0.5]   # semi-transparent green
    color_overlay[slice_mask == 2] = [1, 0, 0, 0.7]   # semi-transparent red

    # three-panel matplotlib figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # panel 1: raw grayscale CT
    axes[0].imshow(slice_ct, cmap="gray")
    axes[0].set_title(f"CT Scan — Slice {slice_idx}", fontsize=14)
    axes[0].axis("off")

    # panel 2: CT with color segmentation overlay
    axes[1].imshow(slice_ct, cmap="gray")
    axes[1].imshow(color_overlay)
    axes[1].set_title("Predicted Segmentation", fontsize=14)
    axes[1].axis("off")
    green_patch = mpatches.Patch(color="green", alpha=0.5, label="Liver")
    red_patch = mpatches.Patch(color="red", alpha=0.7, label="Tumor")
    axes[1].legend(handles=[green_patch, red_patch], loc="lower right")

    # panel 3: raw binary segmentation mask
    seg_cmap = ListedColormap(["black", "green", "red"])
    axes[2].imshow(slice_mask, cmap=seg_cmap, vmin=0, vmax=2)
    axes[2].set_title("Segmentation Mask (Green=Liver, Red=Tumor)", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    # summary statistics for the current volume
    st.subheader("Slice Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Slices", ct.shape[2])
    with col2:
        st.metric("Liver/Tumor Slices", len(spleen_slices))
    with col3:
        # count only liver + tumor voxels (exclude background class 0)
        target_voxels = int((mask == 1).sum() + (mask == 2).sum())
        st.metric("Liver+Tumor Voxels", f"{target_voxels:,}")
else:
    st.info("Upload a CT scan above and click 'Run Slice Analysis' to explore slices interactively.")


# ── Section 4: 3D Volume Viewer ────────────────────────────────────────────────
# Uses marching cubes (scikit-image) to extract isosurfaces from the binary
# liver and tumor masks, then renders them as interactive 3D meshes in Plotly.
# Downsampled by step=2 to keep browser rendering fast.
st.header("🧊 3D Volume Viewer")

if "ct" in st.session_state and st.session_state["ct"] is not None:
    ct = st.session_state["ct"]
    mask = st.session_state["mask"]

    st.markdown("Interactive 3D view of the segmentation mask. Rotate and zoom using your mouse.")

    # downsample volume to reduce mesh complexity — step=2 halves resolution in each axis
    step = 2
    mask_down = mask[::step, ::step, ::step]

    fig_data = []

    # ── Liver mesh (class 1) ──
    liver_mask = (mask_down == 1).astype(float)
    if liver_mask.sum() > 0:
        try:
            # marching cubes extracts a triangular mesh from the binary volume
            liver_mask_smooth = gaussian_filter(liver_mask.astype(float), sigma=1.5)
            liver_verts, liver_faces, _, _ = measure.marching_cubes(liver_mask_smooth, level=0.5)
            fig_data.append(go.Mesh3d(
                x=liver_verts[:, 0],
                y=liver_verts[:, 1],
                z=liver_verts[:, 2],
                i=liver_faces[:, 0],
                j=liver_faces[:, 1],
                k=liver_faces[:, 2],
                color="green",
                opacity=0.4,
                name="Liver"
            ))
        except Exception:
            pass  # skip if marching cubes fails (e.g. too few voxels)

    # ── Tumor mesh (class 2) ──
    tumor_mask = (mask_down == 2).astype(float)
    if tumor_mask.sum() > 0:
        try:
            tumor_mask_smooth = gaussian_filter(tumor_mask.astype(float), sigma=1.0)
            tumor_verts, tumor_faces, _, _ = measure.marching_cubes(tumor_mask_smooth, level=0.5)
            fig_data.append(go.Mesh3d(
                x=tumor_verts[:, 0],
                y=tumor_verts[:, 1],
                z=tumor_verts[:, 2],
                i=tumor_faces[:, 0],
                j=tumor_faces[:, 1],
                k=tumor_faces[:, 2],
                color="red",
                opacity=0.8,
                name="Tumor"
            ))
        except Exception:
            pass

    if fig_data:
        fig = go.Figure(data=fig_data)
        fig.update_layout(
            title="3D Liver & Tumor Segmentation",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z (Slice)",
                bgcolor="black",
                xaxis=dict(backgroundcolor="black", gridcolor="gray"),
                yaxis=dict(backgroundcolor="black", gridcolor="gray"),
                zaxis=dict(backgroundcolor="black", gridcolor="gray"),
            ),
            paper_bgcolor="black",
            font=dict(color="white"),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True, key="3d_view")
    else:
        st.warning("No liver or tumor voxels found for 3D rendering.")
else:
    st.info("Run slice analysis above to enable the 3D viewer.")


# ── Section 5: Inference Latency Monitor ───────────────────────────────────────
# Tracks response time for every API request made during the session.
# Shows last, average, min, and max latency plus a rolling line chart.
st.header("⚡ Inference Latency Monitor")

if st.session_state["latency_history"]:
    latencies = st.session_state["latency_history"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Last Request", f"{latencies[-1]:.2f}s")
    with col2:
        st.metric("Average", f"{np.mean(latencies):.2f}s")
    with col3:
        st.metric("Min / Max", f"{min(latencies):.2f}s / {max(latencies):.2f}s")

    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    # gradient fill under the line
    ax.fill_between(
        range(len(latencies)),
        latencies,
        alpha=0.15,
        color="#4A90D9"
    )

    # main line
    ax.plot(
        latencies,
        color="#4A90D9",
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="#4A90D9"
    )

    # add average reference line
    avg = np.mean(latencies)
    ax.axhline(y=avg, color="#FFD700", linestyle="--", linewidth=1, alpha=0.7, label=f"Avg: {avg:.1f}s")

    # styling
    ax.set_xlabel("Request #", color="white", fontsize=11)
    ax.set_ylabel("Latency (s)", color="white", fontsize=11)
    ax.set_title("Inference Latency History", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color="white")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("No requests made yet — run a segmentation to see latency metrics.")


# ── Section 6: Model Information ───────────────────────────────────────────────
# Static reference panel describing the model architecture and performance.
st.header("ℹ️ Model Information")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Architecture")
    st.markdown("""
    - **Model:** 3D U-Net
    - **Framework:** MONAI + PyTorch
    - **Dataset:** Medical Segmentation Decathlon (Task03 Liver)
    - **Input:** 3D CT volumes (NIfTI format)
    - **Output:** 3-class mask (background / liver / tumor)
    """)

with col2:
    st.subheader("Performance")
    st.markdown("""
    - **Validation Dice Score:** 0.6968
    - **Target Dice:** 0.70 ✓
    - **Loss Function:** Tversky Loss (α=0.3, β=0.7) + DiceCELoss
    - **Inference:** Sliding window (96³ patches, Gaussian weighting)
    - **Post-processing:** KeepLargestConnectedComponent (liver + tumor)
    - **Deployment:** HF Spaces (CPU) + Streamlit Cloud
    """)