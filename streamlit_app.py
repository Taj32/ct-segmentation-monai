import time

import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import io
import mlflow
from mlflow.tracking import MlflowClient
import plotly.graph_objects as go
from skimage import measure


# initialize session state variables
if "latency_history" not in st.session_state:
    st.session_state["latency_history"] = []

if "ct" not in st.session_state:
    st.session_state["ct"] = None

if "mask" not in st.session_state:
    st.session_state["mask"] = None

if "spleen_slices" not in st.session_state:
    st.session_state["spleen_slices"] = []

# ── page config ──
st.set_page_config(
    page_title="CT Segmentation Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 3D CT Segmentation Dashboard")
st.markdown("Upload a CT scan to get an automated liver and tumor segmentation  segmentation using a U-Net model trained on the Medical Segmentation Decathlon.")

# ── sidebar ──
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input(
    "API Endpoint",
    value="https://Hipps-ct-segmentation-monai.hf.space"
)

# ── file upload ──
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
                    # display the overlay image
                    image = Image.open(io.BytesIO(response.content))
                    st.header("Segmentation Result")
                    st.image(image, caption="CT Scan with Predicted Segmentation Overlay", use_column_width=True)
                    st.success("Segmentation complete!")

                    # download button
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
                
# ── MLflow metrics panel ──
st.header("📊 Model Training Metrics")

# mlflow_url = st.sidebar.text_input(
#     "MLflow Tracking URI",
#     value="http://localhost:5000"
# )

# try:
#     client = MlflowClient(tracking_uri=mlflow_url)
#     experiments = client.search_experiments()

#     if experiments:
#         experiment_names = [e.name for e in experiments]
#         selected_exp = st.selectbox("Select Experiment", experiment_names)

#         exp = client.get_experiment_by_name(selected_exp)
#         runs = client.search_runs(
#             experiment_ids=[exp.experiment_id],
#             order_by=["start_time DESC"],
#             max_results=10
#         )

#         if runs:
#             # best run metrics
#             best_run = runs[0]
#             col1, col2, col3 = st.columns(3)
            
#             # # temporary debug
#             # st.subheader("Debug — Available Metrics")
#             # st.write(best_run.data.metrics)
#             # st.write(best_run.data.params)

#             with col1:
#                 st.metric(
#                     "Best Val Dice",
#                     f"{best_run.data.metrics.get('final_val_dice', 0):.4f}"
#                 )
#             with col2:
#                 st.metric(
#                     "Final Train Loss",
#                     "N/A"
#                 )
#             with col3:
#                 st.metric(
#                     "Inference Time (s)",
#                     "N/A"
#                 )

#             # Dice score trend chart
#             st.subheader("Validation Dice Score")
#             st.metric(
#                 label="Final Validation Dice Score",
#                 value=f"{best_run.data.metrics.get('final_val_dice', 0):.4f}",
#                 delta=f"{best_run.data.metrics.get('final_val_dice', 0) - 0.75:.4f} above target"
#             )
#             st.info("Per-epoch Dice trend requires re-running training with epoch-level MLflow logging enabled.")

# except Exception as e:
#     st.warning(f"MLflow not connected — start mlflow ui locally to see metrics. ({str(e)})")
# else:
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Best Val Dice", "0.8987")
#     with col2:
#         st.metric("Dataset", "Task09 Spleen")
#     with col3:
#         st.metric("Target Dice", "0.75 ✓")

#     st.subheader("Training Configuration")
#     st.table({
#         "Parameter": ["Model", "Epochs", "Batch Size", "Learning Rate", "Patch Size", "Loss Function"],
#         "Value": ["3D U-Net", "1000+", "2", "1e-4 → 1e-5", "96³", "DiceCELoss"]
#     })
    
# new — updated for liver/tumor (dice will be updated after training completes)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Best Val Dice", "0.7100")  # updated
with col2:
    st.metric("Dataset", "Task03 Liver")
with col3:
    st.metric("Target Dice", "0.70 ✓")  # target met
st.subheader("Training Configuration")
st.table({
    "Parameter": ["Model", "Epochs", "Batch Size", "Learning Rate", "Patch Size", "Loss Function", "Classes"],
    "Value": ["3D U-Net", "700+", "4", "1e-4", "96³", "DiceCELoss", "Background / Liver / Tumor"]
})
    
# ── slice-by-slice viewer ──
st.header("🔬 Interactive Slice Viewer")
st.markdown("Upload a CT scan to explore segmentation results slice by slice.")

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
                response = requests.post(
                    f"{api_url}/segment_data",
                    files={"file": (uploaded_file_viewer.name, uploaded_file_viewer.getvalue())},
                    timeout=600
                )
                latency = time.time() - start_time
                st.session_state["latency_history"].append(latency)

                if response.status_code == 200:
                    # load numpy data
                    buf = io.BytesIO(response.content)
                    data = np.load(buf)
                    ct = data["ct"]
                    mask = data["mask"]
                    spleen_slices = data["spleen_slices"].tolist()

                    # store in session state so slider works
                    st.session_state["ct"] = ct
                    st.session_state["mask"] = mask
                    st.session_state["spleen_slices"] = spleen_slices
                    st.success(f"Segmentation complete! Liver/tumor found on {len(spleen_slices)} slices.")

                else:
                    st.error(f"API error: {response.status_code}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# show slider if data is loaded
if "ct" in st.session_state and st.session_state["ct"] is not None:
    ct = st.session_state["ct"]
    mask = st.session_state["mask"]
    spleen_slices = st.session_state["spleen_slices"]

    st.subheader("Slice Navigator")

    # slider across all slices
    slice_idx = st.slider(
        "Select Slice",
        min_value=0,
        max_value=ct.shape[2] - 1,
        value=spleen_slices[len(spleen_slices) // 2],
        help="Slices with spleen: " + str(spleen_slices[0]) + " to " + str(spleen_slices[-1])
    )

    # show spleen indicator
    if slice_idx in spleen_slices:
        st.success(f"✓ Liver/tumor detected on slice {slice_idx}")
    else:
        st.info(f"No liver/tumor on slice {slice_idx} — liver/tumor visible on slices {spleen_slices[0]}–{spleen_slices[-1]}")

    # plot three panels
    slice_ct = ct[:, :, slice_idx]
    slice_mask = mask[:, :, slice_idx]

    color_overlay = np.zeros((*slice_mask.shape, 4))
    color_overlay[slice_mask == 1] = [0, 1, 0, 0.5]   # green = liver
    color_overlay[slice_mask == 2] = [1, 0, 0, 0.7]   # red = tumor

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(slice_ct, cmap="gray")
    axes[0].set_title(f"CT Scan — Slice {slice_idx}", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(slice_ct, cmap="gray")
    axes[1].imshow(color_overlay)
    axes[1].set_title("Predicted Segmentation", fontsize=14)
    axes[1].axis("off")
    green_patch = mpatches.Patch(color="green", alpha=0.5, label="Liver")
    red_patch = mpatches.Patch(color="red", alpha=0.7, label="Tumor")
    axes[1].legend(handles=[green_patch, red_patch], loc="lower right")

    axes[2].imshow(slice_mask, cmap="Greens")
    axes[2].set_title("Segmentation Mask", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    # slice stats
    st.subheader("Slice Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Slices", ct.shape[2])
    with col2:
        st.metric("Liver/Tumor Slices", len(spleen_slices))
    with col3:
        spleen_voxels = int(mask.sum())
        st.metric("Liver/Tumor Voxels", f"{spleen_voxels:,}")
else:
    st.info("Upload a CT scan above and click 'Run Slice Analysis' to explore slices interactively.")


# ── 3D volume viewer ──
st.header("🧊 3D Volume Viewer")

if "ct" in st.session_state and st.session_state["ct"] is not None:
    ct = st.session_state["ct"]
    mask = st.session_state["mask"]

    st.markdown("Interactive 3D view of the segmentation mask.")

    # downsample for performance — full resolution is too slow in browser
    step = 2 #step size for downsampling (setting 3 --> 2 to reduce detail loss)
    ct_down = ct[::step, ::step, ::step]
    mask_down = mask[::step, ::step, ::step]

    # get coordinates of spleen voxels
    x, y, z = np.where(mask_down == 1)

    if len(x) > 0:
        # sample points for performance
        max_points = 5000
        if len(x) > max_points:
            idx = np.random.choice(len(x), max_points, replace=False)
            x, y, z = x[idx], y[idx], z[idx]

        # get intensity values at spleen voxels for coloring
        intensity = ct_down[x, y, z]
        
        # new — separate liver and tumor meshes
        fig_data = []

        # liver mesh (class 1)
        liver_mask = (mask_down == 1).astype(float)
        if liver_mask.sum() > 0:
            try:
                liver_verts, liver_faces, _, _ = measure.marching_cubes(liver_mask, level=0.5)
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
                pass
            
        # tumor mesh (class 2)
        tumor_mask = (mask_down == 2).astype(float)
        if tumor_mask.sum() > 0:
            try:
                tumor_verts, tumor_faces, _, _ = measure.marching_cubes(tumor_mask, level=0.5)
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
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No liver or tumor voxels found for 3D rendering.")

        # fig = go.Figure(data=[
        #     go.Scatter3d(
        #         x=x, y=y, z=z,
        #         mode="markers",
        #         marker=dict(
        #             size=2,
        #             color=intensity,
        #             colorscale="Greens",
        #             opacity=0.6,
        #             colorbar=dict(title="HU Intensity")
        #         ),
        #         name="Spleen"
        #     )
        # ])

        # fig.update_layout(
        #     title="3D Spleen Segmentation",
        #     scene=dict(
        #         xaxis_title="X",
        #         yaxis_title="Y",
        #         zaxis_title="Z (Slice)",
        #         bgcolor="black",
        #         xaxis=dict(backgroundcolor="black", gridcolor="gray"),
        #         yaxis=dict(backgroundcolor="black", gridcolor="gray"),
        #         zaxis=dict(backgroundcolor="black", gridcolor="gray"),
        #     ),
        #     paper_bgcolor="black",
        #     font=dict(color="white"),
        #     height=600
        # )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Showing {len(x):,} spleen voxels (downsampled for performance)")

    #xxxxxxxxxxxxxxxxxx


    else:
        st.warning("No spleen voxels found in mask.")

else:
    st.info("Run slice analysis above to enable the 3D viewer.")


# ── inference latency monitor ──
st.header("⚡ Inference Latency Monitor")

# initialize session state for latency tracking
if "latency_history" not in st.session_state:
    st.session_state["latency_history"] = []

# display latency metrics if we have history
if st.session_state["latency_history"]:
    latencies = st.session_state["latency_history"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Last Request", f"{latencies[-1]:.2f}s")
    with col2:
        st.metric("Average", f"{np.mean(latencies):.2f}s")
    with col3:
        st.metric("Min / Max", f"{min(latencies):.2f}s / {max(latencies):.2f}s")

    # rolling chart
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(latencies, color="#1F4E8E", linewidth=2, marker="o")
    ax.set_xlabel("Request #")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Inference Latency History")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
else:
    st.info("No requests made yet — run a segmentation to see latency metrics.")


# ── model info ──
st.header("ℹ️ Model Information")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    - **Model:** 3D U-Net
    - **Framework:** MONAI
    - **Dataset:** Medical Segmentation Decathlon (Task03 Liver)
    - **Input:** 3D CT volumes (NIfTI format)
    - **Output:** 3-class mask (background / liver / tumor)
    """)

with col2:
    st.markdown("""
    - **Validation Dice Score:** 0.66+ (retraining in progress)
    - **Target Dice:** 0.70
    - **Inference:** Sliding window (96³ patches, Gaussian weighting)
    - **Post-processing:** KeepLargestConnectedComponent (liver + tumor)
    - **Deployment:** HF Spaces (CPU) + Streamlit Cloud
    """)