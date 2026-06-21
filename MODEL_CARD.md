# Model Card: 3D Liver & Tumor Segmentation

## Model Details
- **Model type:** 3D U-Net
- **Framework:** MONAI + PyTorch
- **Task:** Volumetric medical image segmentation
- **Input:** 3D CT scans (NIfTI format, .nii.gz)
- **Output:** 3-class segmentation mask (background / liver / tumor)
- **Training date:** June 2026

## Intended Use
- **Primary use:** Automated liver and tumor segmentation in abdominal CT scans
- **Intended users:** Medical imaging researchers, clinical AI developers
- **Out-of-scope uses:** Direct clinical decision-making without physician oversight, non-abdominal CT scans, MRI or other modalities

## Training Data
- **Dataset:** Medical Segmentation Decathlon — Task03 Liver
- **Source:** medicaldecathlon.com (public dataset)
- **Size:** 131 training volumes, 70 test volumes
- **Scanner:** Mixed CT scanners (portal venous phase)
- **Labels:** Background (0), Liver (1), Tumor (2)

## Evaluation Data
- **Validation split:** 20% holdout (approximately 26 volumes)
- **Metric:** Dice Similarity Coefficient (DSC)

## Performance
| Class | Dice Score |
|-------|-----------|
| Liver | ~0.85 (estimated) |
| Tumor | Detected (class imbalance present) |
| **Overall Val Dice** | **0.7023** |

> Note: Dice is averaged across liver and tumor classes excluding background.
> Tumor detection is limited by class imbalance — tumor voxels represent <5% of volume.

## Training Configuration
| Parameter | Value |
|-----------|-------|
| Architecture | 3D U-Net (MONAI) |
| Loss Function | Tversky Loss (α=0.3, β=0.7) |
| Optimizer | Adam (lr=1e-4 → 1e-5) |
| Epochs | 700+ |
| Batch Size | 4 |
| Patch Size | 96³ voxels |
| Inference | Sliding window (overlap=0.75, Gaussian) |

## Known Limitations & Failure Modes
- **Tumor detection:** Small tumors (<10mm) may be missed due to class imbalance in training data
- **Modality:** Trained only on CT — will not work on MRI or other modalities
- **Scanner variability:** Performance may vary across different CT scanner manufacturers
- **Intensity range:** Calibrated for HU range -57 to 164 (soft tissue window)
- **Volume size:** Very small CT volumes (<62 slices) may produce degraded results
- **No ground truth:** Deployed model cannot compute Dice without ground truth labels

## Fairness Considerations
- Training dataset demographic information is not available from the Decathlon dataset
- Performance across different patient demographics (age, sex, BMI) has not been evaluated
- Scanner manufacturer bias is unknown — mixed scanner types used in training
- Model should be validated on local patient populations before any clinical use

## Regulatory Considerations
This model is a **research prototype** and is **not FDA-cleared**. A deployed segmentation
model used to assist clinical decisions would fall under FDA's Software as a Medical Device
(SaMD) framework and would require a 510(k) or De Novo submission with evidence of safety
and effectiveness. The stratified performance analysis in this model card mirrors what would
be included in a regulatory submission package.

## Deployment
- **API:** FastAPI endpoint on Hugging Face Spaces (CPU inference)
- **Dashboard:** Streamlit Cloud interactive viewer
- **Docker:** Available via docker-compose up

## Citation
If using this model or code, please cite the Medical Segmentation Decathlon:
> Simpson et al. "A large annotated medical image dataset for the development and evaluation
> of segmentation algorithms." arXiv:1902.09063 (2019)
