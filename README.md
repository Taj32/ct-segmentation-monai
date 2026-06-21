---
title: CT Segmentation MONAI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 3D CT Liver & Tumor Segmentation Platform

![CI](https://github.com/Taj32/ct-segmentation-monai/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![MONAI](https://img.shields.io/badge/MONAI-1.5.2-green)
![License](https://img.shields.io/badge/license-MIT-blue)

A production-grade 3D medical image segmentation system that automatically
detects and segments liver and tumors in abdominal CT scans using a U-Net
architecture trained on the Medical Segmentation Decathlon.

## 🔴 Live Demo
- **API:** https://Hipps-ct-segmentation-monai.hf.space/docs
- **Dashboard:** https://ct-segmentation-monai-k8fnmcjqfv8h2e5ehoddta.streamlit.app
- **Demo Video:** https://www.youtube.com/watch?v=GoKC0avQY2g

## 🏗️ Architecture
```markdown
CT Upload (.nii.gz)
        ↓
FastAPI /segment endpoint
        ↓
MONAI Preprocessing
(Spacingd → Orientationd → ScaleIntensityRanged)
        ↓
3D U-Net (96³ sliding window patches)
        ↓
Softmax + Threshold
(Liver > 0.5, Tumor > 0.3)
        ↓
KeepLargestConnectedComponent
        ↓
Segmentation Overlay PNG
        ↓
Streamlit Dashboard
```


## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/ct-segmentation-monai
cd ct-segmentation-monai
docker-compose up
```

Visit https://ct-segmentation-monai-k8fnmcjqfv8h2e5ehoddta.streamlit.app/ to upload a CT scan and get segmentation results.

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| Val Dice Score | 0.7023 |
| Dataset | Medical Segmentation Decathlon Task03 |
| Classes | Background / Liver / Tumor |
| Architecture | 3D U-Net (MONAI) |
| Loss Function | Tversky Loss |

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| Model | MONAI UNet, PyTorch |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Experiment Tracking | MLflow |
| Containerization | Docker + docker-compose |
| Cloud | Hugging Face Spaces |
| CI/CD | GitHub Actions |

## 📁 Project Structure
```text
ct-segmentation-monai/
├── app.py             # FastAPI endpoint
├── streamlit_app.py   # Streamlit dashboard
├── download_model.py  # HF model hub download
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── MODEL_CARD.md
└── tests/
    └── test_model.py
```

## 📋 Model Card
See [MODEL_CARD.md](MODEL_CARD.md) for full documentation including
performance metrics, limitations, and regulatory considerations.