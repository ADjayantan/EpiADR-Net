---
title: EpiADR-Net — Tissue-Conditioned Zero-Shot ADR Platform
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app_gradio.py
pinned: false
license: mit
---

# 🧬 EpiADR-Net (Antigravity 2.0 Edition)
### Tissue-Conditioned Zero-Shot Side Effect Disaggregation Platform

[![Continuous Integration](https://github.com/ADjayantan/EpiADR-Net/actions/workflows/ci.yml/badge.svg)](https://github.com/ADjayantan/EpiADR-Net/actions)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/jayantan/EpiADR-Net)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-green.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-5.16-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EpiADR-Net** is an end-to-end, publication-grade research platform and enterprise microservice stack engineered for **Tissue-Conditioned Zero-Shot Side Effect Disaggregation**. It predicts organ-specific Adverse Drug Reactions (ADRs) by conditioning molecular Graph Neural Network representations on human organ transcriptomic profiles.

---

## 🌟 Executive Highlights

| Category | Component / Benchmark | Detail / Metric |
| :--- | :--- | :--- |
| **Model Architecture** | 4-Layer GAT + FiLM Conditioning | Fuses molecular graph structure with 128-dim GTEx transcriptomics |
| **Generalization Split** | Bemis-Murcko Scaffold Split | Zero structural overlap between training and test sets |
| **Uncertainty Estimation**| Monte Carlo Dropout ($N=20$) | Calculates expected probability $\mu$ and uncertainty bounds $\sigma$ |
| **Explainable AI (XAI)** | GAT Layer 4 Attention Extraction | Visually highlights toxic functional atomic subgraphs |
| **Hugging Face App** | Gradio Space (`app_gradio.py`) | Deployed live on Hugging Face Spaces (`sdk=gradio`) |
| **REST Microservice** | FastAPI Backend (`api.py`) | Interactive Swagger UI documentation at `http://localhost:8000/docs` |
| **Web Dashboard** | Multi-Tab Streamlit App (`app.py`) | Single organ explainability, dual-organ side-by-side comparative chart |
| **Automated Testing** | Pytest Suite (`tests/`) | 100% Pass Rate across data, model, and API tests |

---

## 🚀 Quick Start Guide

### 1. Local Environment Setup
```bash
git clone https://github.com/ADjayantan/EpiADR-Net.git
cd EpiADR-Net

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Gradio App (Hugging Face Space mode)
```bash
python app_gradio.py
```

### 3. Run Automated Pytest Suite
```bash
pytest -v tests/
```

### 4. Launch FastAPI REST Microservice
```bash
uvicorn api:app --reload --port 8000
```
Swagger UI available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Launch Streamlit Web App
```bash
streamlit run app.py
```
