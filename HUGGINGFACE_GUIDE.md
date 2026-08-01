# 🔗 GitHub & Hugging Face Spaces Connection Guide

This guide explains how to connect your local repository and GitHub repository ([`github.com/ADjayantan/EpiADR-Net`](https://github.com/ADjayantan/EpiADR-Net)) directly to your Hugging Face Space ([`huggingface.co/spaces/jayantan/EpiADR-Net`](https://huggingface.co/spaces/jayantan/EpiADR-Net)).

---

## ⚡ Option A: Push Directly from Local Machine to Hugging Face
Run these commands in PowerShell inside `C:\Users\adjay\.gemini\antigravity\scratch\EpiADR-Net`:
```powershell
git remote add hf https://huggingface.co/spaces/jayantan/EpiADR-Net
git push hf main
```
*When prompted for a password, enter your Hugging Face Access Token with Write permissions.*

---

## ⚡ Option B: Automated Sync via GitHub Actions
1. Get your Hugging Face Access Token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Role: **Write**).
2. Open your GitHub Repository: `https://github.com/ADjayantan/EpiADR-Net`
3. Go to **Settings** -> **Secrets and variables** -> **Actions**.
4. Click **New repository secret**:
   - **Name**: `HF_TOKEN`
   - **Secret**: Paste your Hugging Face access token (`hf_...`).
5. Every `git push` to `main` branch will automatically sync to Hugging Face!

---

## 📱 App Configuration on Hugging Face
The root `README.md` includes the Hugging Face Space metadata header:
```yaml
---
title: EpiADR-Net — Tissue-Conditioned Zero-Shot ADR Platform
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 3.35.0
app_file: app_gradio.py
pinned: false
license: mit
---
```
When Hugging Face detects this frontmatter, it automatically installs `requirements.txt` and launches [`app_gradio.py`](file:///C:/Users/adjay/.gemini/antigravity/scratch/EpiADR-Net/app_gradio.py)!
