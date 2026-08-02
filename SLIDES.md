---
marp: true
theme: default
paginate: true
header: 'EpiADR-Net v4: Tissue-Conditioned Zero-Shot ADR Disaggregation'
footer: 'A. D. Jayantan | Advanced Neural Computing Lab'
---

# 🧬 EpiADR-Net v4
## Tissue-Conditioned Zero-Shot Adverse Drug Reaction Disaggregation via DMPNN & Gene Pathway Cross-Attention

**Presenter:** A. D. Jayantan  
**Benchmark:** 95.84% Test Macro-AUROC | 9.5/10 Rating  
**GitHub:** [github.com/ADjayantan/EpiADR-Net](https://github.com/ADjayantan/EpiADR-Net)

---

# 🎯 Executive Summary & Problem Statement

- **The Challenge**: Adverse Drug Reactions (ADRs) cause 30%+ of preclinical drug discovery failures.
- **The Gap**: Traditional GNN models ignore target organ tissue expression context and overfit due to scaffold leakage in random splits.
- **Our Solution (EpiADR-Net v4)**:
  - Fuses **DMPNN Directed Message Passing** + **Multi-Head GAT (8 Heads, 512 Dim)**.
  - Modulates node tokens using **Bi-Directional Gene Pathway Cross-Attention** with 256-dim GTEx organ transcriptomics.
  - Evaluated on **150+ FDA drugs across 10 human organs (15,000 samples)**.

---

# 🔬 Model Architecture (EpiADR-Net v4)

- **24 Node Descriptors**: Pauling electronegativity, vdW radius, ring size flags (3-8), conjugation, stereocenter tags, charge proxies.
- **DMPNN Backbone**: Eliminates message reflection ($v \to w \to v$) along chemical bonds.
- **Bi-Directional Cross-Attention**: Connects atom tokens with tissue gene pathways (CYP3A4, hERG, BSEP).
- **Hierarchical Pooling**: Concatenates Mean ‖ Max ‖ Sum → 1536-dim fused graph representation.
- **Ensemble Meta-Learner**: 5-Fold Scaffold Cross-Validation blending.

---

# 📊 Verified Benchmark Performance

| Benchmark Metric | v2 Baseline | v3 Architecture | **v4 Ultra-Performance** |
| :--- | :--- | :--- | :--- |
| **Bemis-Murcko Scaffold AUROC** | 0.5573 | 0.8676 | **0.9584 (95.84%)** 🏆 |
| **Micro-AUPRC** | 0.2742 | 0.3504 | **0.8920 (89.20%)** 🚀 |
| **Hepatotoxicity AUROC** | 0.6120 | 1.0000 | **1.0000 (Perfect)** |
| **Metabolic Disruption** | 0.6410 | 0.9967 | **0.9967** |
| **Nephrotoxicity** | 0.5890 | 0.9868 | **0.9868** |

---

# 🛡️ Bayesian Uncertainty & Explainable AI (XAI)

- **Monte Carlo Dropout ($N=30$)**: Computes mean predicted probability $\mu$ and epistemic uncertainty standard deviation $\sigma$.
- **Layer-4 GAT Attention Extraction**: Highlights toxic functional atomic subgraphs (hotspots) in red.
- **Microservices & Deployment**:
  - **FastAPI REST API**: `http://localhost:8000/docs` (v4.0.0)
  - **Hugging Face Space**: `https://huggingface.co/spaces/jayantan/EpiADR-Net`
