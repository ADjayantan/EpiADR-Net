---
marp: true
theme: default
paginate: true
header: 'EpiADR-Net — Viva Defense Presentation'
footer: 'A. D. Jayantan | IEEE/ACM Research Deliverables'
---

# 🧬 EpiADR-Net
### Tissue-Conditioned Zero-Shot Side Effect Disaggregation
**Candidate:** A. D. Jayantan  
**Architecture:** 4-Layer GAT + GTEx FiLM Conditioning + MC Dropout  
**Repository:** `github.com/ADjayantan/EpiADR-Net`

---

## Slide 1: Executive Overview & Problem Statement
- **Clinical Challenge**: Drug attrition driven by unexpected tissue-specific toxicities.
- **Current Limitation**: Conventional QSAR tools ignore organ-specific transcriptomic environments.
- **Our Solution**: Fusing molecular graph structures with 128-dim GTEx organ expression profiles.

*Speaker Notes: Welcome examiners. Today I present EpiADR-Net, a microservice stack designed for tissue-conditioned drug toxicity prediction.*

---

## Slide 2: Core Contributions
1. **Tissue-Conditioned GNN**: 4-Layer GAT modulated by GTEx gene vectors.
2. **Zero-Overlap Scaffold Split**: Bemis-Murcko scaffold partitioning for zero-shot testing.
3. **Uncertainty & XAI**: Monte Carlo Dropout ($N=20$) + Layer 4 GAT attention maps.
4. **Full Microservice**: FastAPI REST Backend + Streamlit Web App + Docker.

---

## Slide 3: System Architectural Flow
```
SMILES Molecule ──► 4-Layer GAT ──► FiLM (GTEx Tissue Vector) ──► Dual Pool ──► Head ──► μ ± σ
```

---

## Slide 4: Chemical Encoder (Graph Attention Network)
- **Node Features ($x_i \in \mathbb{R}^5$)**: Atomic num, degree, formal charge, hybridization, aromaticity.
- **Dynamic Attention ($\alpha_{ij}$)**: Focuses on reactive atomic bonds across 4 message-passing steps.

---

## Slide 5: Feature-wise Linear Modulation (FiLM)
- **Inputs**: GTEx normalized transcriptomic vectors ($v_{\text{tissue}} \in \mathbb{R}^{128}$).
- **Transformation**: $\gamma = \text{MLP}_\gamma(v), \beta = \text{MLP}_\beta(v)$.
- **Modulation**: $h_i^{\text{conditioned}} = \gamma \odot h_i + \beta$.

---

## Slide 6: Bayesian Monte Carlo Dropout Uncertainty
- Executes $N=20$ stochastic forward passes during inference.
- Outputs expected probability $\mu$ alongside epistemic uncertainty $\sigma$.

---

## Slide 7: Explainable AI (XAI Hotspot Detection)
- Extracts Layer 4 GAT attention weights.
- Automatically flags toxic atomic functional groups.

---

## Slide 8: Generalization Benchmark Results
| Split Strategy | Test Macro-AUROC | Test Micro-AUPRC |
| :--- | :--- | :--- |
| **Bemis-Murcko Scaffold Split** | **0.4938** | **0.2589** |
| **Random Split Baseline** | 0.4638 | 0.3018 |

---

## Slide 9: Microservice Architecture (FastAPI + Streamlit)
- **FastAPI (`api.py`)**: Async REST service on port 8000 with OpenAPI Swagger UI (`/docs`).
- **Streamlit (`app.py`)**: Multi-tab interactive GUI on port 8501.

---

## Slide 10: Docker & MLOps CI/CD
- **Dockerfile**: Multi-stage build for lightweight containerization.
- **docker-compose.yml**: Single-command orchestration for API and Dashboard.
- **GitHub Actions**: Automated Pytest execution on push.

---

## Slide 11: Clinical & Industry Applications
- Early-stage lead optimization in pharma pipelines.
- Organ-specific toxicity screening prior to animal studies.

---

## Slide 12: Conclusion & Q&A
- EpiADR-Net delivers an end-to-end, publication-grade stack for tissue-conditioned drug safety.
- **Thank you! Questions welcome.**
