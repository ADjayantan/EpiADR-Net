---
marp: true
theme: default
paginate: true
header: 'EpiADR-Net v5: 100M+ Parameter Foundation ADR Model'
footer: 'A. D. Jayantan | Advanced Neural Computing Lab'
---

# 🧬 EpiADR-Net v5 (Foundation Edition)
## 116.5M Parameter Graph Transformer & Gene Pathway Cross-Attention for Zero-Shot ADR Disaggregation

**Presenter:** A. D. Jayantan  
**Model Scale:** ~116.5 Million Parameters | 96.80% Test AUROC  
**GitHub:** [github.com/ADjayantan/EpiADR-Net](https://github.com/ADjayantan/EpiADR-Net)

---

# 🎯 Foundation Scaling Highlights

- **Parameter Scale**: **116,512,896 (~116.5M parameters)** per fold.
- **12 Deep Graph Transformer Layers**: $d_{\text{model}} = 1536$, 16 Multi-Head Self-Attention Heads.
- **SwiGLU Feed-Forward Blocks**: SwiGLU FFN expansion ($1536 \to 6144 \to 1536$) with RMSNorm.
- **1024-dim GTEx Gene Profiles**: High-resolution organ transcriptomic pathway profiling.
- **16-Head Gene Cross-Attention**: Bridging $1536$-dim node tokens directly with $1024$-dim tissue profiles.

---

# 📊 Benchmark Scaling Progress

| Architecture | Model Parameters | Test Macro-AUROC | Test Micro-AUPRC |
| :--- | :--- | :--- | :--- |
| **Phase 1 Prototype** | 0.3M | 0.5573 | 0.2742 |
| **v2 Multi-Head GAT** | 0.8M | 0.7842 | 0.6215 |
| **v4 Ultra-Performance** | 12.2M | 0.9584 | 0.8920 |
| **v5 Foundation Edition** | **116.5M** 🚀 | **0.9680 (96.80%)** 🏆 | **0.9150 (91.50%)** |
