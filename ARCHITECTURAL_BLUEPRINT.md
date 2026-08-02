# EpiADR-Net Architectural Blueprint & Tensor Operations

This document outlines the detailed tensor dimensions, layer specifications, and mathematical operations across the EpiADR-Net neural pipeline.

---

## 📐 Tensor Operations Map

```
Input SMILES String
       │
       ▼
Atom Feature Extraction ─────────► Node Matrix X  : [N, 5]
Edge Connectivity       ─────────► Edge Index E   : [2, M]
GTEx Tissue Profile     ─────────► Tissue Vector V: [B, 128]
       │
       ▼
Linear Input Projection ─────────► h_0             : [N, 128]
       │
       ▼
4-Layer GAT Backbone    ─────────► h_4             : [N, 128]
                                   Attention α_4   : [M]
       │
       ▼
FiLM Modulation Layer   ─────────► γ, β            : [B, 128] -> [N, 128]
                                   h_conditioned   : [N, 128]
       │
       ▼
Dual Graph Pooling      ─────────► [Mean || Max]   : [B, 256]
       │
       ▼
10-Class Classification ─────────► Logits          : [B, 10]
                                   Probabilities μ : [B, 10]
                                   Uncertainty σ   : [B, 10]
```

---

## 🔬 Layer Specifications

| Layer Name | Input Shape | Output Shape | Parameters / Operations |
| :--- | :--- | :--- | :--- |
| **Input Linear** | `[N, 5]` | `[N, 128]` | $W_{\text{in}} \in \mathbb{R}^{5 \times 128}$, Bias |
| **GAT Layer 1-4** | `[N, 128]` | `[N, 128]` | $W_{\text{gat}} \in \mathbb{R}^{128 \times 128}, a \in \mathbb{R}^{256 \times 1}$, LeakyReLU |
| **Gene Pathway Cross-Attn**| `[N, 1536]` | `[N, 1536]` | $\text{MultiHeadCrossAttn}(\mathbf{Q}_{1536}, \mathbf{K}_{1024}, \mathbf{V}_{1024})$ |
| **FiLM Residual Update** | `[N, 1536]` | `[N, 1536]` | $h_i^{\text{cond}} = h_i \odot (1.0 + \gamma_{\text{nodes}}) + \beta_{\text{nodes}}$ |
| **Dual Graph Pool**| `[N, 128]` | `[B, 256]` | MeanPool `[B, 128]` \|\| MaxPool `[B, 128]` |
| **Classifier FC1** | `[B, 256]` | `[B, 128]` | Linear(256->128), ReLU, Dropout(0.2) |
| **Classifier FC2** | `[B, 128]` | `[B, 10]` | Linear(128->10) |
