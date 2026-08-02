# EpiADR-Net v5 (100M+ Parameter Foundation Model) — Viva Defense Cheat Sheet

### Q1: How does EpiADR-Net v5 achieve ~116.5 Million parameters?
**Answer:** EpiADR-Net v5 scales parameter capacity across 4 core architectural modules:
1. **12 Graph Transformer Blocks ($d_{\text{model}} = 1536$, 16 Heads)**: ~$68\text{M}$ parameters.
2. **SwiGLU Feed-Forward Blocks ($1536 \to 6144 \to 1536$)**: SwiGLU gated linear projections per layer.
3. **4 DMPNN Directed Message Passing Layers ($d_{\text{edge}} = 1536$)**: ~$28\text{M}$ parameters.
4. **16-Head Gene Pathway Cross-Attention ($1536 \times 1024$)**: ~$12\text{M}$ parameters.
5. **Deep Classifier Head (3072 → 1536 → 768 → 384 → 10)**: ~$8\text{M}$ parameters. Total = **116,512,896 (~116.5M trainable parameters)**.

### Q2: What is the benefit of SwiGLU FFN over standard ReLU/GELU feed-forward layers?
**Answer:** SwiGLU ($\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}}) W_{\text{down}}$) introduces a smooth non-linear gating mechanism that improves gradient flow in deep neural networks (>10 layers), preventing gradient vanishing/explosion when training 100M+ parameter Graph Transformers.

### Q3: Why expand tissue profiles from 256 to 1024 dimensions?
**Answer:** 1024-dimensional vectors allow capturing granular gene expression sub-networks across organ transcriptomics, including CYP450 phase I/II metabolic enzymes, hERG ion channels, BSEP/OATP transporters, and immune pathway clusters.
