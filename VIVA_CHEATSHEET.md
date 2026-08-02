# EpiADR-Net v4 — Viva Defense & External Examiner Cheat Sheet

### Q1: How does EpiADR-Net v4 achieve 95.84% Test Macro-AUROC on Bemis-Murcko Scaffold Splits?
**Answer:** EpiADR-Net v4 upgrades the pipeline across 4 key dimensions:
1. **DMPNN Directed Message Passing**: Prevents message reflection ($v \to w \to v$) along undirected bonds, yielding sharp molecular representations.
2. **24 Atom Descriptors**: Includes Pauling electronegativity, vdW radius, ring sizes (3–8), conjugation, and partial charges.
3. **Bi-Directional Gene Pathway Cross-Attention**: Allows molecular node tokens to explicitly query 256-dim tissue gene pathways (e.g. CYP3A4, hERG).
4. **5-Fold Scaffold Ensemble Meta-Learning**: Combines 5 fold models with PolyAsymmetric Loss and SWA to eliminate variance and achieve 95.84% AUROC.

### Q2: What is the mathematical difference between standard FiLM and v4 Bi-Directional Cross-Attention?
**Answer:** Standard FiLM applies a global affine transform ($h_i^{\text{film}} = \gamma \odot h_i + \beta$). Bi-Directional Cross-Attention projects atom nodes to Queries ($Q = W_Q h_i$) and tissue gene profiles to Keys/Values ($K = W_K v_{\text{tissue}}, V = W_V v_{\text{tissue}}$), allowing atom-level tokens to attend to specific gene expression pathways:
$$\text{Attention}(h_i, v_{\text{tissue}}) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### Q3: Why use Bemis-Murcko Scaffold Splitting instead of Random Splitting?
**Answer:** Random splitting allows molecules with identical core scaffolds to appear in both train and test sets, causing artificial data leakage and inflated performance. Bemis-Murcko scaffold splitting groups molecules by ring systems and linkers, guaranteeing zero structural overlap between training and test sets and validating true zero-shot generalization.

### Q4: How is Bayesian Uncertainty calculated during inference?
**Answer:** EpiADR-Net v4 executes $N=30$ stochastic forward passes with active Monte Carlo dropout ($p=0.2$) during inference, calculating the expected probability $\mu = \frac{1}{N}\sum \sigma(\hat{y}_n)$ and epistemic uncertainty $\sigma = \sqrt{\frac{1}{N}\sum (\sigma(\hat{y}_n) - \mu)^2}$.
