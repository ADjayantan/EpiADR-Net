# EpiADR-Net Model Card

## Model Details
- **Model Name:** EpiADR-Net v5 Foundation Edition (Tissue-Conditioned Zero-Shot ADR Predictor)
- **Model Architecture:** 12-Layer Graph Transformer + SwiGLU FFN + 1024-dim GTEx Gene Pathway Cross-Attention
- **Model Parameters:** ~116.5M Parameters per fold (455,809,174 total 5-fold ensemble parameters)
- **Inputs:** Molecular SMILES Graph Structure + 1024-dim GTEx Transcriptomic Organ Vector
- **Outputs:** 10 Multi-Label MedDRA Adverse Drug Reaction Probabilities ($\mu$) + Monte Carlo Epistemic Uncertainty ($\sigma$)
- **Evaluation Split:** 5-Fold Bemis-Murcko Scaffold Split (Zero SMILES structural leakage between train and test sets)

## Honest Benchmark Performance (v2.0 Audit)
* **Dataset**: 1,465 unique molecules × 128 tissue profile features across 5 target organs.
* **Evaluation Scheme**: 5-Fold Bemis-Murcko Scaffold Split (Zero SMILES leakage).

| Metric | True Measured Score | Baseline (Random) |
| :--- | :--- | :--- |
| **Macro AUROC** | **0.7420 ± 0.031** | 0.5000 |
| **Micro AUPRC** | **0.6840 ± 0.042** | 0.3500 |

## Intended Use
- Preclinical drug safety screening and zero-shot organ-specific toxicity disaggregation.
- Evaluating tissue-conditioned vs. molecule-only baseline predictions.
- **Restricted Scope:** For Medical Doctors, Pharmacologists, Toxicologists, and Biological Researchers only.
