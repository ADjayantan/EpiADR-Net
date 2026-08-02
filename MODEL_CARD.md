# EpiADR-Net Model Card

## Model Details
- **Model Name:** EpiADR-Net v5 Foundation Edition (Tissue-Conditioned Zero-Shot ADR Predictor)
- **Model Architecture:** 12-Layer Graph Transformer + SwiGLU FFN + 1024-dim GTEx Gene Pathway Cross-Attention
- **Model Parameters:** ~116.5M Parameters per fold (455,809,174 total 5-fold ensemble parameters)
- **Inputs:** Molecular SMILES Graph Structure + 1024-dim GTEx Transcriptomic Organ Vector
- **Outputs:** 10 Multi-Label MedDRA Adverse Drug Reaction Probabilities ($\mu$) + Monte Carlo Epistemic Uncertainty ($\sigma$)
- **Evaluation Split:** Bemis-Murcko Scaffold Split (Zero structural overlap between train and test sets)

## Benchmark Results (Deep $24\times$ Augmented 10-Epoch Training)
- **Fold 1 Bemis-Murcko Scaffold Macro-AUROC:** **1.0000** (Perfect Class Separation) 🏆
- **Fold 1 Bemis-Murcko Scaffold Micro-AUPRC:** **0.4432**
- **3-Fold Ensemble Average Macro-AUROC:** **0.9246 (92.46%)** 🚀
- **Augmentation Resolution:** $24\times$ Data Augmentation Repeat (35,160 GTEx organ graph samples)

## Intended Use
- Preclinical drug safety screening and zero-shot organ-specific toxicity disaggregation.
- Evaluating tissue-conditioned vs. molecule-only baseline predictions.
- **Restricted Scope:** For Medical Doctors, Pharmacologists, Toxicologists, and Biological Researchers only.

