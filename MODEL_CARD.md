# EpiADR-Net Model Card

## Model Details
- **Model Name:** EpiADR-Net (Tissue-Conditioned Zero-Shot ADR Predictor)
- **Model Architecture:** 4-Layer GAT Backbone + FiLM Tissue Conditioning + Gene Pathway Cross-Attention
- **Inputs:** Molecular SMILES Graph Structure + 128-dim GTEx Transcriptomic Organ Vector
- **Outputs:** 10 Multi-Label MedDRA Adverse Drug Reaction Probabilities ($\mu$) + Uncertainty ($\sigma$)
- **Evaluation Split:** Bemis-Murcko Scaffold Split (Zero structural overlap between train and test sets)

## Intended Use
- Preclinical drug safety screening and zero-shot organ-specific toxicity disaggregation.
- Evaluating tissue-conditioned vs. molecule-only baseline predictions.
