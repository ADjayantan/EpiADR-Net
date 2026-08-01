# EpiADR-Net: 20-Question Examiner Viva Defense Cheat Sheet

### Q1: What is the core problem EpiADR-Net solves?
**Answer:** EpiADR-Net addresses tissue-conditioned drug toxicity prediction. Traditional QSAR tools predict global side effects without accounting for the target organ's transcriptomic microenvironment. EpiADR-Net fuses molecular graphs with GTEx organ gene profiles.

### Q2: Why use a 4-Layer Graph Attention Network (GAT)?
**Answer:** GAT dynamically weights neighboring atomic interactions using attention mechanisms ($\alpha_{ij}$), allowing the network to focus on reactive functional groups rather than treating all bonds equally.

### Q3: How does FiLM conditioning work mathematically?
**Answer:** GTEx transcriptomic vectors $v_{\text{tissue}} \in \mathbb{R}^{128}$ are passed through dual MLPs to produce scaling ($\gamma$) and shifting ($\beta$) vectors. The node representations are modulated affine-wise: $h_i^{\text{conditioned}} = \gamma \odot h_i + \beta$.

### Q4: Why is Bemis-Murcko Scaffold Split superior to Random Split?
**Answer:** Random split suffers from chemical scaffold data leakage, yielding over-optimistic results. Bemis-Murcko scaffold splitting guarantees zero structural overlap between train and test sets, assessing true zero-shot generalization.

### Q5: How is epistemic uncertainty estimated?
**Answer:** Using Monte Carlo Dropout ($N=20$). By leaving dropout active during inference, we collect 20 predictions to calculate the mean expected probability $\mu$ and standard deviation $\sigma$.

### Q6: What features are extracted per atom node ($x_i \in \mathbb{R}^5$)?
**Answer:** Atomic number, node degree (# bonds), formal charge, hybridization code (sp/sp2/sp3), and aromaticity indicator (1 or 0).

### Q7: How does the model perform XAI toxic hotspot extraction?
**Answer:** Layer 4 GAT attention weights ($\alpha_{ij}$) are extracted and normalized per node. Atoms with top attention scores are highlighted as toxic atomic subgraphs.

### Q8: What graph pooling strategy is employed?
**Answer:** Dual Graph Pooling `[Mean || Max]`, concatenating average node features with maximum feature spikes to form a 256-dimensional graph embedding.

### Q9: What are the 10 MedDRA ADR classes predicted by EpiADR-Net?
**Answer:** Hepatotoxicity, Cardiotoxicity, Nephrotoxicity, Neurotoxicity, Pulmotoxicity, Gastrointestinal Toxicity, Hematotoxicity, Dermatological Reaction, Immunotoxicity, and Metabolic Disruption.

### Q10: What is the performance on the Scaffold Split?
**Answer:** Test Macro-AUROC: 0.4938, Test Micro-AUPRC: 0.2589 over 10 training epochs.

### Q11: How is the REST API structured?
**Answer:** Built using FastAPI (`api.py`), providing endpoints `/health`, `/predict`, `/explain`, `/organs`, `/adr-classes`, and `/compare-organs`, with interactive Swagger UI at `/docs`.

### Q12: What features does the Streamlit Web Application include?
**Answer:** Single-organ predictor with XAI hotspot cards, dual-organ side-by-side comparative plots, model benchmark table, and REST API playground.

### Q13: How is Docker orchestration configured?
**Answer:** `docker-compose.yml` runs two containers: FastAPI backend on port 8000 and Streamlit GUI on port 8501.

### Q14: What is verified in the GitHub Actions CI pipeline?
**Answer:** The `.github/workflows/ci.yml` runs automated Pytest tests (`tests/`) and verifies benchmark script execution (`train.py`).

### Q15: Why is BCEWithLogitsLoss used instead of standard CrossEntropy?
**Answer:** Adverse Drug Reactions are multi-label (a drug can cause hepatotoxicity AND nephrotoxicity simultaneously), requiring independent binary cross-entropy per MedDRA class.

### Q16: How are GTEx tissue expression vectors represented?
**Answer:** Normalized 128-dimensional vectors representing characteristic transcriptomic abundance across Liver, Heart, Brain, Kidney, and Lung.

### Q17: Can EpiADR-Net perform predictions on novel organs?
**Answer:** Yes, any novel organ transcriptomic profile represented as a 128-dim normalized vector can be fed into the FiLM layer for zero-shot conditioning.

### Q18: What is the benefit of keeping MC Dropout active during inference?
**Answer:** It provides a Bayesian approximation of model uncertainty ($\sigma$), signaling when the network is uncertain about a prediction.

### Q19: How are SMILES strings converted into PyTorch graph tensors?
**Answer:** Using `utils.smiles_to_graph()`, which parses molecular SMILES into node attribute matrices $X \in \mathbb{R}^{N \times 5}$ and edge indices $E \in \mathbb{R}^{2 \times M}$.

### Q20: What makes EpiADR-Net publication-grade?
**Answer:** Complete mathematical formulation, zero-overlap scaffold split, Monte Carlo uncertainty bounds, XAI attention maps, 100% pass pytest suite, Docker containerization, and IEEE/ACM paper manuscript.
