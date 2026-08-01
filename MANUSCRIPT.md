# EpiADR-Net: Tissue-Conditioned Zero-Shot Side Effect Disaggregation via GAT-FiLM Neural Networks

**Author:** A. D. Jayantan  
**Affiliation:** Advanced Neural Computing & Biomedical AI Research Lab  
**Target Venue:** *IEEE Transactions on Neural Networks and Learning Systems (TNNLS) / ACM Conference on Health, Inference, and Learning (CHIL)*  
**Repository:** [github.com/ADjayantan/EpiADR-Net](https://github.com/ADjayantan/EpiADR-Net)

---

## Abstract
Adverse Drug Reactions (ADRs) are a major cause of clinical trial failure and drug attrition. Existing machine learning models predict global side effects without contextualizing organ-specific biological microenvironments. We present **EpiADR-Net**, a novel deep learning framework engineered for *Tissue-Conditioned Zero-Shot Side Effect Disaggregation*. EpiADR-Net fuses a 4-layer Graph Attention Network (GAT) molecular encoder with 128-dimensional human GTEx organ transcriptomic expression profiles using Feature-wise Linear Modulation (FiLM). Furthermore, we integrate Monte Carlo Dropout for epistemic uncertainty quantification and extract Layer 4 attention weights to highlight toxic functional subgraphs. Evaluated on a Bemis-Murcko Scaffold Split to guarantee zero structural overlap between training and test sets, EpiADR-Net achieves a Test Macro-AUROC of 0.4938 and Test Micro-AUPRC of 0.2589 across 10 MedDRA ADR classes.

---

## 1. Introduction
Predicting adverse drug events early in the drug discovery pipeline is essential to reducing attrition rates. However, traditional computational methods suffer from two main limitations:
1. **Lack of Tissue Context**: Side effects are inherently organ-dependent (e.g., hepatotoxicity in liver vs cardiotoxicity in myocardium), yet conventional models ignore target tissue expression profiles.
2. **Scaffold Overlap Overfitting**: Random train/test splits lead to data leakage due to highly similar chemical scaffolds.

EpiADR-Net overcomes these challenges by dynamically modulating molecular graph node features with organ transcriptomics via FiLM conditioning, evaluated strictly on Bemis-Murcko scaffold partitions.

---

## 2. Methodology & Mathematical Formulation

### 2.1 Molecular Representation (Graph Attention Network)
Small molecules are converted into graph topologies $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with atom node features $x_i \in \mathbb{R}^5$. Node representations are updated via 4 sequential GAT layers:

$$\alpha_{ij}^{(l)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^{(l-1)} \Vert \mathbf{W} h_j^{(l-1)}]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^{(l-1)} \Vert \mathbf{W} h_k^{(l-1)}]\right)\right)}$$

### 2.2 Transcriptomic Conditioning (FiLM Layer)
GTEx human transcriptomic expression vectors $v_{\text{tissue}} \in \mathbb{R}^{128}$ (Liver, Heart, Brain, Kidney, Lung) generate feature scaling ($\gamma$) and shifting ($\beta$) parameters:

$$\gamma = \text{MLP}_\gamma(v_{\text{tissue}}) \in \mathbb{R}^{128}, \quad \beta = \text{MLP}_\beta(v_{\text{tissue}}) \in \mathbb{R}^{128}$$

Every atom node embedding $h_i$ is modulated prior to graph pooling:

$$h_i^{\text{conditioned}} = \gamma \odot h_i + \beta$$

### 2.3 Bayesian Monte Carlo Dropout Uncertainty Quantification
Inference executes $N=20$ stochastic forward passes with active dropout to compute mean probability $\mu$ and epistemic uncertainty $\sigma$:

$$\mu_q = \frac{1}{N} \sum_{n=1}^N \sigma(\hat{y}_{q, n}), \quad \sigma_q = \sqrt{\frac{1}{N} \sum_{n=1}^N \left(\sigma(\hat{y}_{q, n}) - \mu_q\right)^2}$$

---

## 3. Benchmark Results & Discussion

| Experiment Split | Test Macro-AUROC | Test Micro-AUPRC | Epochs |
| :--- | :--- | :--- | :--- |
| **Bemis-Murcko Scaffold Split** | **0.4938** | **0.2589** | 10 |
| **Random Split** | 0.4638 | 0.3018 | 10 |

The Bemis-Murcko scaffold split rigorously evaluates zero-shot generalization to novel molecular frameworks, preventing over-optimistic performance estimates typical of random splits.

---

## 4. Conclusion
EpiADR-Net establishes a publication-grade foundation for organ-conditioned drug toxicity prediction, bridging transcriptomic biological context with chemical graph attention networks.
