# EpiADR-Net Presentation & Slide Walkthrough Guide

This guide provides a script for presenting EpiADR-Net during a thesis defense, viva exam, or conference presentation.

---

## 🎤 Presentation Timing & Structure (15 Minutes)

### Part 1: Problem & Motivation (3 Mins)
- **Slide 1 & 2**: Introduce drug attrition rates and the missing dimension of transcriptomic tissue context in traditional QSAR approaches.
- **Key Message**: "Drugs don't act in a vacuum; their toxicity depends on organ-specific expression profiles."

### Part 2: Architecture & Mathematical Formulation (5 Mins)
- **Slides 3, 4, 5, 6 & 7**: Walk through the 4-layer GAT, GTEx FiLM modulation ($\gamma, \beta$), Dual Graph Pooling, MC Dropout ($N=20$), and XAI Layer 4 attention hotspots.
- **Key Message**: "FiLM allows organ gene profiles to dynamically alter molecular graph representations before classification."

### Part 3: Benchmark Results & Validation (4 Mins)
- **Slide 8**: Discuss the Bemis-Murcko scaffold split vs random split. Highlight zero structural overlap between training and test sets.

### Part 4: Enterprise Stack & Live Demo (3 Mins)
- **Slides 9, 10, 11 & 12**: Show the FastAPI Swagger UI backend, Streamlit multi-tab dashboard, Docker compose setup, and GitHub Actions CI pipeline.
