# Data Provenance & Ingestion Log

This document records all external data sources, exact URLs, versions, download dates, licensing, and processing rules used for building real ADR datasets and GTEx tissue profiles in EpiADR-Net.

---

## 1. SIDER 4.1 Side Effect Resource
- **Source Name:** SIDER: Side Effect Resource (Version 4.1)
- **Publisher / Organization:** European Molecular Biology Laboratory (EMBL)
- **URL:** [http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz](http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz)
- **Secondary Resource:** SIDER Drug Names ([http://sideeffects.embl.de/media/download/drug_names.tsv](http://sideeffects.embl.de/media/download/drug_names.tsv))
- **Download Date:** 2026-08-02
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **MedDRA Target Mapping:**
  - `Hepatotoxicity` -> MedDRA Concept / Preferred Term matching hepatic injury / liver disorders
  - `Arrhythmia` -> MedDRA Concept / Preferred Term matching cardiac arrhythmia
  - `Seizure` -> MedDRA Concept / Preferred Term matching seizure / convulsion
  - `Nephrotoxicity` -> MedDRA Concept / Preferred Term matching renal impairment / nephrotoxicity
  - `Pneumonitis` -> MedDRA Concept / Preferred Term matching pneumonitis / pulmonary toxicity
  - `Nausea` -> MedDRA Concept / Preferred Term matching nausea
  - `Headache` -> MedDRA Concept / Preferred Term matching headache
  - `Dizziness` -> MedDRA Concept / Preferred Term matching dizziness
  - `Fatigue` -> MedDRA Concept / Preferred Term matching fatigue
  - `Rash` -> MedDRA Concept / Preferred Term matching rash / dermatological reaction
- **Compound Resolution:** Drug identifiers / names resolved to canonical SMILES via PubChem REST PUG API (`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/`). Unresolvable or invalid SMILES strings dropped and logged.
- **Organ Assignment Rule:** Organ targets (Liver, Heart, Brain, Kidney, Lung) mapped systematically using MedDRA System Organ Class (SOC) primary physiological organ targets and WHO ATC organ classification.

---

## 2. GTEx Analysis Release V8 (RNA-Seq Transcriptomics)
- **Source Name:** Genotype-Tissue Expression (GTEx) Project (Analysis Release V8)
- **Publisher / Organization:** The Broad Institute of MIT and Harvard / NIH Common Fund
- **URL:** [https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz](https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz)
- **Download Date:** 2026-08-02
- **License:** Open Access (GTEx Data Use Agreement / NIH Public Domain)
- **Organ Granularity Aggregation:**
  - **Liver**: `Liver`
  - **Heart**: Average of `Heart - Atrial Appendage` and `Heart - Left Ventricle`
  - **Brain**: Average of GTEx brain sub-region columns (`Brain - Cortex`, `Brain - Cerebellum`, `Brain - Hippocampus`, `Brain - Caudate`, `Brain - Substantia nigra`, `Brain - Amygdala`, `Brain - Anterior cingulate cortex`, `Brain - Hypothalamus`, `Brain - Nucleus accumbens`, `Brain - Putamen`, `Brain - Spinal cord`, `Brain - Subthalamic nucleus`)
  - **Kidney**: `Kidney - Cortex`
  - **Lung**: `Lung`
- **Dimensionality Reduction:** Scikit-Learn `PCA(n_components=128, random_state=42)` fitted across genes.
