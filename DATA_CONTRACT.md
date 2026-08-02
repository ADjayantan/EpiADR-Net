# EpiADR-Net Data Contract Specification

This document defines the strict validation schema and data integrity constraints enforced for all input data files consumed by EpiADR-Net.

---

## 1. Organ Transcriptomic Profiles Schema (`data/tissue_profiles.csv`)

| Property | Requirement |
| :--- | :--- |
| **File Format** | CSV with UTF-8 encoding |
| **Row Count** | Exactly 5 rows corresponding to valid organs |
| **Valid Organs** | `Liver`, `Heart`, `Brain`, `Kidney`, `Lung` |
| **Header Format** | `organ,feature_000,feature_001,...,feature_127` (129 columns total) |
| **Value Domain** | Finite 32-bit floating point numbers (no NaNs, Infs, or missing values) |

---

## 2. Labeled Adverse Drug Reaction Records Schema (`data/adr_records.csv`)

| Property | Requirement |
| :--- | :--- |
| **File Format** | CSV with UTF-8 encoding |
| **Header Format** | `smiles,organ,Hepatotoxicity,Arrhythmia,Seizure,Nephrotoxicity,Pneumonitis,Nausea,Headache,Dizziness,Fatigue,Rash` |
| **SMILES Column** | Valid, RDKit-parseable SMILES string (`Chem.MolFromSmiles(smiles) is not None`) |
| **Organ Column** | Must belong to the valid tissue profiles (`Liver`, `Heart`, `Brain`, `Kidney`, `Lung`) |
| **ADR Columns** | Exactly 10 binary target labels ($\in \{0, 1\}$) |
| **Data Integrity** | No missing values, NaNs, or empty fields across any row |
