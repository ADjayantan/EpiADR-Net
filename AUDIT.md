# EpiADR-Net — Pipeline Validity Audit

**Date:** 2026-11-10
**Scope:** the v6 training pipeline as published at commit `main`, and the metrics
quoted in `README.md`, `MODEL_CARD.md`, `SLIDES.md` and `VIVA_CHEATSHEET.md`.
**Reproduce:** `python scripts/audit_pipeline.py` → `results/audit_findings.json`

This document exists because the previously reported headline figure —
**Macro AUROC 0.7420 ± 0.031** — is not supported by the code that produced it.
It is kept in the repository rather than deleted, because the corrected pipeline
only makes sense against a record of what was wrong.

---

## Summary

| # | Finding | Severity |
|:--|:--------|:---------|
| 1 | `data/adr_records.csv` (1 465 molecules) is never read during training | blocking |
| 2 | Tissue vectors used in training are `np.random.dirichlet` noise, not GTEx | blocking |
| 3 | The training target does not depend on the organ it is conditioned on | blocking |
| 4 | The benchmark trains and evaluates on **5 molecules** | blocking |
| 5 | The `± 0.031` / `± 0.042` dispersions are not produced by any script | reporting |
| 6 | `pooled` AUROC is inflated by differing per-organ base rates | reporting |

---

## 1. The labelled dataset never enters training

`data_loader.py` defines `GTExTissueLoader.from_csv()` and
`load_labeled_records()`, both with full data-contract validation. Neither is
called anywhere in the repository:

```
modules_using_csv_loaders: []
train_py_imports_csv_loader: False
```

`train.py:192` instead constructs `EpiADRDataset(repeat=repeat_val)`, which
falls back to `BENCHMARK_DRUGS` (`data_loader.py:162`). The 791 KB of real SIDER
records in `data/adr_records.csv` are dead weight in the repository.

## 2. Training conditions on random noise, not transcriptomics

`DATA_PROVENANCE.md` documents GTEx V8 median-TPM download, organ aggregation
and `PCA(n_components=128)`. `data/tissue_profiles.csv` contains exactly that:
5 organs × 128 features.

`data_loader.py` does not use it. It builds:

```python
TISSUE_DIM = 1024
GTEX_TISSUE_PROFILES = {
    organ: torch.tensor(np.random.dirichlet(np.ones(TISSUE_DIM) * alpha)...)
    for organ, alpha in _TISSUE_ALPHA.items()
}
```

The model is conditioned on 1 024-dimensional Dirichlet samples. The documented
128-dimensional GTEx profiles are never loaded. The dimensionalities do not even
match, so the two were never interchangeable.

## 3. The target is independent of the organ

```python
def _build_target_vector(toxic_organs: list[str]) -> torch.Tensor:
```

The function takes the drug's toxicity list and **not the organ**. In
`EpiADRDataset._build()` it is called inside `for organ_name in ORGAN_NAMES:`,
so all ten organs of a molecule receive an identical 10-dimensional target.

The same holds independently in the real data. Of 1 465 molecules in
`data/adr_records.csv`, the number whose label vector differs across its five
organ rows is:

```
csv_molecules_whose_labels_vary_by_organ: 0
```

A tissue-conditioned model therefore has no organ-specific target to predict, in
either the synthetic or the real data. "Tissue-Conditioned Side Effect
Disaggregation" cannot be measured on this task as formulated.

## 4. The benchmark sees five molecules

`_RAW_DRUG_DEFS` holds Acetaminophen, Aspirin, Ibuprofen, Diclofenac and
Naproxen. With `--repeat 8` (the default) and 10 organs this yields 400 rows —
but only **5 independent label vectors**, each duplicated 80 times.

`run_5fold_ensemble_benchmark` slices those 400 rows into 5 folds of 80, so each
fold holds out **one molecule**. Within a single fold's validation set every
label column is constant, so `calculate_metrics` finds no scoreable class and
returns its `0.5` fallback; the reported figure comes from pooling all five folds'
predictions afterwards. A 116.5 M-parameter model is fitted to five compounds.

## 5. The reported dispersion has no source

`± 0.031` (AUROC) and `± 0.042` (AUPRC) appear in four documents. `train.py`
computes no standard deviation anywhere — `run_5fold_ensemble_benchmark` pools
predictions and calls `calculate_metrics` once. No script in the repository emits
those values.

## 6. Pooled AUROC is not a meaningful headline

Measured on the corrected task (see `results/baselines.json`), a predictor that
outputs **nothing but each organ's training base rate** scores:

| predictor | macro-AUROC | pooled-AUROC |
|:--|:--|:--|
| per-organ base rate, no molecular input | **0.5000** | **0.6561** |

Pooled AUROC rewards separating Lung (7.8 % positive) from Brain (38.0 %), which
requires no knowledge of any molecule. Per-organ AUROC holds the base rate
constant and is the only figure that measures drug ranking. All corrected results
report per-organ AUROC as primary.

---

## What was done about it

| Finding | Remedy |
|:--|:--|
| 1 | `epiadr/task.py` loads and validates `data/adr_records.csv`; all 1 465 molecules are used |
| 2 | `epiadr/features.py::load_tissue_profiles` reads the real 128-dim GTEx profiles |
| 3 | `epiadr/task.py` reformulates the task so the target depends on the organ (see below) |
| 4 | Scaffold CV over 1 465 molecules, `epiadr/splits.py`, with leakage assertions |
| 5 | Every reported mean ± std is emitted by a script into `results/*.json` |
| 6 | `epiadr/eval.py` reports per-organ AUROC/AUPRC/ECE as primary |

### The reformulated task

The ten ADR labels separate by the organ their names already name:

| scope | mapping |
|:--|:--|
| organ-specific | Liver→Hepatotoxicity, Heart→Arrhythmia, Brain→Seizure, Kidney→Nephrotoxicity, Lung→Pneumonitis |
| systemic | Nausea, Headache, Dizziness, Fatigue, Rash (organ-invariant) |

A row becomes `(molecule, organ) → P(that organ's ADR)`. The target now varies
across a molecule's five organ rows, which makes tissue conditioning measurable
and makes the tissue-shuffle ablation meaningful.
