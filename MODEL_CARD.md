# Model Card — EpiADR-Net

## Overview

| | |
|:--|:--|
| **Task** | `(molecule, organ) → P(that organ's adverse reaction)` |
| **Current reference model** | Morgan fingerprint (ECFP4, 2048 bit) + XGBoost, one model per organ |
| **Inputs** | SMILES; organ identifier, or the organ's 128-dim GTEx V8 PCA profile |
| **Output** | One calibrated-ish probability per (molecule, organ) pair |
| **Evaluation** | Molecule-level Bemis-Murcko scaffold 5-fold CV, seed 42, leakage asserted |
| **Produced by** | `scripts/run_baselines.py` → `results/baselines.json` |

A graph transformer (`model.py`, ~116 M parameters) also exists in this
repository. It has **not** been re-benchmarked on the corrected task and no
metrics are claimed for it. The fingerprint model is the current reference
because it is the strongest result that has actually been measured.

## Performance

Per-organ AUROC is primary. Pooled AUROC is reported for completeness only — a
base-rate predictor with no molecular input reaches pooled 0.656 (see `AUDIT.md §6`).

| model | macro-AUROC | macro-AUPRC | pooled-AUROC |
|:--|--:|--:|--:|
| per-organ base rate | 0.5000 ± 0.000 | 0.2401 ± 0.008 | 0.6561 |
| Morgan + LogReg, per organ | 0.5990 ± 0.021 | 0.3465 ± 0.016 | 0.6358 |
| **Morgan + XGBoost, per organ** | **0.6364 ± 0.016** | **0.3862 ± 0.027** | 0.6904 |
| Morgan + real GTEx tissue, shared | 0.6243 ± 0.012 | 0.3510 ± 0.020 | 0.6949 |
| Morgan + shuffled tissue, shared | 0.6244 ± 0.013 | 0.3513 ± 0.023 | 0.6954 |

| organ | AUROC | AUPRC | positives |
|:--|--:|--:|--:|
| Liver (hepatotoxicity) | 0.6667 ± 0.015 | 0.3430 ± 0.042 | 18.5 % |
| Heart (arrhythmia) | 0.5990 ± 0.009 | 0.3983 ± 0.050 | 29.2 % |
| Brain (seizure) | 0.6317 ± 0.027 | 0.5315 ± 0.052 | 38.0 % |
| Kidney (nephrotoxicity) | 0.6226 ± 0.041 | 0.4012 ± 0.063 | 26.5 % |
| Lung (pneumonitis) | 0.6621 ± 0.081 | 0.2572 ± 0.077 | 7.8 % |

### Generalisation

| setting | macro-AUROC (best) |
|:--|--:|
| S1 unseen drug, seen organ | 0.6243 |
| S2 seen drug, unseen organ | 0.6975 (molecule-only model is best) |
| S3 **unseen drug + unseen organ** | **0.5517 — near chance** |

## Known negative result

Conditioning on real GTEx transcriptomic profiles does not beat a plain organ
one-hot (+0.005 macro-AUROC, fold s.d. 0.012–0.020). Deranging the profiles so
every organ receives another organ's transcriptome changes macro-AUROC by
−0.0001. The organ labels were assigned by MedDRA System Organ Class terminology
rather than by tissue measurement, so there is no transcriptomic signal to
recover. See `RESULTS.md §3`.

## Training data

SIDER 4.1 side-effect records (CC BY-NC-SA 4.0), SMILES resolved via PubChem PUG
REST; GTEx V8 median TPM reduced to 128 PCA components per organ. Full sources,
URLs, download dates and mapping rules in `DATA_PROVENANCE.md`; schema in
`DATA_CONTRACT.md`.

1 465 molecules, 7 325 (molecule, organ) rows, 5 organs.

## Intended use

Research and teaching: benchmarking organ-conditioned ADR models, and studying
how label provenance limits what conditioning features can contribute.

## Out of scope

**Not for clinical, prescribing, preclinical or regulatory decision-making.**
Macro-AUROC 0.64 is far below any threshold that would make this safe to act on,
S3 generalisation is near chance, and the labels are a literature-mined
terminology mapping rather than clinical outcomes. Predictions carry no
indication, dose, route, duration or patient context.

## Limitations

- Evaluated only on SIDER-derived labels; no external validation (OFFSIDES, FAERS).
- Five organs and one ADR per organ; systemic ADRs are not modelled by the
  reference model.
- ECE 0.111 for the reference model — the probabilities are not well calibrated
  and should not be read as risk estimates.
- Class imbalance is severe for Lung (7.8 % positive); its AUROC has the widest
  fold spread (± 0.081).

## History

Figures previously published for this model — Macro AUROC 0.7420 ± 0.031 (README,
model card, slides) and 0.9680 (manuscript) — were withdrawn after an audit found
that the training pipeline did not read the labelled dataset, conditioned on
random Dirichlet vectors instead of GTEx profiles, used a target that did not
depend on the organ, and trained on five molecules. See `AUDIT.md`.
