---
title: EpiADR-Net — Organ-Conditioned ADR Prediction
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.16.0
app_file: app_gradio.py
pinned: false
license: mit
---

# EpiADR-Net

**A leakage-audited benchmark for organ-conditioned adverse drug reaction prediction — and a negative result on transcriptomic conditioning.**

[![CI](https://github.com/ADjayantan/EpiADR-Net/actions/workflows/ci.yml/badge.svg)](https://github.com/ADjayantan/EpiADR-Net/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Given a drug's molecular structure and a target organ, predict whether the drug
causes that organ's characteristic adverse reaction. 1 465 molecules, 5 organs,
SIDER-derived labels, GTEx V8 transcriptomic organ profiles.

The headline finding is negative, and that is the contribution:

> **Conditioning on real GTEx transcriptomic profiles does not outperform a plain
> organ one-hot (+0.005 macro-AUROC, against a fold s.d. of 0.012–0.020), and
> replacing the real profiles with shuffled ones changes the result by −0.0001.
> True zero-shot generalisation — an unseen drug in an unseen organ — is near
> chance at 0.552.**

![tissue conditioning does not beat an organ indicator](results/figures/fig2_zeroshot.png)

`RESULTS.md` has the full tables. `AUDIT.md` explains why the figures previously
published in this repository (Macro AUROC 0.7420, and 0.9680 in the manuscript)
were withdrawn.

---

## Results at a glance

Molecule-level Bemis-Murcko scaffold 5-fold CV. 0 shared molecules and 0 shared
scaffolds between train and test, asserted on every fold.

| model | macro-AUROC | macro-AUPRC |
|:--|--:|--:|
| per-organ base rate (no molecular input) | 0.5000 ± 0.000 | 0.2401 ± 0.008 |
| Morgan + LogReg, per organ | 0.5990 ± 0.021 | 0.3465 ± 0.016 |
| **Morgan + XGBoost, per organ** | **0.6364 ± 0.016** | **0.3862 ± 0.027** |
| Morgan + organ one-hot, shared | 0.6197 ± 0.020 | 0.3522 ± 0.015 |
| Morgan + real GTEx tissue, shared | 0.6243 ± 0.012 | 0.3510 ± 0.020 |
| Morgan + **shuffled** GTEx tissue, shared | 0.6244 ± 0.013 | 0.3513 ± 0.023 |

**Per-organ AUROC is the primary metric, not pooled AUROC.** A predictor that
outputs nothing but each organ's base rate scores pooled AUROC 0.656 while ranking
no drug correctly — pooled figures on this task partly measure the Lung (7.8 %
positive) versus Brain (38.0 %) prevalence gap. See `RESULTS.md §1`.

---

## Why the negative result is a property of the labels

Organ assignment in this dataset comes from a MedDRA System Organ Class
terminology rule — hepatic terms map to Liver, arrhythmia terms to Heart, and so
on (`DATA_PROVENANCE.md`). The organ label is which ontological class a
side-effect term belongs to. It was never derived from tissue measurement, so
there is no transcriptomic signal for GTEx features to recover. With five organs,
"which organ" is five bits; a one-hot already carries all of them.

Making tissue conditioning measurable needs labels that are organ-resolved by
biology, or features that couple molecule and tissue mechanistically — drug–target
binding (ChEMBL/DrugBank) weighted by that target's per-tissue expression (GTEx).
Such a feature is not derivable from an organ identifier, so it can beat the
one-hot in principle. That is the next step, and it is not claimed here.

---

## Reproducing everything

```bash
pip install -r requirements.txt

make audit        # pipeline validity audit      -> results/audit_findings.json
make baselines    # baselines + tissue ablation  -> results/baselines.json
make zeroshot     # S1 / S2 / S3 generalisation  -> results/zeroshot.json
make results      # all three, in order
make test         # pytest
python scripts/make_figures.py
```

Every number in `README.md` and `RESULTS.md` is written by one of those scripts
into `results/*.json`. Seeds are fixed (42). Nothing is transcribed by hand.

---

## Repository layout

```
epiadr/
  task.py        organ-conditioned task construction + schema validation
  splits.py      molecule-level Bemis-Murcko scaffold CV + leakage assertions
  features.py    Morgan fingerprints; real 128-dim GTEx profile loader
  eval.py        per-organ AUROC / AUPRC / ECE, fold aggregation
scripts/
  audit_pipeline.py   reproduces the validity audit in AUDIT.md
  run_baselines.py    baseline suite + tissue-shuffle ablation
  run_zeroshot.py     leave-one-organ-out generalisation (S1/S2/S3)
  make_figures.py     renders results/figures/*.png from the JSON
data/
  adr_records.csv      1 465 molecules x 5 organs, 10 MedDRA ADR labels
  tissue_profiles.csv  5 organs x 128 GTEx V8 PCA features
model.py, train.py, utils.py, data_loader.py
                     the original graph-transformer pipeline, retained for
                     reference. Its reported metrics are withdrawn; see AUDIT.md.
app_gradio.py, app.py, api.py
                     Gradio Space, Streamlit dashboard, FastAPI service
docs/                presentation and viva material
```

---

## Status and honest limitations

- The graph-transformer model in `model.py` has **not** been re-benchmarked on the
  corrected task. The fingerprint baselines above are the current reference; any
  GNN must beat **0.6364 ± 0.016** to be worth its parameters.
- `data/adr_records.csv` is derived from SIDER 4.1 (CC BY-NC-SA 4.0) with
  PubChem SMILES resolution; `data/tissue_profiles.csv` from GTEx V8. See
  `DATA_PROVENANCE.md`.
- No external validation set. Cross-dataset evaluation (OFFSIDES / FAERS) is not
  done, so these numbers describe SIDER-derived labels only.
- **Not for clinical or preclinical decision-making.** This is a student research
  benchmark, not a validated safety model.

## License

MIT. Data under the licences recorded in `DATA_PROVENANCE.md`.
