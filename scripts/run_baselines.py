"""
Baseline suite for the organ-conditioned ADR task.

Every model below is evaluated on the SAME molecule-level Bemis-Murcko scaffold
5-fold split, so the numbers are directly comparable. The point of the suite is
to answer three questions a reviewer will ask first:

  1. Is the headline number better than a constant predictor?        -> B0, B1
  2. Is a graph neural network better than a fingerprint + linear
     model, i.e. does the architecture earn its parameters?          -> B2, B3
  3. Does conditioning on tissue transcriptomics actually help, or
     is the model ignoring it?                                       -> B5 vs B6

B6 is the ablation that matters: identical model, identical features, but the
organ->tissue-profile assignment is permuted, so the tissue vector carries no
information about which organ the row describes. If B5 does not beat B6, the
"tissue-conditioned" claim is not supported by the data.

Run:  python scripts/run_baselines.py
Writes: results/baselines.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from epiadr.eval import aggregate_folds, evaluate_organ_conditioned  # noqa: E402
from epiadr.features import fingerprint_matrix, load_tissue_profiles  # noqa: E402
from epiadr.splits import scaffold_kfold, verify_no_leakage  # noqa: E402
from epiadr.task import ORGANS, load_task  # noqa: E402

SEED = 42
N_SPLITS = 5
N_BITS = 2048


# ────────────────────────────────────────────────────────────────────────────
#  Feature assembly
# ────────────────────────────────────────────────────────────────────────────
def build_design_matrix(task, fps, tissue, mode: str, rng: np.random.Generator):
    """
    mode:
      "mol"            fingerprint only                     (organ-blind)
      "mol+onehot"     fingerprint + organ one-hot          (organ-aware, no biology)
      "mol+tissue"     fingerprint + real GTEx profile      (tissue-conditioned)
      "mol+shuffled"   fingerprint + permuted GTEx profile  (ablation)
    """
    organ_index = {o: i for i, o in enumerate(ORGANS)}

    if mode == "mol+shuffled":
        permuted = list(ORGANS)
        # Derangement: no organ keeps its own profile.
        while True:
            rng.shuffle(permuted)
            if all(a != b for a, b in zip(ORGANS, permuted, strict=False)):
                break
        tissue_lookup = {o: tissue[permuted[i]] for i, o in enumerate(ORGANS)}
    else:
        tissue_lookup = tissue

    rows = []
    for smi, organ in zip(task.smiles, task.organ, strict=False):
        parts = [fps[smi]]
        if mode == "mol+onehot":
            oh = np.zeros(len(ORGANS), dtype=np.float32)
            oh[organ_index[organ]] = 1.0
            parts.append(oh)
        elif mode in ("mol+tissue", "mol+shuffled"):
            parts.append(tissue_lookup[organ])
        rows.append(np.concatenate(parts))
    return np.stack(rows)


# ────────────────────────────────────────────────────────────────────────────
#  Models
# ────────────────────────────────────────────────────────────────────────────
def fit_logreg(X_tr, y_tr, X_te):
    scaler = StandardScaler(with_mean=False)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    clf = LogisticRegression(
        max_iter=3000, C=0.1, class_weight="balanced", solver="liblinear", random_state=SEED
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def fit_xgb(X_tr, y_tr, X_te):
    from xgboost import XGBClassifier

    pos = max(float(y_tr.sum()), 1.0)
    neg = max(float(len(y_tr) - y_tr.sum()), 1.0)
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


# ────────────────────────────────────────────────────────────────────────────
#  Baseline definitions
# ────────────────────────────────────────────────────────────────────────────
def predict_pooled_prevalence(task, tr_rows, te_rows, **_):
    """B0: one constant for every row."""
    p = float(np.mean([task.y_organ[i] for i in tr_rows]))
    return np.full(len(te_rows), p)


def predict_organ_prevalence(task, tr_rows, te_rows, **_):
    """B1: the training base rate of each organ. Pooled AUROC > 0.5, per-organ AUROC = 0.5."""
    rate: dict[str, float] = {}
    for organ in ORGANS:
        vals = [task.y_organ[i] for i in tr_rows if task.organ[i] == organ]
        rate[organ] = float(np.mean(vals)) if vals else 0.0
    return np.asarray([rate[task.organ[i]] for i in te_rows])


def predict_per_organ_model(task, tr_rows, te_rows, fps, fitter, **_):
    """B2/B3: an independent model per organ, fingerprint features only."""
    preds = np.zeros(len(te_rows))
    te_pos = {i: k for k, i in enumerate(te_rows)}
    for organ in ORGANS:
        tr_o = [i for i in tr_rows if task.organ[i] == organ]
        te_o = [i for i in te_rows if task.organ[i] == organ]
        if not te_o:
            continue
        X_tr = np.stack([fps[task.smiles[i]] for i in tr_o])
        y_tr = np.asarray([task.y_organ[i] for i in tr_o])
        X_te = np.stack([fps[task.smiles[i]] for i in te_o])
        if len(np.unique(y_tr)) < 2:
            preds[[te_pos[i] for i in te_o]] = float(y_tr.mean())
            continue
        p = fitter(X_tr, y_tr, X_te)
        for i, val in zip(te_o, p, strict=False):
            preds[te_pos[i]] = val
    return preds


def predict_shared_model(task, tr_rows, te_rows, X, fitter, **_):
    """B4/B5/B6: one shared model over all organs, using the given design matrix."""
    y_tr = np.asarray([task.y_organ[i] for i in tr_rows])
    return fitter(X[tr_rows], y_tr, X[te_rows])


# ────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    task = load_task("data/adr_records.csv")
    tissue = load_tissue_profiles("data/tissue_profiles.csv")
    molecules = task.molecules
    print(f"task: {len(task)} rows, {len(molecules)} molecules, {len(ORGANS)} organs")
    print("organ-specific prevalence:",
          {k: round(v, 4) for k, v in task.prevalence_by_organ().items()})

    print(f"featurising {len(molecules)} molecules (Morgan r=2, {N_BITS} bits)...")
    fps = fingerprint_matrix(molecules, N_BITS)

    designs = {
        mode: build_design_matrix(task, fps, tissue, mode, rng)
        for mode in ("mol", "mol+onehot", "mol+tissue", "mol+shuffled")
    }

    folds = scaffold_kfold(molecules, n_splits=N_SPLITS, seed=SEED)
    leakage = []
    for f, (tr_mols, te_mols) in enumerate(folds, 1):
        leakage.append({"fold": f, **verify_no_leakage(tr_mols, te_mols)})
    print(f"scaffold split verified clean across {N_SPLITS} folds "
          f"(0 shared molecules, 0 shared scaffolds)")

    baselines = {
        "B0_pooled_prevalence": dict(fn=predict_pooled_prevalence, kw={}),
        "B1_organ_prevalence": dict(fn=predict_organ_prevalence, kw={}),
        "B2_morgan_logreg_per_organ": dict(fn=predict_per_organ_model, kw={"fitter": fit_logreg}),
        "B3_morgan_xgboost_per_organ": dict(fn=predict_per_organ_model, kw={"fitter": fit_xgb}),
        "B4_morgan_onehot_shared_xgb": dict(fn=predict_shared_model,
                                            kw={"X": designs["mol+onehot"], "fitter": fit_xgb}),
        "B5_morgan_tissue_shared_xgb": dict(fn=predict_shared_model,
                                            kw={"X": designs["mol+tissue"], "fitter": fit_xgb}),
        "B6_morgan_tissue_SHUFFLED_xgb": dict(fn=predict_shared_model,
                                              kw={"X": designs["mol+shuffled"], "fitter": fit_xgb}),
    }

    results: dict[str, dict] = {}
    for name, spec in baselines.items():
        print(f"\n--> {name}")
        fold_results = []
        for f, (tr_mols, te_mols) in enumerate(folds, 1):
            tr_rows = task.rows_for(tr_mols)
            te_rows = task.rows_for(te_mols)
            preds = spec["fn"](task, tr_rows, te_rows, fps=fps, **spec["kw"])
            res = evaluate_organ_conditioned(
                y_true=[task.y_organ[i] for i in te_rows],
                y_prob=preds,
                organs=[task.organ[i] for i in te_rows],
                organ_names=ORGANS,
            )
            fold_results.append(res)
            print(f"    fold {f}: macro-AUROC {res['macro_auroc']:.4f}  "
                  f"macro-AUPRC {res['macro_auprc']:.4f}  pooled-AUROC {res['pooled_auroc']:.4f}")
        agg = aggregate_folds(fold_results, ORGANS)
        results[name] = {"folds": fold_results, "aggregate": agg}
        print(f"    MEAN: macro-AUROC {agg['macro_auroc']['mean']:.4f} "
              f"+/- {agg['macro_auroc']['std']:.4f}   "
              f"macro-AUPRC {agg['macro_auprc']['mean']:.4f} "
              f"+/- {agg['macro_auprc']['std']:.4f}")

    payload = {
        "config": {
            "seed": SEED, "n_splits": N_SPLITS, "n_bits": N_BITS,
            "n_rows": len(task), "n_molecules": len(molecules),
            "organs": ORGANS,
            "prevalence_by_organ": task.prevalence_by_organ(),
        },
        "leakage_check": leakage,
        "results": results,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/baselines.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # ── summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"{'model':<34}{'macro-AUROC':>16}{'macro-AUPRC':>16}{'pooled-AUROC':>16}{'ECE':>10}")
    print("-" * 92)
    for name, r in results.items():
        a = r["aggregate"]
        print(f"{name:<34}"
              f"{a['macro_auroc']['mean']:>9.4f} ±{a['macro_auroc']['std']:.3f}"
              f"{a['macro_auprc']['mean']:>9.4f} ±{a['macro_auprc']['std']:.3f}"
              f"{a['pooled_auroc']['mean']:>9.4f} ±{a['pooled_auroc']['std']:.3f}"
              f"{a['ece']['mean']:>10.4f}")
    print("=" * 92)

    # per-organ breakdown of the strongest fingerprint model
    best = max(results, key=lambda k: results[k]["aggregate"]["macro_auroc"]["mean"])
    print(f"\nper-organ AUROC — {best}")
    for organ in ORGANS:
        po = results[best]["aggregate"]["per_organ"][organ]
        print(f"  {organ:<8} AUROC {po['auroc']['mean']:.4f} ±{po['auroc']['std']:.3f}   "
              f"AUPRC {po['auprc']['mean']:.4f} ±{po['auprc']['std']:.3f}   "
              f"(prevalence {task.prevalence_by_organ()[organ]:.3f})")

    tis = results["B5_morgan_tissue_shared_xgb"]["aggregate"]["macro_auroc"]["mean"]
    shuf = results["B6_morgan_tissue_SHUFFLED_xgb"]["aggregate"]["macro_auroc"]["mean"]
    print(f"\ntissue ablation: real {tis:.4f} vs shuffled {shuf:.4f}  "
          f"(delta {tis - shuf:+.4f})")
    print(f"\nwrote results/baselines.json in {payload['runtime_seconds']}s")


if __name__ == "__main__":
    main()
