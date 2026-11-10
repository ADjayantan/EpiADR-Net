"""
Leave-one-organ-out zero-shot evaluation.

This is the experiment the project's "zero-shot" claim requires and did not have.

Three generalisation settings, increasingly strict:

  S1  unseen drug,  seen organ    scaffold CV; every organ appears in training
  S2  seen drug,    UNSEEN organ  train on 4 organs, test on the 5th
  S3  UNSEEN drug + UNSEEN organ  held-out organ AND held-out scaffolds

S2/S3 are the settings that separate a tissue-conditioned model from an
organ-indexed one. An organ one-hot has no column for an organ it never saw, so
at test time its organ feature is all-zeros and it can only emit a global
average -- it cannot transfer by construction. A tissue profile is a point in a
128-dimensional transcriptomic space shared by all organs, so a model that has
learned "molecules with feature X are toxic in tissues whose profile looks like
P" can place an unseen organ in that space.

If the tissue model beats the one-hot and molecule-only models in S2/S3, tissue
conditioning is doing real work. If it does not, the honest conclusion is that
SIDER-derived organ labels carry no transcriptomic signal beyond organ identity.

Run:  python scripts/run_zeroshot.py
Writes: results/zeroshot.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from epiadr.eval import expected_calibration_error  # noqa: E402
from epiadr.features import fingerprint_matrix, load_tissue_profiles  # noqa: E402
from epiadr.splits import scaffold_kfold  # noqa: E402
from epiadr.task import ORGANS, load_task  # noqa: E402

SEED = 42
N_BITS = 2048


def fit_xgb(X_tr, y_tr, X_te):
    from xgboost import XGBClassifier

    pos = max(float(y_tr.sum()), 1.0)
    neg = max(float(len(y_tr) - y_tr.sum()), 1.0)
    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.5, reg_lambda=1.0, scale_pos_weight=neg / pos,
        eval_metric="logloss", tree_method="hist", n_jobs=4, random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def featurise(task, fps, tissue, mode: str) -> np.ndarray:
    """
    mode:
      "mol"        fingerprint only              -> organ-blind, identical prediction per organ
      "onehot"     fingerprint + organ one-hot   -> cannot represent an unseen organ
      "tissue"     fingerprint + GTEx profile    -> can place an unseen organ in profile space
    """
    oi = {o: i for i, o in enumerate(ORGANS)}
    rows = []
    for smi, organ in zip(task.smiles, task.organ, strict=False):
        parts = [fps[smi]]
        if mode == "onehot":
            oh = np.zeros(len(ORGANS), dtype=np.float32)
            oh[oi[organ]] = 1.0
            parts.append(oh)
        elif mode == "tissue":
            parts.append(tissue[organ])
        rows.append(np.concatenate(parts))
    return np.stack(rows)


def score(y_true, y_prob) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    single_class = len(np.unique(y_true)) < 2
    return {
        "n": int(y_true.size),
        "prevalence": float(y_true.mean()),
        "auroc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "auprc": float("nan") if single_class else float(average_precision_score(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "pred_std": float(y_prob.std()),
    }


def ms(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}


def main():
    t0 = time.time()
    task = load_task("data/adr_records.csv")
    tissue = load_tissue_profiles("data/tissue_profiles.csv")
    molecules = task.molecules
    fps = fingerprint_matrix(molecules, N_BITS)

    X = {m: featurise(task, fps, tissue, m) for m in ("mol", "onehot", "tissue")}
    y = np.asarray(task.y_organ, dtype=float)
    organ_of_row = np.asarray(task.organ)

    folds = scaffold_kfold(molecules, n_splits=5, seed=SEED)
    out: dict = {"config": {"seed": SEED, "n_bits": N_BITS,
                            "n_molecules": len(molecules), "n_rows": len(task),
                            "organs": ORGANS,
                            "prevalence_by_organ": task.prevalence_by_organ()},
                 "settings": {}}

    # ── S1: unseen drug, seen organ ────────────────────────────────────────
    print("=" * 84)
    print("S1  unseen drug, seen organ   (scaffold 5-fold CV)")
    print("=" * 84)
    s1: dict = {}
    for mode in ("mol", "onehot", "tissue"):
        per_fold_macro = []
        for tr_mols, te_mols in folds:
            tr = task.rows_for(tr_mols)
            te = task.rows_for(te_mols)
            p = fit_xgb(X[mode][tr], y[tr], X[mode][te])
            per_organ = [score(y[[i for i in te if organ_of_row[i] == o]],
                               p[[k for k, i in enumerate(te) if organ_of_row[i] == o]])["auroc"]
                         for o in ORGANS]
            per_fold_macro.append(float(np.nanmean(per_organ)))
        s1[mode] = {"macro_auroc": ms(per_fold_macro), "folds": per_fold_macro}
        print(f"  {mode:<8} macro-AUROC {s1[mode]['macro_auroc']['mean']:.4f} "
              f"±{s1[mode]['macro_auroc']['std']:.4f}")
    out["settings"]["S1_unseen_drug_seen_organ"] = s1

    # ── S2: seen drug, unseen organ ───────────────────────────────────────
    print("\n" + "=" * 84)
    print("S2  seen drug, UNSEEN organ   (train on 4 organs, test on the 5th)")
    print("=" * 84)
    print(f"  {'held-out organ':<16}{'mol':>12}{'onehot':>12}{'tissue':>12}{'prevalence':>13}")
    print("  " + "-" * 65)
    s2: dict = {"per_organ": {}}
    for held in ORGANS:
        tr = [i for i in range(len(task)) if organ_of_row[i] != held]
        te = [i for i in range(len(task)) if organ_of_row[i] == held]
        row = {}
        for mode in ("mol", "onehot", "tissue"):
            p = fit_xgb(X[mode][tr], y[tr], X[mode][te])
            row[mode] = score(y[te], p)
        s2["per_organ"][held] = row
        print(f"  {held:<16}{row['mol']['auroc']:>12.4f}{row['onehot']['auroc']:>12.4f}"
              f"{row['tissue']['auroc']:>12.4f}{row['tissue']['prevalence']:>13.3f}")
    for mode in ("mol", "onehot", "tissue"):
        s2[mode] = {"macro_auroc": ms([s2["per_organ"][o][mode]["auroc"] for o in ORGANS]),
                    "macro_auprc": ms([s2["per_organ"][o][mode]["auprc"] for o in ORGANS])}
    print("  " + "-" * 65)
    print(f"  {'MACRO':<16}{s2['mol']['macro_auroc']['mean']:>12.4f}"
          f"{s2['onehot']['macro_auroc']['mean']:>12.4f}"
          f"{s2['tissue']['macro_auroc']['mean']:>12.4f}")
    out["settings"]["S2_seen_drug_unseen_organ"] = s2

    # ── S3: unseen drug AND unseen organ ──────────────────────────────────
    print("\n" + "=" * 84)
    print("S3  UNSEEN drug + UNSEEN organ   (held-out organ x held-out scaffolds)")
    print("=" * 84)
    print(f"  {'held-out organ':<16}{'mol':>12}{'onehot':>12}{'tissue':>12}")
    print("  " + "-" * 52)
    s3: dict = {"per_organ": {}}
    for held in ORGANS:
        fold_scores = {m: [] for m in ("mol", "onehot", "tissue")}
        for tr_mols, te_mols in folds:
            tr = [i for i in range(len(task))
                  if organ_of_row[i] != held and task.smiles[i] in tr_mols]
            te = [i for i in range(len(task))
                  if organ_of_row[i] == held and task.smiles[i] in te_mols]
            if not te:
                continue
            for mode in ("mol", "onehot", "tissue"):
                p = fit_xgb(X[mode][tr], y[tr], X[mode][te])
                fold_scores[mode].append(score(y[te], p)["auroc"])
        s3["per_organ"][held] = {m: ms(v) for m, v in fold_scores.items()}
        print(f"  {held:<16}"
              f"{s3['per_organ'][held]['mol']['mean']:>12.4f}"
              f"{s3['per_organ'][held]['onehot']['mean']:>12.4f}"
              f"{s3['per_organ'][held]['tissue']['mean']:>12.4f}")
    for mode in ("mol", "onehot", "tissue"):
        s3[mode] = {"macro_auroc": ms([s3["per_organ"][o][mode]["mean"] for o in ORGANS])}
    print("  " + "-" * 52)
    print(f"  {'MACRO':<16}{s3['mol']['macro_auroc']['mean']:>12.4f}"
          f"{s3['onehot']['macro_auroc']['mean']:>12.4f}"
          f"{s3['tissue']['macro_auroc']['mean']:>12.4f}")
    out["settings"]["S3_unseen_drug_unseen_organ"] = s3

    out["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs("results", exist_ok=True)
    with open("results/zeroshot.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # ── verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    for key, label in (("S1_unseen_drug_seen_organ", "S1 unseen drug, seen organ"),
                       ("S2_seen_drug_unseen_organ", "S2 seen drug, unseen organ"),
                       ("S3_unseen_drug_unseen_organ", "S3 unseen drug + unseen organ")):
        d = out["settings"][key]
        t = d["tissue"]["macro_auroc"]["mean"]
        o = d["onehot"]["macro_auroc"]["mean"]
        m = d["mol"]["macro_auroc"]["mean"]
        print(f"{label:<34} tissue {t:.4f}   onehot {o:.4f}   mol {m:.4f}   "
              f"tissue-onehot {t - o:+.4f}")
    print(f"\nwrote results/zeroshot.json in {out['runtime_seconds']}s")


if __name__ == "__main__":
    main()
