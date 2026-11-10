"""
Permutation test for the mechanism feature.

run_mechanism.py compares the real organ->expression assignment against ONE
derangement. One derangement is a single draw from the null, so a gain of +0.008
against it could easily be luck. This script builds the null properly: it refits
the best model under N independent derangements and reports where the real
assignment falls in that distribution, as an empirical p-value.

Null hypothesis: the mechanism features carry no information about WHICH organ a
row describes beyond what the rest of the model already has, so permuting the
organ->expression map should not matter.

Run:  python scripts/run_ablation_test.py [--permutations 10]
Writes: results/ablation_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from epiadr.eval import aggregate_folds, evaluate_organ_conditioned  # noqa: E402
from epiadr.features import fingerprint_matrix  # noqa: E402
from epiadr.mechanism import (  # noqa: E402
    derange,
    load_drug_targets,
    load_gene_expression,
    mechanism_features,
)
from epiadr.splits import scaffold_kfold, verify_no_leakage  # noqa: E402
from epiadr.task import ORGANS, load_task  # noqa: E402

SEED = 42
N_SPLITS = 5
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)

    task = load_task("data/adr_records.csv")
    molecules = task.molecules
    organ_idx = {o: i for i, o in enumerate(ORGANS)}
    fps = fingerprint_matrix(molecules, N_BITS)
    binding, genes = load_drug_targets()
    frac, logn = load_gene_expression(genes)
    zero = np.zeros(len(genes), dtype=np.float32)

    folds = scaffold_kfold(molecules, n_splits=N_SPLITS, seed=SEED)
    for tr, te in folds:
        verify_no_leakage(tr, te)
    y = np.asarray(task.y_organ, dtype=float)

    def build(frac_m: np.ndarray, logn_m: np.ndarray) -> np.ndarray:
        """M8: fingerprint + raw binding + organ one-hot + mechanism coupling."""
        rows = []
        for smi, organ in zip(task.smiles, task.organ, strict=True):
            oi = organ_idx[organ]
            b = binding.get(smi, zero)
            oh = np.zeros(len(ORGANS), dtype=np.float32)
            oh[oi] = 1.0
            rows.append(np.concatenate(
                [fps[smi], b, oh, mechanism_features(b, oi, frac_m, logn_m)]
            ))
        return np.stack(rows)

    def score(X: np.ndarray) -> float:
        fold_results = []
        for tr_mols, te_mols in folds:
            tr = task.rows_for(tr_mols)
            te = task.rows_for(te_mols)
            p = fit_xgb(X[tr], y[tr], X[te])
            fold_results.append(evaluate_organ_conditioned(
                y[te], p, [task.organ[i] for i in te], ORGANS))
        return aggregate_folds(fold_results, ORGANS)["macro_auroc"]["mean"]

    print(f"M8 permutation test — {args.permutations} derangements of the "
          f"organ->expression map\n")
    real = score(build(frac, logn))
    print(f"  real organ->expression assignment : macro-AUROC {real:.4f}\n")

    null, perms = [], []
    for k in range(1, args.permutations + 1):
        perm = derange(rng)
        mapping = {ORGANS[i]: ORGANS[p] for i, p in enumerate(perm)}
        s = score(build(frac[:, perm], logn[:, perm]))
        null.append(s)
        perms.append({"mapping": mapping, "macro_auroc": s})
        print(f"  [{k:>2}/{args.permutations}] {s:.4f}   "
              + " ".join(f"{a[:2]}->{b[:2]}" for a, b in mapping.items()), flush=True)

    null_arr = np.asarray(null)
    beaten = int((null_arr >= real).sum())
    p_value = (beaten + 1) / (len(null_arr) + 1)      # conservative, +1 correction

    payload = {
        "config": {"seed": SEED, "n_splits": N_SPLITS, "n_bits": N_BITS,
                   "model": "M8_morgan_binding_onehot_MECHANISM",
                   "n_permutations": args.permutations},
        "real_macro_auroc": real,
        "null_macro_auroc": null,
        "permutations": perms,
        "null_mean": float(null_arr.mean()),
        "null_std": float(null_arr.std(ddof=1)) if len(null_arr) > 1 else 0.0,
        "effect": real - float(null_arr.mean()),
        "n_null_ge_real": beaten,
        "p_value": p_value,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_test.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 72)
    print(f"  real                 {real:.4f}")
    print(f"  null (deranged)      {null_arr.mean():.4f} ± {payload['null_std']:.4f}"
          f"   [min {null_arr.min():.4f}, max {null_arr.max():.4f}]")
    print(f"  effect               {payload['effect']:+.4f}")
    print(f"  permutations >= real {beaten}/{len(null_arr)}")
    print(f"  empirical p          {p_value:.3f}")
    print("=" * 72)
    verdict = ("the correct organ->expression map matters"
               if p_value <= 0.05 else
               "NOT significant — the gain is consistent with chance")
    print(f"  verdict: {verdict}")
    print(f"\nwrote results/ablation_test.json in {payload['runtime_seconds']}s")


if __name__ == "__main__":
    main()
