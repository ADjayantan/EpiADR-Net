"""
Does mechanistic coupling beat an organ indicator?

RESULTS.md established that a GTEx profile adds nothing over a 5-dim organ
one-hot, because any feature that is a function of the organ alone is capped by
what the one-hot already encodes. This script tests the one class of feature
that is not such a function: drug-target binding multiplied by that target's
expression in the organ.

Models, all on the identical molecule-level scaffold 5-fold CV:

  M0  Morgan only                       organ-blind control
  M1  Morgan + organ one-hot            the bar to beat
  M2  Morgan + GTEx PCA profile         the feature that already failed
  M3  Morgan + binding vector           targets, but no organ coupling
  M4  Morgan + binding + one-hot        targets and organ, still not coupled
  M5  Morgan + MECHANISM coupling       the hypothesis
  M6  M5 with organ->expression DERANGED    the ablation
  M7  mechanism coupling alone          no fingerprint at all

M5 > M4 is the result that would support tissue conditioning: M4 has the same
target information and the same organ information, but not their product.
M5 > M6 shows the gain needs the CORRECT organ-expression assignment.

A caveat the script then tests directly: a gradient-boosted tree can already
form "organ == Liver AND binds ABCB11" from M4's separate columns, so M5's
advantage within a seen organ may be small. Where the two genuinely diverge is
an UNSEEN organ. A one-hot has no column for an organ it never saw and cannot
transfer at all; an expression vector places the new organ in the same space the
model already learned over. The leave-one-organ-out section at the end is
therefore the decisive comparison, not the in-distribution table.

Run:  python scripts/run_mechanism.py
Writes: results/mechanism.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from epiadr.eval import aggregate_folds, evaluate_organ_conditioned  # noqa: E402
from epiadr.features import fingerprint_matrix, load_tissue_profiles  # noqa: E402
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
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    task = load_task("data/adr_records.csv")
    molecules = task.molecules
    organ_idx = {o: i for i, o in enumerate(ORGANS)}

    fps = fingerprint_matrix(molecules, N_BITS)
    tissue_pca = load_tissue_profiles("data/tissue_profiles.csv")
    binding, genes = load_drug_targets()
    frac, logn = load_gene_expression(genes)

    n_genes = len(genes)
    zero = np.zeros(n_genes, dtype=np.float32)
    covered = sum(1 for m in molecules if binding.get(m) is not None and binding[m].any())
    print(f"molecules {len(molecules)} | panel genes {n_genes} | "
          f"{covered} molecules ({100 * covered / len(molecules):.1f}%) hit >=1 panel gene")

    # Deranged organ -> expression map for the ablation.
    perm = derange(rng)
    print("derangement for ablation: " +
          ", ".join(f"{ORGANS[i]}->{ORGANS[p]}" for i, p in enumerate(perm)))
    frac_d, logn_d = frac[:, perm], logn[:, perm]

    # ── assemble every design matrix once ────────────────────────────────
    def build(mode: str) -> np.ndarray:
        rows = []
        for smi, organ in zip(task.smiles, task.organ, strict=True):
            oi = organ_idx[organ]
            b = binding.get(smi, zero)
            parts: list[np.ndarray] = []
            if mode not in ("mech_only", "binding_onehot_nofp"):
                parts.append(fps[smi])
            if mode == "onehot":
                oh = np.zeros(len(ORGANS), dtype=np.float32)
                oh[oi] = 1.0
                parts.append(oh)
            elif mode == "pca_tissue":
                parts.append(tissue_pca[organ])
            elif mode == "binding":
                parts.append(b)
            elif mode == "binding_onehot":
                oh = np.zeros(len(ORGANS), dtype=np.float32)
                oh[oi] = 1.0
                parts.extend([b, oh])
            elif mode in ("mechanism", "mech_only"):
                parts.append(mechanism_features(b, oi, frac, logn))
            elif mode == "full":
                oh = np.zeros(len(ORGANS), dtype=np.float32)
                oh[oi] = 1.0
                parts.extend([b, oh, mechanism_features(b, oi, frac, logn)])
            elif mode == "binding_onehot_nofp":
                oh = np.zeros(len(ORGANS), dtype=np.float32)
                oh[oi] = 1.0
                parts.extend([b, oh])
            elif mode == "mechanism_deranged":
                parts.append(mechanism_features(b, oi, frac_d, logn_d))
            rows.append(np.concatenate(parts))
        return np.stack(rows)

    models = {
        "M0_morgan_only": "mol",
        "M1_morgan_onehot": "onehot",
        "M2_morgan_gtex_pca": "pca_tissue",
        "M3_morgan_binding": "binding",
        "M4_morgan_binding_onehot": "binding_onehot",
        "M5_morgan_MECHANISM": "mechanism",
        "M6_morgan_mechanism_DERANGED": "mechanism_deranged",
        "M7_mechanism_only": "mech_only",
        "M8_morgan_binding_onehot_MECHANISM": "full",
        "M9_binding_onehot_no_fingerprint": "binding_onehot_nofp",
    }
    designs = {name: build(mode) for name, mode in models.items()}
    for name, X in designs.items():
        print(f"  {name:<30} {X.shape[1]:>5} features")

    folds = scaffold_kfold(molecules, n_splits=N_SPLITS, seed=SEED)
    for tr, te in folds:
        verify_no_leakage(tr, te)
    print(f"scaffold split verified clean across {N_SPLITS} folds\n")

    y = np.asarray(task.y_organ, dtype=float)
    results: dict[str, dict] = {}
    for name, X in designs.items():
        fold_results = []
        for tr_mols, te_mols in folds:
            tr = task.rows_for(tr_mols)
            te = task.rows_for(te_mols)
            p = fit_xgb(X[tr], y[tr], X[te])
            fold_results.append(evaluate_organ_conditioned(
                y_true=y[te], y_prob=p,
                organs=[task.organ[i] for i in te], organ_names=ORGANS,
            ))
        agg = aggregate_folds(fold_results, ORGANS)
        results[name] = {"folds": fold_results, "aggregate": agg,
                         "n_features": int(X.shape[1])}
        print(f"  {name:<30} macro-AUROC {agg['macro_auroc']['mean']:.4f} "
              f"±{agg['macro_auroc']['std']:.4f}   "
              f"macro-AUPRC {agg['macro_auprc']['mean']:.4f} "
              f"±{agg['macro_auprc']['std']:.4f}", flush=True)

    # ── leave-one-organ-out: the decisive comparison ─────────────────────
    print("\n" + "=" * 86)
    print("LEAVE-ONE-ORGAN-OUT — can the feature transfer to an organ never seen in training?")
    print("=" * 86)
    organ_of_row = np.asarray(task.organ)
    loo_models = {
        "M1_morgan_onehot": designs["M1_morgan_onehot"],
        "M4_morgan_binding_onehot": designs["M4_morgan_binding_onehot"],
        "M5_morgan_MECHANISM": designs["M5_morgan_MECHANISM"],
        "M6_morgan_mechanism_DERANGED": designs["M6_morgan_mechanism_DERANGED"],
        "M8_morgan_binding_onehot_MECHANISM": designs["M8_morgan_binding_onehot_MECHANISM"],
    }

    # S2: molecule seen, organ unseen
    s2: dict[str, dict] = {"per_organ": {}}
    print("\nS2  seen drug, UNSEEN organ")
    print(f"  {'held-out organ':<16}" + "".join(f"{k.split('_')[0]:>12}" for k in loo_models))
    print("  " + "-" * (16 + 12 * len(loo_models)))
    for held in ORGANS:
        tr = [i for i in range(len(task)) if organ_of_row[i] != held]
        te = [i for i in range(len(task)) if organ_of_row[i] == held]
        row = {}
        for name, X in loo_models.items():
            p = fit_xgb(X[tr], y[tr], X[te])
            r = evaluate_organ_conditioned(y[te], p, [task.organ[i] for i in te], [held])
            row[name] = r["per_organ"][held]["auroc"]
        s2["per_organ"][held] = row
        print(f"  {held:<16}" + "".join(f"{row[k]:>12.4f}" for k in loo_models), flush=True)
    for name in loo_models:
        vals = [s2["per_organ"][o][name] for o in ORGANS]
        s2[name] = {"macro_auroc": float(np.nanmean(vals))}
    print("  " + "-" * (16 + 12 * len(loo_models)))
    print(f"  {'MACRO':<16}" + "".join(f"{s2[k]['macro_auroc']:>12.4f}" for k in loo_models))

    # S3: molecule unseen AND organ unseen
    s3: dict[str, dict] = {"per_organ": {}}
    print("\nS3  UNSEEN drug + UNSEEN organ")
    print(f"  {'held-out organ':<16}" + "".join(f"{k.split('_')[0]:>12}" for k in loo_models))
    print("  " + "-" * (16 + 12 * len(loo_models)))
    for held in ORGANS:
        acc = {name: [] for name in loo_models}
        for tr_mols, te_mols in folds:
            tr = [i for i in range(len(task))
                  if organ_of_row[i] != held and task.smiles[i] in tr_mols]
            te = [i for i in range(len(task))
                  if organ_of_row[i] == held and task.smiles[i] in te_mols]
            if not te:
                continue
            for name, X in loo_models.items():
                p = fit_xgb(X[tr], y[tr], X[te])
                r = evaluate_organ_conditioned(y[te], p, [task.organ[i] for i in te], [held])
                acc[name].append(r["per_organ"][held]["auroc"])
        s3["per_organ"][held] = {n: float(np.nanmean(v)) for n, v in acc.items()}
        print(f"  {held:<16}"
              + "".join(f"{s3['per_organ'][held][k]:>12.4f}" for k in loo_models), flush=True)
    for name in loo_models:
        vals = [s3["per_organ"][o][name] for o in ORGANS]
        s3[name] = {"macro_auroc": float(np.nanmean(vals))}
    print("  " + "-" * (16 + 12 * len(loo_models)))
    print(f"  {'MACRO':<16}" + "".join(f"{s3[k]['macro_auroc']:>12.4f}" for k in loo_models))

    payload = {
        "leave_one_organ_out": {"S2_seen_drug_unseen_organ": s2,
                                "S3_unseen_drug_unseen_organ": s3},
        "config": {"seed": SEED, "n_splits": N_SPLITS, "n_bits": N_BITS,
                   "n_panel_genes": n_genes, "genes": genes,
                   "derangement": {ORGANS[i]: ORGANS[p] for i, p in enumerate(perm)},
                   "molecules_with_any_target": covered,
                   "n_molecules": len(molecules)},
        "results": results,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/mechanism.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # ── verdict ──────────────────────────────────────────────────────────
    def m(k):
        return results[k]["aggregate"]["macro_auroc"]["mean"]

    print("\n" + "=" * 86)
    print(f"{'model':<32}{'features':>10}{'macro-AUROC':>18}{'vs one-hot':>14}")
    print("-" * 86)
    for name in models:
        a = results[name]["aggregate"]["macro_auroc"]
        print(f"{name:<32}{results[name]['n_features']:>10}"
              f"{a['mean']:>11.4f} ±{a['std']:.3f}{m(name) - m('M1_morgan_onehot'):>+14.4f}")
    print("=" * 86)

    print("\nthe three comparisons that decide it:")
    print(f"  M5 mechanism      vs M1 one-hot        {m('M5_morgan_MECHANISM') - m('M1_morgan_onehot'):+.4f}")
    print(f"  M5 mechanism      vs M4 binding+onehot {m('M5_morgan_MECHANISM') - m('M4_morgan_binding_onehot'):+.4f}"
          "   <- coupling vs the same parts uncoupled")
    print(f"  M5 mechanism      vs M6 deranged       {m('M5_morgan_MECHANISM') - m('M6_morgan_mechanism_DERANGED'):+.4f}"
          "   <- needs the CORRECT organ")
    print("\nleave-one-organ-out (the decisive test):")
    for label, d in (("S2 seen drug, unseen organ", s2),
                     ("S3 unseen drug + unseen organ", s3)):
        gain = d["M5_morgan_MECHANISM"]["macro_auroc"] - d["M1_morgan_onehot"]["macro_auroc"]
        abl = d["M5_morgan_MECHANISM"]["macro_auroc"] - d["M6_morgan_mechanism_DERANGED"]["macro_auroc"]
        print(f"  {label:<32} mechanism vs one-hot {gain:+.4f}   "
              f"vs deranged {abl:+.4f}")

    print(f"\nwrote results/mechanism.json in {payload['runtime_seconds']}s")


if __name__ == "__main__":
    main()
