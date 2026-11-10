"""
Honest evaluation for the organ-conditioned ADR task.

Design notes that matter for interpreting the numbers:

* PER-ORGAN AUROC is the primary metric. Pooling all organs into one AUROC is
  inflated by the differing base rates (Lung 7.9% vs Brain 38.0%): a model that
  only learns "Brain is riskier than Lung" scores well while ranking no drug
  correctly. Within one organ the base rate is constant, so per-organ AUROC
  measures only what we care about -- can the model rank drugs by toxicity.
* AUPRC is reported alongside because the positives are the minority class for
  every organ, and AUROC is optimistic under imbalance.
* ECE is reported because the deployed app shows a probability; a probability
  that is not calibrated is not usable clinically.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def _safe(fn, y_true, y_prob) -> float:
    """Return NaN instead of raising when a fold has a single class."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(fn(y_true, y_prob))


def evaluate_organ_conditioned(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    organs: list[str],
    organ_names: list[str],
) -> dict:
    """
    Per-organ and macro metrics for the organ-specific binary target.

    y_true, y_prob: (n_rows,)   organs: (n_rows,) organ name per row
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    organs_arr = np.asarray(organs)

    per_organ: dict[str, dict[str, float]] = {}
    for organ in organ_names:
        mask = organs_arr == organ
        if not mask.any():
            continue
        per_organ[organ] = {
            "n": int(mask.sum()),
            "positives": int(y_true[mask].sum()),
            "prevalence": float(y_true[mask].mean()),
            "auroc": _safe(roc_auc_score, y_true[mask], y_prob[mask]),
            "auprc": _safe(average_precision_score, y_true[mask], y_prob[mask]),
            "ece": expected_calibration_error(y_true[mask], y_prob[mask]),
        }

    aurocs = [v["auroc"] for v in per_organ.values() if not np.isnan(v["auroc"])]
    auprcs = [v["auprc"] for v in per_organ.values() if not np.isnan(v["auprc"])]

    return {
        "per_organ": per_organ,
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "pooled_auroc": _safe(roc_auc_score, y_true, y_prob),
        "pooled_auprc": _safe(average_precision_score, y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "n_rows": int(y_true.size),
    }


def aggregate_folds(fold_results: list[dict], organ_names: list[str]) -> dict:
    """Mean +/- std across folds. The std is the number the README may quote."""
    def ms(vals):
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n_folds": 0}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_folds": len(vals),
        }

    out = {
        "macro_auroc": ms([r["macro_auroc"] for r in fold_results]),
        "macro_auprc": ms([r["macro_auprc"] for r in fold_results]),
        "pooled_auroc": ms([r["pooled_auroc"] for r in fold_results]),
        "ece": ms([r["ece"] for r in fold_results]),
        "per_organ": {},
    }
    for organ in organ_names:
        out["per_organ"][organ] = {
            "auroc": ms([r["per_organ"][organ]["auroc"] for r in fold_results if organ in r["per_organ"]]),
            "auprc": ms([r["per_organ"][organ]["auprc"] for r in fold_results if organ in r["per_organ"]]),
        }
    return out
