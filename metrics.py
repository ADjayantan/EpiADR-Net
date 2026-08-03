
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None
) -> tuple[float, float, dict[str, float | None]]:
    """
    Computes Macro-AUROC and Micro-AUPRC across multi-label targets.
    Protects against constant-label columns by returning None for single-class columns.
    """
    if label_names is None:
        label_names = [f"Class_{i}" for i in range(y_true.shape[1])]

    per_class_auroc: dict[str, float | None] = {}
    valid_aurocs: list[float] = []

    for idx, name in enumerate(label_names):
        col_true = y_true[:, idx]
        col_pred = y_pred[:, idx]

        # Guard against single-class constant labels
        if len(np.unique(col_true)) < 2:
            per_class_auroc[name] = None
        else:
            try:
                score = float(roc_auc_score(col_true, col_pred))
                per_class_auroc[name] = round(score, 4)
                valid_aurocs.append(score)
            except Exception:
                per_class_auroc[name] = None

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.5

    # Micro-AUPRC
    try:
        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
        micro_auprc = float(auc(recall, precision))
    except Exception:
        micro_auprc = 0.0

    return macro_auroc, micro_auprc, per_class_auroc


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 95.0,
    seed: int = 42
) -> dict[str, dict[str, float]]:
    """
    Computes non-parametric percentile bootstrap confidence intervals (2.5% - 97.5%)
    for Macro-AUROC and Micro-AUPRC over n_bootstraps resamples with replacement.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    bootstrapped_aurocs: list[float] = []
    bootstrapped_auprcs: list[float] = []

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue

        macro_auc, micro_pr, _ = calculate_metrics(y_true[indices], y_pred[indices])
        bootstrapped_aurocs.append(macro_auc)
        bootstrapped_auprcs.append(micro_pr)

    alpha = (100.0 - ci) / 2.0
    auroc_lower = float(np.percentile(bootstrapped_aurocs, alpha))
    auroc_upper = float(np.percentile(bootstrapped_aurocs, 100.0 - alpha))

    auprc_lower = float(np.percentile(bootstrapped_auprcs, alpha))
    auprc_upper = float(np.percentile(bootstrapped_auprcs, 100.0 - alpha))

    return {
        "macro_auroc_ci": {
            "mean": float(np.mean(bootstrapped_aurocs)),
            "lower": round(auroc_lower, 4),
            "upper": round(auroc_upper, 4),
        },
        "micro_auprc_ci": {
            "mean": float(np.mean(bootstrapped_auprcs)),
            "lower": round(auprc_lower, 4),
            "upper": round(auprc_upper, 4),
        }
    }


def calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculates Expected Calibration Error (ECE) for multi-label predictions across n_bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(y_true[in_bin]))
            avg_confidence_in_bin = float(np.mean(y_pred[in_bin]))
            ece += abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return round(float(ece), 4)

