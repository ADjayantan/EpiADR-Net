import pytest
import numpy as np
from metrics import calculate_metrics, bootstrap_confidence_intervals

def test_calculate_metrics_constant_label_protection():
    # Column 0 is constant 0, Column 1 is balanced
    y_true = np.array([[0, 0], [0, 1], [0, 0], [0, 1]], dtype=int)
    y_pred = np.array([[0.1, 0.2], [0.2, 0.8], [0.3, 0.1], [0.4, 0.9]], dtype=float)

    macro_auc, micro_pr, per_class = calculate_metrics(y_true, y_pred, label_names=["ConstClass", "BalancedClass"])

    assert per_class["ConstClass"] is None
    assert per_class["BalancedClass"] is not None
    assert macro_auc > 0.0

def test_bootstrap_confidence_intervals():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 1], [0, 0]], dtype=int)
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.1, 0.9], [0.7, 0.8], [0.1, 0.1]], dtype=float)

    res = bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=50, ci=95.0)

    assert "macro_auroc_ci" in res
    assert "micro_auprc_ci" in res
    assert res["macro_auroc_ci"]["lower"] <= res["macro_auroc_ci"]["upper"]
