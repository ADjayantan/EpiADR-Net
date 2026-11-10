"""
Tests for the mechanism feature.

The point of these tests is to pin the property that makes the feature a valid
test of the tissue-conditioning hypothesis: it must be a function of the drug
AND the organ jointly, not of either alone. If a refactor ever breaks that, the
mechanism experiment silently becomes another organ-indicator experiment.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epiadr.gene_panel import ORGAN_TO_HPA_TISSUE, PANEL  # noqa: E402
from epiadr.mechanism import derange, load_gene_expression, mechanism_features  # noqa: E402
from epiadr.task import ORGANS  # noqa: E402

EXPR = os.path.join(ROOT, "data", "gene_tissue_expression.csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(EXPR),
    reason="run scripts/build_expression.py first",
)


@pytest.fixture(scope="module")
def expression():
    return load_gene_expression(PANEL, EXPR)


def test_panel_is_deduplicated_and_mapped():
    assert len(PANEL) == len(set(PANEL))
    assert set(ORGAN_TO_HPA_TISSUE) == set(ORGANS)


def test_expression_shapes_and_frac_sums_to_one(expression):
    frac, logn = expression
    assert frac.shape == (len(PANEL), len(ORGANS))
    assert logn.shape == (len(PANEL), len(ORGANS))
    assert np.isfinite(frac).all() and np.isfinite(logn).all()
    assert (frac >= 0).all() and (logn >= 0).all()
    sums = frac.sum(axis=1)
    assert np.allclose(sums[sums > 0], 1.0, atol=1e-4)


def test_known_biology_is_in_the_expression_table(expression):
    """hERG in heart, CYP3A4/BSEP in liver, OAT1 in kidney, NMDA receptor in brain."""
    frac, _ = expression
    gi = {g: i for i, g in enumerate(PANEL)}
    oi = {o: i for i, o in enumerate(ORGANS)}
    for gene, organ in [("KCNH2", "Heart"), ("CYP3A4", "Liver"),
                        ("ABCB11", "Liver"), ("SLC22A6", "Kidney"),
                        ("GRIN1", "Brain")]:
        row = frac[gi[gene]]
        assert row.argmax() == oi[organ], f"{gene} should peak in {organ}"


def test_feature_varies_across_organs_for_one_drug(expression):
    """Not a function of the drug alone — this is why it can beat a fingerprint."""
    frac, logn = expression
    binding = np.zeros(len(PANEL), dtype=np.float32)
    binding[PANEL.index("KCNH2")] = 1.0
    vectors = [mechanism_features(binding, i, frac, logn) for i in range(len(ORGANS))]
    assert any(not np.allclose(vectors[0], v) for v in vectors[1:])


def test_feature_varies_across_drugs_in_one_organ(expression):
    """Not a function of the organ alone — this is why it can beat a one-hot."""
    frac, logn = expression
    a = np.zeros(len(PANEL), dtype=np.float32)
    a[PANEL.index("KCNH2")] = 1.0
    b = np.zeros(len(PANEL), dtype=np.float32)
    b[PANEL.index("CYP3A4")] = 1.0
    heart = ORGANS.index("Heart")
    assert not np.allclose(
        mechanism_features(a, heart, frac, logn),
        mechanism_features(b, heart, frac, logn),
    )


def test_non_binder_gets_a_zero_coupled_block(expression):
    """A drug that hits nothing in the panel must contribute no coupled signal."""
    frac, logn = expression
    none = np.zeros(len(PANEL), dtype=np.float32)
    v = mechanism_features(none, ORGANS.index("Liver"), frac, logn)
    assert np.allclose(v[: 2 * len(PANEL)], 0.0)


def test_organ_share_is_a_fraction(expression):
    """The scalar summary of the hypothesis must stay in [0, 1] and sum to ~1."""
    frac, logn = expression
    rng = np.random.default_rng(0)
    binding = (rng.random(len(PANEL)) < 0.1).astype(np.float32)
    shares = [mechanism_features(binding, i, frac, logn)[-1] for i in range(len(ORGANS))]
    assert all(0.0 <= s <= 1.0 for s in shares)
    assert sum(shares) == pytest.approx(1.0, abs=1e-4)


def test_derangement_has_no_fixed_point():
    rng = np.random.default_rng(1)
    for _ in range(20):
        perm = derange(rng, len(ORGANS))
        assert sorted(perm) == list(range(len(ORGANS)))
        assert all(i != p for i, p in enumerate(perm))
