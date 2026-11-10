"""Tests for the corrected pipeline: real data, organ-dependent target, clean splits."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epiadr.eval import evaluate_organ_conditioned, expected_calibration_error  # noqa: E402
from epiadr.features import load_tissue_profiles, morgan_fingerprint  # noqa: E402
from epiadr.splits import bemis_murcko_scaffold, scaffold_kfold, verify_no_leakage  # noqa: E402
from epiadr.task import ORGAN_TO_ADR, ORGANS, load_task  # noqa: E402

CSV = os.path.join(ROOT, "data", "adr_records.csv")
TISSUE = os.path.join(ROOT, "data", "tissue_profiles.csv")


@pytest.fixture(scope="module")
def task():
    return load_task(CSV)


def test_real_dataset_is_loaded(task):
    """The 1465-molecule CSV must actually be in use, not the 5-drug stub."""
    assert len(task) == 7325
    assert len(task.molecules) == 1465


def test_target_depends_on_organ(task):
    """The whole point of the reformulation: y must vary across a molecule's organs."""
    by_mol = {}
    for smi, organ, y in zip(task.smiles, task.organ, task.y_organ, strict=False):
        by_mol.setdefault(smi, {})[organ] = y
    varying = sum(1 for d in by_mol.values() if len(set(d.values())) > 1)
    assert varying > 0, "target is organ-invariant; tissue conditioning is unmeasurable"
    # Most molecules should differ across organs for this to be a real task.
    assert varying / len(by_mol) > 0.5


def test_organ_mapping_is_complete():
    assert set(ORGAN_TO_ADR) == set(ORGANS)
    assert len(set(ORGAN_TO_ADR.values())) == len(ORGANS)


def test_tissue_profiles_are_real_and_distinct():
    """Must be the documented 128-dim GTEx PCA output, not random Dirichlet noise."""
    p = load_tissue_profiles(TISSUE)
    assert set(p) == set(ORGANS)
    dims = {len(v) for v in p.values()}
    assert dims == {128}
    M = np.stack([p[o] for o in ORGANS])
    assert np.isfinite(M).all()
    # No two organs may share a profile.
    for i in range(len(ORGANS)):
        for j in range(i + 1, len(ORGANS)):
            assert not np.allclose(M[i], M[j])


def test_scaffold_split_has_no_leakage(task):
    folds = scaffold_kfold(task.molecules, n_splits=5, seed=42)
    assert len(folds) == 5
    covered = set()
    for train, test in folds:
        stats = verify_no_leakage(train, test)   # raises on leakage
        assert stats["shared_molecules"] == 0
        assert stats["shared_scaffolds"] == 0
        covered |= test
    # Every molecule tested exactly once.
    assert covered == set(task.molecules)


def test_split_is_molecule_level_not_row_level(task):
    """All five organ rows of a molecule must land in the same fold."""
    train, test = scaffold_kfold(task.molecules, n_splits=5, seed=42)[0]
    train_rows = set(task.rows_for(train))
    test_rows = set(task.rows_for(test))
    assert not (train_rows & test_rows)
    assert len(train_rows) + len(test_rows) == len(task)
    for rows, mols in ((train_rows, train), (test_rows, test)):
        assert len(rows) == 5 * len(mols)


def test_scaffold_is_deterministic():
    smi = "CC(=O)NC1=CC=C(O)C=C1"
    assert bemis_murcko_scaffold(smi) == bemis_murcko_scaffold(smi)
    assert bemis_murcko_scaffold(smi) == "c1ccccc1"


def test_morgan_fingerprint_shape():
    fp = morgan_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert fp.shape == (2048,)
    assert set(np.unique(fp)).issubset({0.0, 1.0})
    assert fp.sum() > 0


def test_per_organ_auroc_ignores_base_rate():
    """A per-organ base-rate predictor must score 0.5 per organ, however high pooled is."""
    organs = ["Lung"] * 100 + ["Brain"] * 100
    y = [0] * 92 + [1] * 8 + [0] * 60 + [1] * 40
    prob = [0.08] * 100 + [0.40] * 100   # constant within each organ
    res = evaluate_organ_conditioned(np.array(y), np.array(prob), organs, ["Lung", "Brain"])
    assert res["macro_auroc"] == pytest.approx(0.5, abs=1e-9)
    assert res["pooled_auroc"] > 0.6     # inflated, as documented in AUDIT.md


def test_ece_is_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert expected_calibration_error(y, p, n_bins=10) < 0.02
