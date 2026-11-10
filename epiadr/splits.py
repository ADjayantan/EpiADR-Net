"""
Molecule-level Bemis-Murcko scaffold cross-validation.

Why molecule-level: the task has five rows per molecule (one per organ). Splitting
by ROW puts the same molecule in train and test, which leaks the molecular
structure the model is supposed to generalise over. Every split here assigns a
whole scaffold -- and therefore every row of every molecule sharing it -- to
exactly one fold.

`verify_no_leakage` is the proof that goes in the results table: it asserts
zero shared molecules AND zero shared scaffolds between train and test.
"""
from __future__ import annotations

import random
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def bemis_murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES; empty string for acyclic molecules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_groups(molecules: list[str]) -> dict[str, list[str]]:
    """Map scaffold -> molecules carrying it. Acyclic molecules each form their own group."""
    groups: dict[str, list[str]] = defaultdict(list)
    for i, smi in enumerate(molecules):
        sc = bemis_murcko_scaffold(smi)
        # An empty scaffold means acyclic: treat each as a singleton group rather
        # than pooling every acyclic molecule into one giant group.
        key = sc if sc else f"__acyclic_{i}__"
        groups[key].append(smi)
    return dict(groups)


def scaffold_kfold(
    molecules: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[set[str], set[str]]]:
    """
    Balanced scaffold k-fold over MOLECULES.

    Scaffolds are shuffled then greedily assigned to the currently smallest fold,
    which keeps fold sizes close without ever splitting a scaffold.
    Returns [(train_molecules, test_molecules), ...].
    """
    groups = scaffold_groups(molecules)
    keys = sorted(groups, key=lambda k: (-len(groups[k]), k))

    rng = random.Random(seed)
    rng.shuffle(keys)
    # Largest-first assignment after shuffling gives more even folds.
    keys.sort(key=lambda k: -len(groups[k]))

    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for key in keys:
        target = min(range(n_splits), key=lambda f: len(folds[f]))
        folds[target].extend(groups[key])

    out = []
    for f in range(n_splits):
        test = set(folds[f])
        train = set(m for g in range(n_splits) if g != f for m in folds[g])
        out.append((train, test))
    return out


def verify_no_leakage(train: set[str], test: set[str]) -> dict[str, int]:
    """Assert the split is clean. Raises AssertionError with the offending counts."""
    shared_mols = train & test
    train_sc = {bemis_murcko_scaffold(m) or f"__acyclic__{m}" for m in train}
    test_sc = {bemis_murcko_scaffold(m) or f"__acyclic__{m}" for m in test}
    shared_sc = {s for s in (train_sc & test_sc) if not s.startswith("__acyclic__")}

    assert not shared_mols, f"molecule leakage: {len(shared_mols)} shared molecules"
    assert not shared_sc, f"scaffold leakage: {len(shared_sc)} shared scaffolds"

    return {
        "train_molecules": len(train),
        "test_molecules": len(test),
        "shared_molecules": 0,
        "train_scaffolds": len(train_sc),
        "test_scaffolds": len(test_sc),
        "shared_scaffolds": 0,
    }
