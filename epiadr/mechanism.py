"""
Mechanism features: drug-target binding coupled to per-organ target expression.

The negative result in RESULTS.md has a specific cause. With five organs, organ
identity is ~2.3 bits, which a one-hot carries in full; a 128-dimensional GTEx
profile adds 123 dimensions the label does not depend on. Any feature that is a
function of the organ ALONE is therefore capped at what the one-hot already does.

A mechanism feature is not such a function. It is a product:

    m_g(drug, organ) = binds(drug, g) x expression(g, organ)

The mask depends on the drug, the magnitude on the organ, and the feature on
both jointly. It cannot be recovered from an organ indicator, because two drugs
in the same organ get different vectors; nor from the molecule alone, because
one drug gets different vectors across organs. This is what makes the
tissue-conditioning hypothesis testable rather than vacuous.

`frac` weighting is the form the hypothesis actually predicts: a target's SHARE
of its expression in an organ measures whether the drug's target is
preferentially present there, which is the claim ("the drug hits something this
organ depends on"), rather than whether the gene is abundant everywhere.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from epiadr.gene_panel import PANEL
from epiadr.task import ORGANS


def load_drug_targets(path: str = "data/drug_target_matrix.csv") -> tuple[dict[str, np.ndarray], list[str]]:
    """SMILES -> binary panel-gene binding vector. Produced by scripts/fetch_targets.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run scripts/fetch_targets.py first")

    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    genes = [g for g in PANEL if g in rows[0]]
    out = {
        r["smiles"]: np.asarray([float(r[g]) for g in genes], dtype=np.float32)
        for r in rows
    }
    return out, genes


def load_gene_expression(
    genes: list[str],
    path: str = "data/gene_tissue_expression.csv",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (frac, log_ntpm), each shaped (n_genes, n_organs), row-aligned to `genes`.

    frac[g, o]      organ o's share of gene g's expression across the five organs
    log_ntpm[g, o]  log1p of raw nTPM, for the abundance-weighted variant
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run scripts/build_expression.py first")

    with open(path, encoding="utf-8") as fh:
        table = {r["gene"]: r for r in csv.DictReader(fh)}

    frac = np.zeros((len(genes), len(ORGANS)), dtype=np.float32)
    logn = np.zeros((len(genes), len(ORGANS)), dtype=np.float32)
    for i, g in enumerate(genes):
        row = table.get(g)
        if row is None:
            continue
        for j, organ in enumerate(ORGANS):
            frac[i, j] = float(row[f"{organ}_frac"])
            logn[i, j] = np.log1p(float(row[f"{organ}_ntpm"]))
    return frac, logn


def mechanism_features(
    binding: np.ndarray,
    organ_index: int,
    frac: np.ndarray,
    log_ntpm: np.ndarray,
) -> np.ndarray:
    """
    Feature vector for one (drug, organ) pair.

    Layout: [ coupled_frac (n_genes) | coupled_logn (n_genes) | 6 aggregates ]

    The aggregates summarise the drug's total mechanistic load in this organ and,
    crucially, `organ_share` -- how much of the drug's whole-body target
    expression sits in THIS organ. That scalar is the compact statement of the
    hypothesis.
    """
    coupled_frac = binding * frac[:, organ_index]
    coupled_logn = binding * log_ntpm[:, organ_index]

    n_targets = float(binding.sum())
    burden_here = float(coupled_logn.sum())
    burden_all = float((binding[:, None] * log_ntpm).sum())
    organ_share = burden_here / burden_all if burden_all > 0 else 0.0

    aggregates = np.asarray([
        n_targets,
        burden_here,
        float(coupled_frac.sum()),
        float(coupled_frac.max(initial=0.0)),
        float(coupled_logn.max(initial=0.0)),
        organ_share,
    ], dtype=np.float32)

    return np.concatenate([coupled_frac, coupled_logn, aggregates])


def derange(rng: np.random.Generator, n: int = len(ORGANS)) -> list[int]:
    """A permutation with no fixed point, for the ablation."""
    idx = list(range(n))
    while True:
        rng.shuffle(idx)
        if all(a != b for a, b in enumerate(idx)):
            return idx
