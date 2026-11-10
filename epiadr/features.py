"""Molecular and tissue featurisation."""
from __future__ import annotations

import csv
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def morgan_fingerprint(smiles: str, n_bits: int = 2048) -> np.ndarray:
    """ECFP4 (Morgan radius 2) bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    gen = _MORGAN if n_bits == 2048 else rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=n_bits
    )
    return np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)


def fingerprint_matrix(molecules: list[str], n_bits: int = 2048) -> dict[str, np.ndarray]:
    """SMILES -> fingerprint, computed once and cached by the caller."""
    return {smi: morgan_fingerprint(smi, n_bits) for smi in molecules}


def load_tissue_profiles(csv_path: str = "data/tissue_profiles.csv") -> dict[str, np.ndarray]:
    """
    Real GTEx-derived organ profiles from data/tissue_profiles.csv.

    This is the file DATA_PROVENANCE.md documents (GTEx V8 median TPM -> PCA 128).
    The original training pipeline never read it; everything here does.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"tissue profile file not found: {csv_path}")

    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    feat_cols = sorted(c for c in rows[0] if c.startswith("feature_"))
    if not feat_cols:
        raise ValueError(f"{csv_path} has no feature_* columns")

    profiles: dict[str, np.ndarray] = {}
    for row in rows:
        vec = np.asarray([float(row[c]) for c in feat_cols], dtype=np.float32)
        if not np.isfinite(vec).all():
            raise ValueError(f"non-finite tissue features for organ {row['organ']!r}")
        profiles[row["organ"]] = vec
    return profiles
