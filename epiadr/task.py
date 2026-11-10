"""
Organ-conditioned ADR task construction from data/adr_records.csv.

The raw CSV stores, for every (molecule, organ) pair, the same 10 ADR labels --
the label vector does not vary with organ (verified by scripts/audit_pipeline.py).
Ten labels replicated five times is not an organ-conditioned task: a model
conditioned on tissue has no organ-specific target to predict.

This module turns the same records into a task where the target genuinely
depends on the organ, by using the mapping that the ADR names already imply:

    ORGAN-SPECIFIC   Liver  -> Hepatotoxicity
                     Heart  -> Arrhythmia
                     Brain  -> Seizure
                     Kidney -> Nephrotoxicity
                     Lung   -> Pneumonitis

    SYSTEMIC         Nausea, Headache, Dizziness, Fatigue, Rash
                     (organ-invariant; carried as auxiliary targets)

A row is therefore (molecule, organ) -> (y_organ, y_systemic[5]), where y_organ
is the toxicity of THAT organ. y_organ varies across the five organs of one
molecule, so tissue conditioning has a real job and the tissue-shuffle ablation
is a meaningful test of whether the model uses it.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

ORGANS = ["Liver", "Heart", "Brain", "Kidney", "Lung"]

ORGAN_TO_ADR = {
    "Liver": "Hepatotoxicity",
    "Heart": "Arrhythmia",
    "Brain": "Seizure",
    "Kidney": "Nephrotoxicity",
    "Lung": "Pneumonitis",
}

SYSTEMIC_ADRS = ["Nausea", "Headache", "Dizziness", "Fatigue", "Rash"]

ALL_ADRS = list(ORGAN_TO_ADR.values()) + SYSTEMIC_ADRS


class DataContractError(Exception):
    """Raised when the ADR record file violates the documented schema."""


@dataclass
class OrganConditionedTask:
    """One row per (molecule, organ) pair."""

    smiles: list[str] = field(default_factory=list)
    organ: list[str] = field(default_factory=list)
    y_organ: list[int] = field(default_factory=list)
    y_systemic: list[list[int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.smiles)

    @property
    def molecules(self) -> list[str]:
        """Unique SMILES, in first-seen order."""
        seen, out = set(), []
        for s in self.smiles:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def rows_for(self, molecules: set[str]) -> list[int]:
        """Row indices whose molecule is in `molecules`."""
        return [i for i, s in enumerate(self.smiles) if s in molecules]

    def prevalence_by_organ(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for organ in ORGANS:
            idx = [i for i, o in enumerate(self.organ) if o == organ]
            out[organ] = sum(self.y_organ[i] for i in idx) / max(len(idx), 1)
        return out


def load_task(csv_path: str = "data/adr_records.csv") -> OrganConditionedTask:
    """Load and validate the ADR records, returning the organ-conditioned task."""
    if not os.path.exists(csv_path):
        raise DataContractError(f"ADR record file not found: {csv_path}")

    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise DataContractError(f"{csv_path} contains no records")

    missing = [c for c in ["smiles", "organ", *ALL_ADRS] if c not in rows[0]]
    if missing:
        raise DataContractError(f"{csv_path} missing required columns: {missing}")

    task = OrganConditionedTask()
    for i, row in enumerate(rows):
        organ = row["organ"]
        if organ not in ORGAN_TO_ADR:
            raise DataContractError(f"row {i}: unknown organ {organ!r}")

        try:
            y_org = int(row[ORGAN_TO_ADR[organ]])
            y_sys = [int(row[a]) for a in SYSTEMIC_ADRS]
        except ValueError as exc:
            raise DataContractError(f"row {i}: non-integer label ({exc})") from exc

        if y_org not in (0, 1) or any(v not in (0, 1) for v in y_sys):
            raise DataContractError(f"row {i}: labels must be binary")

        task.smiles.append(row["smiles"])
        task.organ.append(organ)
        task.y_organ.append(y_org)
        task.y_systemic.append(y_sys)

    return task
