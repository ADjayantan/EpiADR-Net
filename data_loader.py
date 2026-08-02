import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from rdkit import Chem
from utils import smiles_to_graph

# ─────────────────────────────────────────────────────────────────
#  Data Contract Exceptions & Loaders
# ─────────────────────────────────────────────────────────────────
class DataContractError(Exception):
    """Raised when dataset CSV files fail schema or integrity contracts."""
    pass


class GTExTissueLoader:
    """
    Contract-conforming GTEx tissue profile loader.
    Validates data/tissue_profiles.csv against contract specifications.
    """
    def __init__(self, profiles: Dict[str, torch.Tensor]):
        self.profiles = profiles

    @classmethod
    def from_csv(cls, csv_path: str) -> "GTExTissueLoader":
        df = pd.read_csv(csv_path)

        expected_organs = {"Liver", "Heart", "Brain", "Kidney", "Lung"}
        actual_organs = set(df["organ"].unique())

        if not expected_organs.issubset(actual_organs):
            raise DataContractError(
                f"Missing required organs in {csv_path}. Expected at least {expected_organs}, got {actual_organs}"
            )

        feat_cols = [c for c in df.columns if c.startswith("feature_")]
        if len(feat_cols) != 128:
            raise DataContractError(
                f"Expected 128 tissue feature columns in {csv_path}, found {len(feat_cols)}"
            )

        if df[feat_cols].isna().any().any():
            raise DataContractError(f"NaN values found in tissue features of {csv_path}")

        profiles = {}
        for _, row in df.iterrows():
            organ = str(row["organ"])
            vec = torch.tensor(row[feat_cols].values.astype(np.float32))
            profiles[organ] = vec

        return cls(profiles)

    def __contains__(self, organ: str) -> bool:
        return organ in self.profiles

    def __getitem__(self, organ: str) -> torch.Tensor:
        if organ not in self.profiles:
            raise DataContractError(f"Organ '{organ}' not found in loaded GTEx tissue profiles.")
        return self.profiles[organ]


def load_labeled_records(csv_path: str, tissue: GTExTissueLoader) -> pd.DataFrame:
    """
    Contract-conforming labeled ADR record loader.
    Validates data/adr_records.csv against data contract constraints.
    """
    df = pd.read_csv(csv_path)

    expected_header = [
        "smiles", "organ", "Hepatotoxicity", "Arrhythmia", "Seizure",
        "Nephrotoxicity", "Pneumonitis", "Nausea", "Headache", "Dizziness",
        "Fatigue", "Rash"
    ]

    if list(df.columns) != expected_header:
        raise DataContractError(
            f"Header mismatch in {csv_path}.\nExpected: {expected_header}\nGot: {list(df.columns)}"
        )

    if df.isna().any().any():
        raise DataContractError(f"Missing/NaN values detected in {csv_path}")

    for idx, row in df.iterrows():
        smiles = str(row["smiles"])
        organ = str(row["organ"])

        if organ not in tissue:
            raise DataContractError(f"Row {idx}: Organ '{organ}' not in GTEx tissue loader.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise DataContractError(f"Row {idx}: Invalid SMILES string '{smiles}'")

        for adr in expected_header[2:]:
            val = row[adr]
            if val not in (0, 1, 0.0, 1.0):
                raise DataContractError(f"Row {idx}: Non-binary target label for {adr}: {val}")

    return df


# ─────────────────────────────────────────────────────────────────
#  Reproducibility seed
# ─────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────
#  10 Human Organs
# ─────────────────────────────────────────────────────────────────
ORGAN_NAMES = [
    "Liver", "Heart", "Brain", "Kidney", "Lung",
    "Pancreas", "Spleen", "Intestine", "Skin", "Bone_Marrow",
]

_TISSUE_ALPHA = {
    "Liver":       1.80,
    "Heart":       1.40,
    "Brain":       0.70,
    "Kidney":      1.20,
    "Lung":        1.00,
    "Pancreas":    0.90,
    "Spleen":      1.10,
    "Intestine":   1.50,
    "Skin":        0.85,
    "Bone_Marrow": 0.60,
}

TISSUE_DIM = 1024

GTEX_TISSUE_PROFILES: Dict[str, torch.Tensor] = {
    organ: torch.tensor(
        np.random.dirichlet(np.ones(TISSUE_DIM) * alpha).astype(np.float32)
    )
    for organ, alpha in _TISSUE_ALPHA.items()
}

MEDDRA_ADR_CLASSES = [
    "Hepatotoxicity",        # 0
    "Cardiotoxicity",        # 1
    "Nephrotoxicity",        # 2
    "Neurotoxicity",         # 3
    "Pulmotoxicity",         # 4
    "Gastrointestinal Toxicity",  # 5
    "Hematotoxicity",        # 6
    "Dermatological Reaction",    # 7
    "Immunotoxicity",        # 8
    "Metabolic Disruption",  # 9
]

_RAW_DRUG_DEFS = [
    ("Acetaminophen", "CC(=O)NC1=CC=C(O)C=C1", ["Liver", "Kidney"]),
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", ["Gastrointestinal Toxicity"]),
    ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", ["Kidney", "Gastrointestinal Toxicity"]),
    ("Diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", ["Liver", "Kidney", "Gastrointestinal Toxicity"]),
    ("Naproxen", "COc1ccc2cc(C(C)C(=O)O)ccc2c1", ["Gastrointestinal Toxicity", "Kidney"]),
]

BENCHMARK_DRUGS: List[Dict[str, Any]] = []
for idx, (name, smiles, tox) in enumerate(_RAW_DRUG_DEFS):
    BENCHMARK_DRUGS.append({"name": name, "smiles": smiles, "toxic_organs": tox})

_ORGAN_TO_ADR_IDX = {
    "Liver":      0,
    "Heart":      1,
    "Kidney":     2,
    "Brain":      3,
    "Lung":       4,
    "Intestine":  5,
    "Pancreas":   9,
    "Spleen":     8,
    "Skin":       7,
    "Bone_Marrow": 6,
}

_SYSTEMIC_TO_ADR_IDX = {
    "Hepatotoxicity":           0,
    "Cardiotoxicity":           1,
    "Nephrotoxicity":           2,
    "Neurotoxicity":            3,
    "Pulmotoxicity":            4,
    "Gastrointestinal Toxicity": 5,
    "Hematotoxicity":           6,
    "Dermatological Reaction":  7,
    "Immunotoxicity":           8,
    "Metabolic Disruption":     9,
}

def _build_target_vector(organ_name: str, toxic_organs: List[str]) -> torch.Tensor:
    target = torch.zeros(10, dtype=torch.float32)
    organ_adr_idx = _ORGAN_TO_ADR_IDX.get(organ_name)
    if organ_adr_idx is not None and organ_name in toxic_organs:
        target[organ_adr_idx] = 1.0

    for tox in toxic_organs:
        if tox in _SYSTEMIC_TO_ADR_IDX:
            target[_SYSTEMIC_TO_ADR_IDX[tox]] = 1.0

    return target

class EpiADRDataset(Dataset):
    def __init__(
        self,
        drugs: List[Dict[str, Any]] = None,
        repeat: int = 4,
    ):
        self.drugs = drugs if drugs is not None else BENCHMARK_DRUGS
        self.repeat = repeat
        self.samples: List[Dict[str, Any]] = []
        self._build()

    def _build(self):
        graph_cache: Dict[str, Any] = {}
        for item in self.drugs:
            smiles = item["smiles"]
            if smiles not in graph_cache:
                try:
                    node_feats, edge_index, _ = smiles_to_graph(smiles)
                    graph_cache[smiles] = (node_feats, edge_index)
                except Exception:
                    graph_cache[smiles] = None

        for _ in range(self.repeat):
            for item in self.drugs:
                smiles = item["smiles"]
                drug_name = item.get("name", "Unknown")
                toxic_list = item.get("toxic_organs", [])

                cached = graph_cache.get(smiles)
                if cached is None:
                    continue
                node_feats, edge_index = cached

                for organ_name in ORGAN_NAMES:
                    tissue_vec = GTEX_TISSUE_PROFILES[organ_name]
                    target = _build_target_vector(organ_name, toxic_list)

                    self.samples.append({
                        "smiles": smiles,
                        "drug_name": drug_name,
                        "node_feats": node_feats,
                        "edge_index": edge_index,
                        "organ_name": organ_name,
                        "tissue_vec": tissue_vec,
                        "target": target,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    node_feats_list = []
    edge_index_list = []
    batch_index_list = []
    tissue_vec_list = []
    target_list = []
    smiles_list = []

    node_offset = 0
    for graph_idx, sample in enumerate(batch):
        nf = sample["node_feats"]
        ei = sample["edge_index"]
        n = nf.shape[0]

        node_feats_list.append(nf)
        edge_index_list.append(ei + node_offset)
        batch_index_list.append(torch.full((n,), graph_idx, dtype=torch.long))
        tissue_vec_list.append(sample["tissue_vec"])
        target_list.append(sample["target"])
        smiles_list.append(sample["smiles"])

        node_offset += n

    return {
        "x": torch.cat(node_feats_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "batch": torch.cat(batch_index_list, dim=0),
        "tissue_vec": torch.stack(tissue_vec_list, dim=0),
        "y": torch.stack(target_list, dim=0),
        "smiles": smiles_list,
    }
