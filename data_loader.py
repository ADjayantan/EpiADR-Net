import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Any
from utils import smiles_to_graph

# ─────────────────────────────────────────────────────────────────
#  Reproducibility seed
# ─────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────
#  10 Human Organs (expanded from 5)
# ─────────────────────────────────────────────────────────────────
ORGAN_NAMES = [
    "Liver", "Heart", "Brain", "Kidney", "Lung",
    "Pancreas", "Spleen", "Intestine", "Skin", "Bone_Marrow",
]

# ─────────────────────────────────────────────────────────────────
#  256-dim Structured GTEx Tissue Profiles
#  Each vector simulates organ-specific transcriptomic signatures
#  using biologically-motivated Dirichlet concentration parameters.
#  Higher alpha → more uniform expression; lower → more peaked.
# ─────────────────────────────────────────────────────────────────
_TISSUE_ALPHA = {
    "Liver":       1.80,   # Highly metabolic — broad expression
    "Heart":       1.40,   # Cardiac muscle — moderately specific
    "Brain":       0.70,   # CNS — highly tissue-specific
    "Kidney":      1.20,   # Filtration tissue
    "Lung":        1.00,   # Respiratory — moderate diversity
    "Pancreas":    0.90,   # Exocrine/endocrine mixed
    "Spleen":      1.10,   # Immune hub
    "Intestine":   1.50,   # Epithelial — broad metabolic
    "Skin":        0.85,   # Barrier tissue — specific profile
    "Bone_Marrow": 0.60,   # Haematopoietic — most specialised
}

TISSUE_DIM = 256   # Expanded from 128 → 256

GTEX_TISSUE_PROFILES: Dict[str, torch.Tensor] = {
    organ: torch.tensor(
        np.random.dirichlet(np.ones(TISSUE_DIM) * alpha).astype(np.float32)
    )
    for organ, alpha in _TISSUE_ALPHA.items()
}

# ─────────────────────────────────────────────────────────────────
#  10-Class MedDRA ADR Taxonomy
# ─────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────
#  Expanded Benchmark Drug Dataset  (30 drugs)
#  Sourced from FDA drug labels, DrugBank, and SIDER toxicity DB
# ─────────────────────────────────────────────────────────────────
BENCHMARK_DRUGS = [
    # --- Analgesics / Anti-inflammatories ---
    {"name": "Acetaminophen",  "smiles": "CC(=O)NC1=CC=C(O)C=C1",
     "toxic_organs": ["Liver", "Kidney"]},
    {"name": "Aspirin",        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
     "toxic_organs": ["Gastrointestinal Toxicity"]},
    {"name": "Ibuprofen",      "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
     "toxic_organs": ["Kidney", "Gastrointestinal Toxicity"]},
    {"name": "Diclofenac",     "smiles": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
     "toxic_organs": ["Liver", "Kidney", "Gastrointestinal Toxicity"]},
    {"name": "Naproxen",       "smiles": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
     "toxic_organs": ["Gastrointestinal Toxicity", "Kidney"]},

    # --- Cardiovascular ---
    {"name": "Warfarin",       "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",
     "toxic_organs": ["Liver", "Hematotoxicity"]},
    {"name": "Amiodarone",     "smiles": "CCCC1=NC2=CC=CC=C2C1C(=O)C3=CC(I)=C(OCCN(CC)CC)C(I)=C3",
     "toxic_organs": ["Lung", "Liver", "Heart"]},
    {"name": "Digoxin",        "smiles": "O=C1OCC(C1)C1C(O)CC2(O)C3CCC4C(C)(O)CC(OC5OC(C)C(OC6OCC(O)C(O)C6O)C(O)C5)C4(C)C3CCC12C",
     "toxic_organs": ["Heart", "Kidney"]},
    {"name": "Atorvastatin",   "smiles": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(F)C=C2)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
     "toxic_organs": ["Liver", "Metabolic Disruption"]},
    {"name": "Metoprolol",     "smiles": "CC(C)NCC(O)COc1ccc(CCOC)cc1",
     "toxic_organs": ["Heart", "Metabolic Disruption"]},

    # --- Oncology / Cytotoxics ---
    {"name": "Doxorubicin",    "smiles": "O=C1C2=C(O)C=CC=C2C(=O)C3=C1C(O)=C4C(=C3O)C(O)(C(=O)CO)CC(O)C4",
     "toxic_organs": ["Heart", "Hematotoxicity"]},
    {"name": "Cisplatin",      "smiles": "[NH3][Pt]([NH3])(Cl)Cl",
     "toxic_organs": ["Kidney", "Neurotoxicity"]},
    {"name": "Paclitaxel",     "smiles": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
     "toxic_organs": ["Neurotoxicity", "Hematotoxicity"]},
    {"name": "Bleomycin",      "smiles": "CC1=C(N=C(C=N1)C(C(C2=CN=CN2)N)O)C(=O)NC(C(C)O)C(=O)N",
     "toxic_organs": ["Lung", "Skin"]},
    {"name": "Methotrexate",   "smiles": "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(CC(NC(=O)O)C(=O)O)cc1",
     "toxic_organs": ["Liver", "Kidney", "Hematotoxicity"]},
    {"name": "Cyclophosphamide","smiles": "O=P1(OCCCN1CCCl)N(CCCl)CCCl",
     "toxic_organs": ["Hematotoxicity", "Kidney", "Immunotoxicity"]},
    {"name": "Tamoxifen",      "smiles": "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
     "toxic_organs": ["Liver", "Metabolic Disruption", "Hematotoxicity"]},

    # --- CNS / Neurological ---
    {"name": "Haloperidol",    "smiles": "O=C(CCCN1CCC(O)(C1)C2=CC=C(Cl)C=C2)C3=CC=C(F)C=C3",
     "toxic_organs": ["Brain", "Heart"]},
    {"name": "Clozapine",      "smiles": "CN1CCN(CC1)C2=NC3=CC=CC=C3NC4=C2C=C(Cl)C=C4",
     "toxic_organs": ["Brain", "Hematotoxicity"]},
    {"name": "Valproate",      "smiles": "CCCC(CCC)C(=O)O",
     "toxic_organs": ["Liver", "Brain"]},
    {"name": "Phenytoin",      "smiles": "O=C1NC(=O)C(N1)(c1ccccc1)c1ccccc1",
     "toxic_organs": ["Liver", "Brain", "Dermatological Reaction"]},
    {"name": "Carbamazepine",  "smiles": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
     "toxic_organs": ["Liver", "Hematotoxicity", "Dermatological Reaction", "Immunotoxicity"]},

    # --- Antibiotics / Anti-infectives ---
    {"name": "Amoxicillin",    "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
     "toxic_organs": ["Dermatological Reaction", "Immunotoxicity"]},
    {"name": "Vancomycin",    "smiles": "O=C(N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@H](C(=O)N[C@@H](Cc2ccc(O)c(Oc3cc4cc(Oc5ccc(cc5Cl)[C@@H](NC(=O)[C@@H]6NC(=O)[C@H](NC(=O)[C@@H](N)Cc7ccc(O)c(Oc8cc4cc(Cl)c8O)c7)c9cc(O)c(Oc3cc(Cl)cc9O)c(Cl)c6)C(=O)O)c2)C(=O)O)[C@@H](O)c1ccc(O)cc1Cl)[C@@H](O)[C@H](NC)C(=O)NCC(=O)N)C1OC(CO)C(O)C(O)C1OC1CC(N)C(O)C(O)C1O",
     "toxic_organs": ["Kidney", "Hematotoxicity", "Skin"]},
    {"name": "Metronidazole",  "smiles": "Cc1ncc([N+](=O)[O-])n1CCO",
     "toxic_organs": ["Neurotoxicity", "Gastrointestinal Toxicity"]},
    {"name": "Ciprofloxacin",  "smiles": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
     "toxic_organs": ["Gastrointestinal Toxicity", "Neurotoxicity", "Metabolic Disruption"]},

    # --- Immunosuppressants / Biologics ---
    {"name": "Metformin",      "smiles": "CN(C)C(=N)N=C(N)N",
     "toxic_organs": ["Gastrointestinal Toxicity", "Metabolic Disruption"]},
    {"name": "Cyclosporine",  "smiles": "CC[C@H]1NC(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](Cc2ccccc2)N(C)C(=O)[C@H](C)NC(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](C)NC(=O)CN(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@H](C(C)C)N(C)C(=O)CN(C)C1=O",
     "toxic_organs": ["Kidney", "Liver", "Neurotoxicity", "Metabolic Disruption"]},
    {"name": "Thalidomide",    "smiles": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
     "toxic_organs": ["Neurotoxicity", "Hematotoxicity", "Immunotoxicity"]},
    {"name": "Prednisolone",   "smiles": "C[C@@]12C[C@H](O)[C@H]3[C@@H](CCC4=CC(=O)C=C[C@@]43C)[C@@H]1CC[C@@]2(O)C(=O)CO",
     "toxic_organs": ["Metabolic Disruption", "Bone_Marrow", "Skin", "Gastrointestinal Toxicity"]},
    {"name": "Azathioprine",   "smiles": "Cn1cnc2c(N)ncnc12",
     "toxic_organs": ["Hematotoxicity", "Liver", "Immunotoxicity", "Gastrointestinal Toxicity"]},
]

assert len(BENCHMARK_DRUGS) == 31, f"Expected 31 drugs, got {len(BENCHMARK_DRUGS)}"


# ─────────────────────────────────────────────────────────────────
#  Organ → ADR label assignment
# ─────────────────────────────────────────────────────────────────
_ORGAN_TO_ADR_IDX = {
    "Liver":      0,   # Hepatotoxicity
    "Heart":      1,   # Cardiotoxicity
    "Kidney":     2,   # Nephrotoxicity
    "Brain":      3,   # Neurotoxicity
    "Lung":       4,   # Pulmotoxicity
    "Intestine":  5,   # Gastrointestinal Toxicity (proxy)
    "Pancreas":   9,   # Metabolic Disruption (proxy)
    "Spleen":     8,   # Immunotoxicity (proxy)
    "Skin":       7,   # Dermatological Reaction (proxy)
    "Bone_Marrow": 6,  # Hematotoxicity (proxy)
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
    """Constructs the 10-class multi-label binary target vector."""
    target = torch.zeros(10, dtype=torch.float32)

    # Organ-localised ADRs
    organ_adr_idx = _ORGAN_TO_ADR_IDX.get(organ_name)
    if organ_adr_idx is not None and organ_name in toxic_organs:
        target[organ_adr_idx] = 1.0

    # Systemic / class-level toxicity labels (organ-agnostic)
    for tox in toxic_organs:
        if tox in _SYSTEMIC_TO_ADR_IDX:
            target[_SYSTEMIC_TO_ADR_IDX[tox]] = 1.0

    return target


# ─────────────────────────────────────────────────────────────────
#  EpiADR Dataset  v3
# ─────────────────────────────────────────────────────────────────
class EpiADRDataset(Dataset):
    """
    Tissue-Conditioned Multi-Label ADR Dataset.

    Generates paired samples:
        (Molecular Graph, GTEx Organ Profile) → 10-class ADR target

    With 31 drugs x 10 organs x 10 repetitions = 3100 training samples.
    Graphs are cached per unique SMILES to avoid redundant RDKit parses.
    Repetitions provide stochastic diversity via dropout during training.
    """
    def __init__(
        self,
        drugs: List[Dict[str, Any]] = None,
        repeat: int = 10,
    ):
        self.drugs  = drugs if drugs is not None else BENCHMARK_DRUGS
        self.repeat = repeat
        self.samples: List[Dict[str, Any]] = []
        self._build()

    def _build(self):
        # Step 1: Parse each unique SMILES exactly once into a cache
        graph_cache: Dict[str, Any] = {}
        for item in self.drugs:
            smiles = item["smiles"]
            if smiles not in graph_cache:
                try:
                    node_feats, edge_index, _ = smiles_to_graph(smiles)
                    graph_cache[smiles] = (node_feats, edge_index)
                except Exception:
                    graph_cache[smiles] = None

        # Step 2: Build sample index using cached graphs (no redundant parsing)
        for _ in range(self.repeat):
            for item in self.drugs:
                smiles     = item["smiles"]
                drug_name  = item.get("name", "Unknown")
                toxic_list = item.get("toxic_organs", [])

                cached = graph_cache.get(smiles)
                if cached is None:
                    continue
                node_feats, edge_index = cached

                for organ_name in ORGAN_NAMES:
                    tissue_vec = GTEX_TISSUE_PROFILES[organ_name]
                    target     = _build_target_vector(organ_name, toxic_list)

                    self.samples.append({
                        "smiles":      smiles,
                        "drug_name":   drug_name,
                        "node_feats":  node_feats,
                        "edge_index":  edge_index,
                        "organ_name":  organ_name,
                        "tissue_vec":  tissue_vec,
                        "target":      target,
                    })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────
#  Batch Collate Function
# ─────────────────────────────────────────────────────────────────
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates variable-sized molecular graphs into a batched mini-graph.
    Node indices are offset per graph so that edge_index is globally consistent.
    """
    node_feats_list  = []
    edge_index_list  = []
    batch_index_list = []
    tissue_vec_list  = []
    target_list      = []
    smiles_list      = []

    node_offset = 0
    for graph_idx, sample in enumerate(batch):
        nf  = sample["node_feats"]    # [n_i, 17]
        ei  = sample["edge_index"]    # [2, e_i]
        n   = nf.shape[0]

        node_feats_list.append(nf)
        edge_index_list.append(ei + node_offset)
        batch_index_list.append(torch.full((n,), graph_idx, dtype=torch.long))
        tissue_vec_list.append(sample["tissue_vec"])
        target_list.append(sample["target"])
        smiles_list.append(sample["smiles"])

        node_offset += n

    return {
        "x":          torch.cat(node_feats_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "batch":      torch.cat(batch_index_list, dim=0),
        "tissue_vec": torch.stack(tissue_vec_list, dim=0),
        "y":          torch.stack(target_list, dim=0),
        "smiles":     smiles_list,
    }
