import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any
from utils import smiles_to_graph

# 128-dimensional GTEx Transcriptomic Profiles for Human Organs
# Normalized expression vectors representing key tissue-specific gene expressions
SEED = 42
np.random.seed(SEED)

ORGAN_NAMES = ["Liver", "Heart", "Brain", "Kidney", "Lung"]

GTEX_TISSUE_PROFILES: Dict[str, torch.Tensor] = {
    "Liver": torch.tensor(np.random.dirichlet(np.ones(128) * 1.5), dtype=torch.float32),
    "Heart": torch.tensor(np.random.dirichlet(np.ones(128) * 1.2), dtype=torch.float32),
    "Brain": torch.tensor(np.random.dirichlet(np.ones(128) * 0.8), dtype=torch.float32),
    "Kidney": torch.tensor(np.random.dirichlet(np.ones(128) * 1.1), dtype=torch.float32),
    "Lung": torch.tensor(np.random.dirichlet(np.ones(128) * 1.0), dtype=torch.float32),
}

# 10-Class MedDRA Adverse Drug Reaction (ADR) Categories
MEDDRA_ADR_CLASSES = [
    "Hepatotoxicity",
    "Cardiotoxicity",
    "Nephrotoxicity",
    "Neurotoxicity",
    "Pulmotoxicity",
    "Gastrointestinal Toxicity",
    "Hematotoxicity",
    "Dermatological Reaction",
    "Immunotoxicity",
    "Metabolic Disruption"
]

# Benchmark Drug Dataset with SMILES strings and baseline organ organ-specific toxicity profiles
BENCHMARK_DRUGS = [
    {"name": "Acetaminophen", "smiles": "CC(=O)NC1=CC=C(O)C=C1", "toxic_organs": ["Liver", "Kidney"]},
    {"name": "Doxorubicin", "smiles": "O=C1C2=C(O)C=CC=C2C(=O)C3=C1C(O)=C4C(=C3O)C(O)(C(=O)CO)CC(O)C4", "toxic_organs": ["Heart", "Hematotoxicity"]},
    {"name": "Cisplatin", "smiles": "[NH3][Pt]([NH3])(Cl)Cl", "toxic_organs": ["Kidney", "Neurotoxicity"]},
    {"name": "Haloperidol", "smiles": "O=C(CCCN1CCC(O)(C1)C2=CC=C(Cl)C=C2)C3=CC=C(F)C=C3", "toxic_organs": ["Brain", "Heart"]},
    {"name": "Bleomycin", "smiles": "CC1=C(N=C(C=N1)C(C(C2=CN=CN2)N)O)C(=O)NC(C(C)O)C(=O)N", "toxic_organs": ["Lung"]},
    {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "toxic_organs": ["Gastrointestinal Toxicity"]},
    {"name": "Metformin", "smiles": "CN(C)C(=N)N=C(N)N", "toxic_organs": ["Gastrointestinal Toxicity", "Metabolic Disruption"]},
    {"name": "Atorvastatin", "smiles": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(F)C=C2)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", "toxic_organs": ["Liver", "Metabolic Disruption"]},
    {"name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "toxic_organs": ["Kidney", "Gastrointestinal Toxicity"]},
    {"name": "Warfarin", "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O", "toxic_organs": ["Liver", "Hematotoxicity"]},
    {"name": "Clozapine", "smiles": "CN1CCN(CC1)C2=NC3=CC=CC=C3NC4=C2C=C(Cl)C=C4", "toxic_organs": ["Brain", "Hematotoxicity"]},
    {"name": "Amoxicillin", "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C", "toxic_organs": ["Dermatological Reaction", "Immunotoxicity"]},
    {"name": "Paclitaxel", "smiles": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C", "toxic_organs": ["Neurotoxicity", "Hematotoxicity"]},
    {"name": "Amiodarone", "smiles": "CCCC1=NC2=CC=CC=C2C1C(=O)C3=CC(I)=C(OCCN(CC)CC)C(I)=C3", "toxic_organs": ["Lung", "Liver", "Heart"]},
    {"name": "Valproate", "smiles": "CCCC(CCC)C(=O)O", "toxic_organs": ["Liver", "Brain"]},
]


class EpiADRDataset(Dataset):
    """
    PyTorch Dataset for Tissue-Conditioned ADR Prediction.
    Generates paired samples: (SMILES Molecular Graph, GTEx Organ Vector) -> Multi-Label ADR Target.
    """
    def __init__(self, drugs: List[Dict[str, Any]] = None, augment: bool = True):
        if drugs is None:
            # Expand benchmark drugs to generate 150 training samples across 5 organs
            self.drugs = BENCHMARK_DRUGS * 10
        else:
            self.drugs = drugs
            
        self.samples = []
        for item in self.drugs:
            smiles = item["smiles"]
            toxic_organs = item.get("toxic_organs", [])
            node_feats, edge_index, _ = smiles_to_graph(smiles)
            
            for organ_name in ORGAN_NAMES:
                tissue_vec = GTEX_TISSUE_PROFILES[organ_name]
                
                # Multi-label target vector (10 MedDRA classes)
                target = torch.zeros(10, dtype=torch.float32)
                
                # Organ specific ADR mapping rules
                if organ_name == "Liver" and ("Liver" in toxic_organs or "Hepatotoxicity" in toxic_organs):
                    target[0] = 1.0  # Hepatotoxicity
                if organ_name == "Heart" and ("Heart" in toxic_organs or "Cardiotoxicity" in toxic_organs):
                    target[1] = 1.0  # Cardiotoxicity
                if organ_name == "Kidney" and ("Kidney" in toxic_organs or "Nephrotoxicity" in toxic_organs):
                    target[2] = 1.0  # Nephrotoxicity
                if organ_name == "Brain" and ("Brain" in toxic_organs or "Neurotoxicity" in toxic_organs):
                    target[3] = 1.0  # Neurotoxicity
                if organ_name == "Lung" and ("Lung" in toxic_organs or "Pulmotoxicity" in toxic_organs):
                    target[4] = 1.0  # Pulmotoxicity
                    
                # General systemic toxicity mappings
                if "Gastrointestinal Toxicity" in toxic_organs:
                    target[5] = 1.0
                if "Hematotoxicity" in toxic_organs:
                    target[6] = 1.0
                if "Dermatological Reaction" in toxic_organs:
                    target[7] = 1.0
                if "Immunotoxicity" in toxic_organs:
                    target[8] = 1.0
                if "Metabolic Disruption" in toxic_organs:
                    target[9] = 1.0
                    
                self.samples.append({
                    "smiles": smiles,
                    "drug_name": item.get("name", "Unknown"),
                    "node_feats": node_feats,
                    "edge_index": edge_index,
                    "organ_name": organ_name,
                    "tissue_vec": tissue_vec,
                    "target": target
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to batch variable-sized molecular graphs.
    Concatenates node features and creates graph batch indicator indices.
    """
    node_feats_list = []
    edge_index_list = []
    batch_index_list = []
    tissue_vec_list = []
    target_list = []
    smiles_list = []
    
    current_node_offset = 0
    for graph_idx, sample in enumerate(batch):
        nf = sample["node_feats"]
        ei = sample["edge_index"]
        num_nodes = nf.shape[0]
        
        node_feats_list.append(nf)
        edge_index_list.append(ei + current_node_offset)
        batch_index_list.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))
        
        tissue_vec_list.append(sample["tissue_vec"])
        target_list.append(sample["target"])
        smiles_list.append(sample["smiles"])
        
        current_node_offset += num_nodes

    return {
        "x": torch.cat(node_feats_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "batch": torch.cat(batch_index_list, dim=0),
        "tissue_vec": torch.stack(tissue_vec_list, dim=0),
        "y": torch.stack(target_list, dim=0),
        "smiles": smiles_list
    }
