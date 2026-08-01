import pytest
import torch
import numpy as np
from utils import smiles_to_graph, get_bemis_murcko_scaffold, bemis_murcko_scaffold_split
from data_loader import GTEX_TISSUE_PROFILES, EpiADRDataset, ORGAN_NAMES

def test_smiles_to_graph():
    smiles = "CC(=O)NC1=CC=C(O)C=C1"  # Acetaminophen
    node_feats, edge_index, atom_symbols = smiles_to_graph(smiles)

    assert isinstance(node_feats, torch.Tensor)
    assert isinstance(edge_index, torch.Tensor)
    assert node_feats.ndim == 2
    assert node_feats.shape[1] == 10  # 10 features per atom node
    assert edge_index.shape[0] == 2  # Source -> Target edges
    assert len(atom_symbols) == node_feats.shape[0]

def test_gtex_tissue_profiles():
    assert len(GTEX_TISSUE_PROFILES) == 5
    for organ in ORGAN_NAMES:
        vec = GTEX_TISSUE_PROFILES[organ]
        assert vec.shape[0] == 128
        assert torch.is_tensor(vec)

def test_bemis_murcko_scaffold_split():
    dataset = EpiADRDataset()
    smiles_list = [sample["smiles"] for sample in dataset.samples]

    train_idx, val_idx, test_idx = bemis_murcko_scaffold_split(dataset, smiles_list)

    assert len(train_idx) + len(val_idx) + len(test_idx) == len(dataset)
    assert len(set(train_idx).intersection(set(test_idx))) == 0

    train_scaffolds = set(get_bemis_murcko_scaffold(smiles_list[i]) for i in train_idx)
    test_scaffolds = set(get_bemis_murcko_scaffold(smiles_list[i]) for i in test_idx)
    assert len(train_scaffolds.intersection(test_scaffolds)) == 0
