import pytest
import torch
from model import MultiHeadGATLayer, GatedResidualFiLM, EpiADRNet
from utils import smiles_to_graph
from data_loader import GTEX_TISSUE_PROFILES

def test_gat_layer_forward():
    x = torch.randn(10, 512)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    gat = MultiHeadGATLayer(in_features=512, out_features=512, num_heads=8)
    h_out, alpha = gat(x, edge_index)

    assert h_out.shape == (10, 512)
    assert alpha.shape[0] == edge_index.shape[1]

def test_gated_film_layer_forward():
    h = torch.randn(10, 512)
    v_tissue = torch.randn(2, 256)
    batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    film = GatedResidualFiLM(tissue_dim=256, feature_dim=512)
    h_cond = film(h, v_tissue, batch_idx)

    assert h_cond.shape == (10, 512)

def test_epiadrnet_forward_and_mc_dropout():
    smiles = "CC(=O)NC1=CC=C(O)C=C1"
    node_feats, edge_index, _ = smiles_to_graph(smiles)
    batch = torch.zeros(node_feats.size(0), dtype=torch.long)
    tissue_vec = GTEX_TISSUE_PROFILES["Liver"].unsqueeze(0)

    model = EpiADRNet(
        in_features=17, hidden_dim=512, tissue_dim=256,
        num_classes=10, num_gat_layers=6, num_heads=8, dropout=0.3
    )
    assert model.count_parameters() > 1_000_000, "v3 model should have >1M parameters"

    logits, attn = model(node_feats, edge_index, batch, tissue_vec, return_attention=True)

    assert logits.shape == (1, 10)
    assert attn is not None

    mc_res = model.predict_mc_dropout(node_feats, edge_index, batch, tissue_vec, num_samples=5)
    assert mc_res["mean_probabilities"].shape == (1, 10)
    assert mc_res["uncertainty_sigma"].shape == (1, 10)
