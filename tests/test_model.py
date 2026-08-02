import torch

from data_loader import GTEX_TISSUE_PROFILES
from model import (
    DMPNNLayer,
    EpiADRNet,
    GenePathwayCrossAttention100M,
    GraphTransformerBlock,
)
from utils import smiles_to_graph


def test_graph_transformer_block_forward():
    x = torch.randn(10, 1536)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    block = GraphTransformerBlock(in_features=1536, out_features=1536, num_heads=16)
    h_out, alpha = block(x, edge_index)

    assert h_out.shape == (10, 1536)
    assert alpha.shape[0] == edge_index.shape[1]

def test_dmpnn_layer_forward():
    x = torch.randn(10, 1536)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    dmpnn = DMPNNLayer(node_dim=1536)
    x_out = dmpnn(x, edge_index)
    assert x_out.shape == (10, 1536)

def test_gene_pathway_cross_attention_forward():
    h = torch.randn(10, 1536)
    v_tissue = torch.randn(2, 1024)
    batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    attn_mod = GenePathwayCrossAttention100M(node_dim=1536, tissue_dim=1024, num_heads=16)
    h_out = attn_mod(h, v_tissue, batch_idx)
    assert h_out.shape == (10, 1536)

def test_epiadrnet_v5_foundation_forward_and_mc_dropout():
    smiles = "CC(=O)NC1=CC=C(O)C=C1"
    node_feats, edge_index, _ = smiles_to_graph(smiles)
    batch = torch.zeros(node_feats.size(0), dtype=torch.long)
    tissue_vec = GTEX_TISSUE_PROFILES["Liver"].unsqueeze(0)

    model = EpiADRNet(
        in_features=24, hidden_dim=1536, tissue_dim=1024,
        num_classes=10, num_gat_layers=12, num_heads=16, dropout=0.1
    )
    n_params = model.count_parameters()
    print(f"EpiADRNet v5 Parameters: {n_params:,}")
    assert n_params > 100_000_000, f"v5 model should have >100M parameters, got {n_params}"

    logits, attn = model(node_feats, edge_index, batch, tissue_vec, return_attention=True)

    assert logits.shape == (1, 10)
    assert attn is not None

    mc_res = model.predict_mc_dropout(node_feats, edge_index, batch, tissue_vec, num_samples=5)
    assert mc_res["mean_probabilities"].shape == (1, 10)
    assert mc_res["uncertainty_sigma"].shape == (1, 10)
