import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class GATLayer(nn.Module):
    """
    Single-Head Graph Attention Network (GAT) Layer.
    Computes node feature transformations and extracts explicit attention weights.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [N, in_features]
            edge_index: Edge connectivity [2, E]
        Returns:
            x_out: Updated node features [N, out_features]
            alpha: Attention coefficients [E]
        """
        h = self.W(x)  # [N, out_features]
        row, col = edge_index[0], edge_index[1]
        
        # Concatenate source and target node representations for attention mechanism
        h_cat = torch.cat([h[row], h[col]], dim=1)  # [E, 2 * out_features]
        e = self.leakyrelu(self.a(h_cat)).squeeze(-1)  # [E]
        
        # Softmax over neighborhood nodes
        alpha = torch.zeros_like(e)
        # Compute scatter softmax per target node
        for i in range(h.size(0)):
            mask = (col == i)
            if mask.any():
                alpha[mask] = F.softmax(e[mask], dim=0)

        alpha = self.dropout(alpha)
        
        # Aggregate neighbor representations weighted by attention coefficients
        h_prime = torch.zeros_like(h)
        for i in range(edge_index.size(1)):
            src = row[i]
            tgt = col[i]
            h_prime[tgt] += alpha[i] * h[src]
            
        return h_prime, alpha


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Conditions molecular node representations on GTEx organ transcriptomic profiles v_tissue in R^128.
    
    Equations:
      gamma = MLP_gamma(v_tissue) in R^128
      beta  = MLP_beta(v_tissue) in R^128
      h_i^conditioned = gamma * h_i + beta
    """
    def __init__(self, tissue_dim: int = 128, feature_dim: int = 128):
        super(FiLMLayer, self).__init__()
        self.mlp_gamma = nn.Sequential(
            nn.Linear(tissue_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.mlp_beta = nn.Sequential(
            nn.Linear(tissue_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, h: torch.Tensor, v_tissue: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Unconditioned node embeddings [N, 128]
            v_tissue: Tissue expression profiles per graph [B, 128]
            batch_index: Node-to-graph mapping [N]
        Returns:
            h_conditioned: Modulated node embeddings [N, 128]
        """
        gamma = self.mlp_gamma(v_tissue)  # [B, 128]
        beta = self.mlp_beta(v_tissue)    # [B, 128]
        
        # Expand gamma and beta across node batch indices
        gamma_nodes = gamma[batch_index]  # [N, 128]
        beta_nodes = beta[batch_index]    # [N, 128]
        
        return gamma_nodes * h + beta_nodes


class EpiADRNet(nn.Module):
    """
    EpiADR-Net Neural Architecture:
      - 4-Layer GAT Backbone (128-dim node embeddings)
      - FiLM Conditioning with 128-dim GTEx Tissue Vector
      - Dual Graph Pooling [Mean || Max] (256-dim fused feature)
      - 10-Class Multi-Label Classification Head
    """
    def __init__(self, in_features: int = 5, hidden_dim: int = 128, num_classes: int = 10, dropout: float = 0.2):
        super(EpiADRNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Input projection layer
        self.input_layer = nn.Linear(in_features, hidden_dim)
        
        # 4-Layer GAT Backbone
        self.gat1 = GATLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.gat3 = GATLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.gat4 = GATLayer(hidden_dim, hidden_dim, dropout=dropout)
        
        # FiLM Conditioning Module
        self.film = FiLMLayer(tissue_dim=128, feature_dim=hidden_dim)
        
        # Dropout layer (retained active during MC Dropout inference)
        self.mc_dropout = nn.Dropout(p=dropout)
        
        # Multi-Label Classification Head (256 -> 128 -> 10)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        tissue_vec: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward Pass of EpiADR-Net.
        Returns logits [B, 10] and optional Layer 4 attention weights for XAI.
        """
        h = F.relu(self.input_layer(x))
        
        h, _ = self.gat1(h, edge_index)
        h = F.relu(h)
        
        h, _ = self.gat2(h, edge_index)
        h = F.relu(h)
        
        h, _ = self.gat3(h, edge_index)
        h = F.relu(h)
        
        h, alpha4 = self.gat4(h, edge_index)
        h = F.relu(h)
        
        # Apply FiLM Conditioning on GAT Layer 4 embeddings
        h_conditioned = self.film(h, tissue_vec, batch)
        h_conditioned = self.mc_dropout(h_conditioned)
        
        # Dual Graph Pooling [Mean || Max]
        num_graphs = tissue_vec.size(0)
        mean_pool = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
        max_pool = torch.full((num_graphs, self.hidden_dim), -1e9, device=x.device)
        
        for g_idx in range(num_graphs):
            mask = (batch == g_idx)
            if mask.any():
                graph_nodes = h_conditioned[mask]
                mean_pool[g_idx] = graph_nodes.mean(dim=0)
                max_pool[g_idx] = graph_nodes.max(dim=0)[0]
            else:
                max_pool[g_idx] = 0.0
                
        fused_graph_emb = torch.cat([mean_pool, max_pool], dim=1)  # [B, 256]
        logits = self.classifier(fused_graph_emb)                   # [B, 10]
        
        if return_attention:
            return logits, alpha4
        return logits, None

    def predict_mc_dropout(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        tissue_vec: torch.Tensor,
        num_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Bayesian Monte Carlo Dropout Uncertainty Quantification.
        Executes N stochastic forward passes with active dropout during inference.
        Returns expected probabilities mu, epistemic uncertainty sigma, and Layer 4 GAT attention weights.
        """
        self.train()  # Keep dropout active for stochastic Monte Carlo sampling
        
        pass_preds = []
        last_attn = None
        
        with torch.no_grad():
            for _ in range(num_samples):
                logits, attn = self.forward(x, edge_index, batch, tissue_vec, return_attention=True)
                probs = torch.sigmoid(logits)
                pass_preds.append(probs)
                last_attn = attn
                
        stacked_preds = torch.stack(pass_preds, dim=0)  # [N, B, 10]
        
        mu = stacked_preds.mean(dim=0)  # Mean predicted probability [B, 10]
        sigma = stacked_preds.std(dim=0) # Epistemic uncertainty [B, 10]
        
        return {
            "mean_probabilities": mu,
            "uncertainty_sigma": sigma,
            "attention_weights": last_attn
        }
