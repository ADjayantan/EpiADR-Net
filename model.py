import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class MultiHeadGATLayer(nn.Module):
    """
    Multi-Head Graph Attention Network (GAT) Layer with Attention Weight Extraction.
    Computes parallel attention heads and projects output embeddings.
    """
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.2):
        super(MultiHeadGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [N, in_features]
            edge_index: Edge connectivity [2, E]
        Returns:
            x_out: Updated node features [N, out_features]
            alpha: Mean attention coefficients [E]
        """
        N = x.size(0)
        h = self.W(x).view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        row, col = edge_index[0], edge_index[1]
        
        # Concatenate source and target node representations per head
        h_src = h[row]  # [E, num_heads, head_dim]
        h_tgt = h[col]  # [E, num_heads, head_dim]
        h_cat = torch.cat([h_src, h_tgt], dim=-1)  # [E, num_heads, 2 * head_dim]
        
        e = self.leakyrelu(self.a(h_cat)).squeeze(-1)  # [E, num_heads]
        
        alpha = torch.zeros_like(e)
        for i in range(N):
            mask = (col == i)
            if mask.any():
                alpha[mask] = F.softmax(e[mask], dim=0)

        alpha = self.dropout(alpha)
        
        # Aggregate neighbor representations weighted by attention coefficients
        h_prime = torch.zeros_like(h)
        for i in range(edge_index.size(1)):
            src = row[i]
            tgt = col[i]
            # alpha[i] shape: [num_heads], h[src] shape: [num_heads, head_dim]
            h_prime[tgt] += alpha[i].unsqueeze(-1) * h[src]
            
        h_flat = h_prime.view(N, self.out_features)
        h_out = self.norm(x + self.out_proj(h_flat)) if x.size(-1) == self.out_features else self.norm(self.out_proj(h_flat))
        
        mean_alpha = alpha.mean(dim=-1)  # Average attention over heads for XAI [E]
        return h_out, mean_alpha


class ResidualFiLMLayer(nn.Module):
    """
    High-Capacity Residual Feature-wise Linear Modulation (FiLM) Layer.
    Conditions molecular node representations on GTEx organ transcriptomic profiles v_tissue in R^128.
    """
    def __init__(self, tissue_dim: int = 128, feature_dim: int = 256):
        super(ResidualFiLMLayer, self).__init__()
        self.mlp_gamma = nn.Sequential(
            nn.Linear(tissue_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )
        self.mlp_beta = nn.Sequential(
            nn.Linear(tissue_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, h: torch.Tensor, v_tissue: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """
        Modulates node embeddings: h_conditioned = gamma * h + beta + h (Residual connection)
        """
        gamma = self.mlp_gamma(v_tissue)  # [B, 256]
        beta = self.mlp_beta(v_tissue)    # [B, 256]
        
        gamma_nodes = gamma[batch_index]  # [N, 256]
        beta_nodes = beta[batch_index]    # [N, 256]
        
        return (gamma_nodes * h + beta_nodes) + h


class EpiADRNet(nn.Module):
    """
    Enhanced EpiADR-Net High-Capacity Architecture:
      - 4 Multi-Head GAT Layers (4 Attention Heads, 256-dim node embeddings)
      - Residual FiLM Conditioning with 128-dim GTEx Tissue Vector
      - Dual Graph Pooling [Mean || Max] (512-dim fused feature)
      - 3-Layer Deep Residual Classifier (512 -> 256 -> 128 -> 10)
    """
    def __init__(self, in_features: int = 10, hidden_dim: int = 256, num_classes: int = 10, dropout: float = 0.3):
        super(EpiADRNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Input projection layer
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4 Multi-Head GAT Backbone
        self.gat1 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
        self.gat2 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
        self.gat3 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
        self.gat4 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
        
        # Residual FiLM Module
        self.film = ResidualFiLMLayer(tissue_dim=128, feature_dim=hidden_dim)
        
        # Monte Carlo Dropout Layer
        self.mc_dropout = nn.Dropout(p=dropout)
        
        # Deep Classifier Head (512 -> 256 -> 128 -> 10)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)  # 512 -> 256
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # 256 -> 128
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, num_classes) # 128 -> 10

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        tissue_vec: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass returning logits [B, 10] and optional Layer 4 attention weights.
        """
        h = self.input_layer(x)
        
        h, _ = self.gat1(h, edge_index)
        h = F.gelu(h)
        
        h, _ = self.gat2(h, edge_index)
        h = F.gelu(h)
        
        h, _ = self.gat3(h, edge_index)
        h = F.gelu(h)
        
        h, alpha4 = self.gat4(h, edge_index)
        h = F.gelu(h)
        
        # Apply Residual FiLM Conditioning
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
                
        fused_graph_emb = torch.cat([mean_pool, max_pool], dim=1)  # [B, 512]
        
        # Classifier Pass
        c = F.gelu(self.norm1(self.fc1(fused_graph_emb)))
        c = self.mc_dropout(c)
        c = F.gelu(self.norm2(self.fc2(c)))
        c = self.mc_dropout(c)
        logits = self.fc_out(c)  # [B, 10]
        
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
        """
        self.train()
        
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
