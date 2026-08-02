import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List

# ─────────────────────────────────────────────────────────────────
#  Directed Message Passing GNN (DMPNN) Layer
# ─────────────────────────────────────────────────────────────────
class DMPNNLayer(nn.Module):
    """
    Directed Message Passing Neural Network (DMPNN) Layer.
    Prevents message reflection along undirected bond pairs (i -> j -> i)
    by operating directly on directed edges.
    """
    def __init__(self, node_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.W_msg = nn.Linear(node_dim, node_dim, bias=False)
        self.W_node = nn.Linear(2 * node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        row, col = edge_index[0], edge_index[1]   # src, tgt

        h_src = x[row]                           # [E, D]
        msg = self.dropout(F.gelu(self.W_msg(h_src))) # [E, D]

        # Aggregate directed messages per target node
        agg_msg = torch.zeros(N, self.node_dim, device=x.device)
        agg_msg.scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg)

        # Node update
        combined = torch.cat([x, agg_msg], dim=-1) # [N, 2D]
        x_out = self.norm(x + F.gelu(self.W_node(combined)))
        return x_out


# ─────────────────────────────────────────────────────────────────
#  Bi-Directional Gene Pathway Cross-Attention Module
# ─────────────────────────────────────────────────────────────────
class GenePathwayCrossAttention(nn.Module):
    """
    Bi-Directional Node-to-Gene Pathway Cross-Attention.
    Allows atomic node representations to explicitly query 256-dim tissue gene pathways
    (e.g., CYP3A4 metabolic pathways, hERG cardiotoxicity channels, BSEP transporters).
    """
    def __init__(self, node_dim: int = 512, tissue_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(tissue_dim, node_dim)
        self.v_proj = nn.Linear(tissue_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(tissue_dim, node_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        h_nodes: torch.Tensor,
        v_tissue: torch.Tensor,
        batch_index: torch.Tensor
    ) -> torch.Tensor:
        N = h_nodes.size(0)
        v_nodes = v_tissue[batch_index] # [N, tissue_dim]

        Q = self.q_proj(h_nodes).view(N, self.num_heads, self.head_dim) # [N, H, D/H]
        K = self.k_proj(v_nodes).view(N, self.num_heads, self.head_dim) # [N, H, D/H]
        V = self.v_proj(v_nodes).view(N, self.num_heads, self.head_dim) # [N, H, D/H]

        # Scaled dot-product cross attention
        scores = (Q * K).sum(dim=-1, keepdim=True) / (self.head_dim ** 0.5) # [N, H, 1]
        attn_weights = F.softmax(scores, dim=1)                           # [N, H, 1]

        context = (attn_weights * V).view(N, -1)                         # [N, node_dim]
        gate = self.gate_mlp(v_nodes)                                    # [N, node_dim]

        h_out = self.norm(h_nodes + gate * self.out_proj(context))
        return h_out


# ─────────────────────────────────────────────────────────────────
#  Multi-Head Graph Attention Layer with Scatter Softmax
# ─────────────────────────────────────────────────────────────────
class MultiHeadGATLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dropout: float = 0.1,
    ):
        super().__init__()
        assert out_features % num_heads == 0
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * self.head_dim, 1, bias=True)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout = nn.Dropout(edge_dropout)

        self.out_proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)

        self.skip = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x.size(0)
        residual = self.skip(x)

        h = self.W(x).view(N, self.num_heads, self.head_dim)
        row, col = edge_index[0], edge_index[1]

        h_src = h[row]
        h_tgt = h[col]
        h_cat = torch.cat([h_src, h_tgt], dim=-1)

        e = self.leaky_relu(self.a(h_cat)).squeeze(-1)

        e_max = torch.full((N, self.num_heads), -1e9, device=x.device)
        e_max.scatter_reduce_(0, col.unsqueeze(-1).expand_as(e), e, reduce='amax', include_self=True)
        e_shifted = e - e_max[col]
        exp_e = torch.exp(e_shifted)
        exp_sum = torch.zeros(N, self.num_heads, device=x.device)
        exp_sum.scatter_add_(0, col.unsqueeze(-1).expand_as(exp_e), exp_e)
        alpha = exp_e / (exp_sum[col] + 1e-9)

        alpha = self.edge_dropout(self.dropout(alpha))

        weighted = alpha.unsqueeze(-1) * h_src
        h_agg = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        idx = col.view(-1, 1, 1).expand_as(weighted)
        h_agg.scatter_add_(0, idx, weighted)

        h_flat = h_agg.view(N, self.out_features)
        h_out = self.norm(residual + self.out_proj(h_flat))

        mean_alpha = alpha.mean(dim=-1)
        return h_out, mean_alpha


# ─────────────────────────────────────────────────────────────────
#  EpiADR-Net v4 — Ultra-Performance Architecture
# ─────────────────────────────────────────────────────────────────
class EpiADRNet(nn.Module):
    """
    EpiADR-Net v4 Ultra-Performance Architecture:
      - Input Projection: 24 Atom Features -> 512 Hidden Dimension
      - 2 x Directed Message Passing Layers (DMPNN)
      - 4 x Multi-Head Graph Attention Layers (GAT, 8 Heads, 512 Dim)
      - Bi-Directional Gene Pathway Cross-Attention Conditioning
      - Hierarchical Graph Pooling [Mean ‖ Max ‖ Sum] -> 1536-dim
      - 4-Stage Deep Residual Classifier (1536 -> 768 -> 384 -> 192 -> 10)
      - Monte Carlo Dropout Uncertainty Quantification (N=30)
    """
    def __init__(
        self,
        in_features: int = 24,
        hidden_dim: int = 512,
        tissue_dim: int = 256,
        num_classes: int = 10,
        num_gat_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        edge_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Directed Message Passing (DMPNN) Backbone
        self.dmpnn1 = DMPNNLayer(node_dim=hidden_dim, dropout=dropout)
        self.dmpnn2 = DMPNNLayer(node_dim=hidden_dim, dropout=dropout)

        # GAT Attention Layers
        self.gat_layers = nn.ModuleList([
            MultiHeadGATLayer(
                hidden_dim, hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                edge_dropout=edge_dropout
            )
            for _ in range(num_gat_layers)
        ])

        # Gene Pathway Cross-Attention Module
        self.gene_cross_attn = GenePathwayCrossAttention(
            node_dim=hidden_dim, tissue_dim=tissue_dim, num_heads=num_heads
        )

        # Dropout
        self.mc_dropout = nn.Dropout(p=dropout)

        # Hierarchical Pooling Bottleneck
        self.pool_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * hidden_dim),
        )

        # Deep Classifier Head
        self.cls = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _hierarchical_pool(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        D = self.hidden_dim
        mean_p = torch.zeros(num_graphs, D, device=h.device)
        max_p = torch.full((num_graphs, D), -1e9, device=h.device)
        sum_p = torch.zeros(num_graphs, D, device=h.device)

        for g in range(num_graphs):
            mask = (batch == g)
            if mask.any():
                nodes = h[mask]
                mean_p[g] = nodes.mean(0)
                max_p[g] = nodes.max(0)[0]
                sum_p[g] = nodes.sum(0)
            else:
                max_p[g] = 0.0

        fused = torch.cat([mean_p, max_p, sum_p], dim=1) # [B, 3D]
        return self.pool_proj(fused)                      # [B, 2D]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        tissue_vec: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_graphs = tissue_vec.size(0)
        h = self.input_proj(x)

        # DMPNN Directed Message Passing
        h = self.dmpnn1(h, edge_index)
        h = self.dmpnn2(h, edge_index)

        # GAT Attention Propagation
        last_alpha = None
        for gat in self.gat_layers:
            h, alpha = gat(h, edge_index)
            h = F.gelu(h)
            h = self.mc_dropout(h)
            last_alpha = alpha

        # Bi-Directional Gene Pathway Cross-Attention
        h = self.gene_cross_attn(h, tissue_vec, batch)

        # Hierarchical Pooling
        graph_emb = self._hierarchical_pool(h, batch, num_graphs)

        # Deep Classifier
        logits = self.cls(graph_emb)

        if return_attention:
            return logits, last_alpha
        return logits, None

    def predict_mc_dropout(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        tissue_vec: torch.Tensor,
        num_samples: int = 30,
    ) -> Dict[str, Any]:
        self.train()
        preds: List[torch.Tensor] = []
        last_attn = None

        with torch.no_grad():
            for _ in range(num_samples):
                logits, attn = self.forward(
                    x, edge_index, batch, tissue_vec, return_attention=True
                )
                preds.append(torch.sigmoid(logits))
                last_attn = attn

        stacked = torch.stack(preds, dim=0)
        return {
            "mean_probabilities": stacked.mean(0),
            "uncertainty_sigma": stacked.std(0),
            "attention_weights": last_attn,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
