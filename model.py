from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


# ─────────────────────────────────────────────────────────────────
#  SwiGLU Feed-Forward Expansion Block
#  SwiGLU(x) = (Swish(x W_gate) * (x W_up)) W_down
# ─────────────────────────────────────────────────────────────────
class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int = 1536, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(x) = x * sigmoid(x) = silu(x)
        gate = F.silu(self.w_gate(x))
        up   = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


# ─────────────────────────────────────────────────────────────────
#  Directed Message Passing GNN (DMPNN) Layer
# ─────────────────────────────────────────────────────────────────
class DMPNNLayer(nn.Module):
    def __init__(self, node_dim: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.W_msg = nn.Linear(node_dim, node_dim, bias=False)
        self.W_node = nn.Linear(2 * node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        row, col = edge_index[0], edge_index[1]

        h_src = x[row]
        msg = self.dropout(F.gelu(self.W_msg(h_src)))

        agg_msg = torch.zeros(N, self.node_dim, device=x.device)
        agg_msg.scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg)

        combined = torch.cat([x, agg_msg], dim=-1)
        x_out = self.norm(x + F.gelu(self.W_node(combined)))
        return x_out


# ─────────────────────────────────────────────────────────────────
#  16-Head Bi-Directional Gene Pathway Cross-Attention
# ─────────────────────────────────────────────────────────────────
class GenePathwayCrossAttention100M(nn.Module):
    """
    16-Head Multi-Head Cross-Attention Layer.
    Bridges 1536-dim molecular node tokens directly with 1024-dim GTEx organ gene pathways.
    """
    def __init__(self, node_dim: int = 1536, tissue_dim: int = 1024, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

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

        Q = self.q_proj(h_nodes).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(v_nodes).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(v_nodes).view(N, self.num_heads, self.head_dim)

        scores = (Q * K).sum(dim=-1, keepdim=True) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=1)

        context = (attn_weights * V).view(N, -1)
        gate = self.gate_mlp(v_nodes)

        h_out = self.norm(h_nodes + gate * self.out_proj(context))
        return h_out


# ─────────────────────────────────────────────────────────────────
#  Graph Transformer Layer (Multi-Head Self-Attention + SwiGLU FFN)
# ─────────────────────────────────────────────────────────────────
class GraphTransformerBlock(nn.Module):
    def __init__(
        self,
        in_features: int = 1536,
        out_features: int = 1536,
        num_heads: int = 16,
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
        self.norm1 = nn.LayerNorm(out_features)

        # SwiGLU FFN Layer
        self.ffn = SwiGLUFFN(dim=out_features, expansion_factor=4, dropout=dropout)
        self.norm2 = nn.LayerNorm(out_features)

        self.skip = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        h_attn = self.norm1(residual + self.out_proj(h_flat))

        # SwiGLU FFN Pass
        h_out = self.norm2(h_attn + self.ffn(h_attn))

        mean_alpha = alpha.mean(dim=-1)
        return h_out, mean_alpha


# ─────────────────────────────────────────────────────────────────
#  EpiADR-Net v5 — 100M+ Parameter Foundation Architecture
# ─────────────────────────────────────────────────────────────────
class EpiADRNet(nn.Module):
    """
    EpiADR-Net v6 Foundation Edition:
      - Input Projection: 24 Atom Descriptors -> 1536 Hidden Dimension
      - 4 x DMPNN Directed Message Passing Layers (d_edge = 1536)
      - 4 x Shallow Graph Transformer Blocks (16 Attention Heads, SwiGLU FFN 1536->6144->1536)
      - 16-Head Gene Pathway Cross-Attention (1536 x 1024 GTEx Transcriptomics)
      - Hierarchical Graph Pooling [Mean ‖ Max] -> 3072-dim
      - 4-Stage Deep Residual Classifier Head (3072 -> 1536 -> 768 -> 384 -> 10)
      - Monte Carlo Dropout Uncertainty Quantification (N=30)
    """
    def __init__(
        self,
        in_features: int = 24,
        hidden_dim: int = 1536,
        tissue_dim: int = 1024,
        num_classes: int = 10,
        num_gat_layers: int = 4,
        num_heads: int = 16,
        dropout: float = 0.1,
        edge_dropout: float = 0.05,
        use_tissue_conditioning: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_tissue_conditioning = use_tissue_conditioning

        # Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 4 Directed Message Passing (DMPNN) Backbone Layers
        self.dmpnn1 = DMPNNLayer(node_dim=hidden_dim, dropout=dropout)
        self.dmpnn2 = DMPNNLayer(node_dim=hidden_dim, dropout=dropout)
        self.dmpnn3 = DMPNNLayer(node_dim=hidden_dim, dropout=dropout)
        self.dmpnn4 = DMPNNLayer(node_dim=hidden_dim, dropout=dropout)

        # 12 Deep Graph Transformer Blocks (with SwiGLU FFN)
        self.gat_layers = nn.ModuleList([
            GraphTransformerBlock(
                hidden_dim, hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                edge_dropout=edge_dropout
            )
            for _ in range(num_gat_layers)
        ])

        # 16-Head Bi-Directional Gene Pathway Cross-Attention Module
        self.gene_cross_attn = GenePathwayCrossAttention100M(
            node_dim=hidden_dim, tissue_dim=tissue_dim, num_heads=num_heads
        )

        # Dropout
        self.mc_dropout = nn.Dropout(p=dropout)

        # Hierarchical Pooling Bottleneck (2 * 1536 = 3072)
        self.pool_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * hidden_dim),
        )

        # Deep Classifier Head (3072 -> 1536 -> 768 -> 384 -> 10)
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
            else:
                max_p[g] = 0.0

        fused = torch.cat([mean_p, max_p], dim=1) # [B, 2D] = [B, 3072]
        return self.pool_proj(fused)                      # [B, 2D] = [B, 3072]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        tissue_vec: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_graphs = tissue_vec.size(0)
        h = self.input_proj(x)

        # 4 DMPNN Directed Message Passing Layers
        h = self.dmpnn1(h, edge_index)
        h = self.dmpnn2(h, edge_index)
        h = self.dmpnn3(h, edge_index)
        h = self.dmpnn4(h, edge_index)

        # 12 Graph Transformer Blocks
        last_alpha = None
        for gat in self.gat_layers:
            h, alpha = gat(h, edge_index)
            h = F.gelu(h)
            h = self.mc_dropout(h)
            last_alpha = alpha

        # Bi-Directional Gene Pathway Cross-Attention (Skipped if use_tissue_conditioning=False)
        if self.use_tissue_conditioning:
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
    ) -> dict[str, Any]:
        self.train()
        preds: list[torch.Tensor] = []
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

    def model_config(self) -> dict[str, Any]:
        return {
            "in_features": 24,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "num_gat_layers": len(self.gat_layers),
            "use_tissue_conditioning": self.use_tissue_conditioning,
            "parameters": self.count_parameters(),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
