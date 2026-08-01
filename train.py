"""
EpiADR-Net v3 — Advanced Training Pipeline
===========================================

Upgrades over v2:
  ✓ AsymmetricFocalLoss (handles severe class imbalance better than standard Focal)
  ✓ Warmup + Cosine Annealing LR schedule (OneCycleLR)
  ✓ Gradient clipping (max_norm=1.0)
  ✓ Per-class positive-weight calculation
  ✓ Early stopping on validation AUROC (patience=15)
  ✓ Checkpoint save on best validation score
  ✓ 100 epochs with batch_size=32
  ✓ Weight decay schedule with AdamW (lr=5e-4, wd=1e-4)
  ✓ Stochastic Weight Averaging (SWA) for last 20% of epochs
  ✓ Comprehensive per-class metric reporting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
import numpy as np
import os
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")

from data_loader import EpiADRDataset, custom_collate_fn, MEDDRA_ADR_CLASSES
from utils import bemis_murcko_scaffold_split, random_split
from model import EpiADRNet


# ─────────────────────────────────────────────────────────────────
#  Asymmetric Focal Loss
# ─────────────────────────────────────────────────────────────────
class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi-label classification.

    Decouples the focusing parameter for positives (γ+) and negatives (γ-):
        L_+ = α * (1 - p)^γ+ * log(p)          for y=1
        L_- = (1-α) * p^γ- * log(1 - p)        for y=0

    By setting γ- > γ+ we heavily down-weight easy negatives while
    preserving the signal from positive (rare ADR) examples.

    Reference: Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
    """
    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma_pos  = gamma_pos
        self.gamma_neg  = gamma_neg
        self.clip       = clip
        self.reduction  = reduction
        self.pos_weight = pos_weight   # [num_classes]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs     = torch.sigmoid(logits)
        probs_neg = torch.clamp(1.0 - probs, min=self.clip)   # probability shift

        # Focal weights
        focal_pos = (1.0 - probs)      ** self.gamma_pos
        focal_neg = (1.0 - probs_neg)  ** self.gamma_neg      # = probs^γ-

        loss_pos  = -targets      * focal_pos * torch.log(probs.clamp(1e-8))
        loss_neg  = -(1 - targets) * focal_neg * torch.log(probs_neg.clamp(1e-8))

        # Optional per-class positive weighting
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device).unsqueeze(0)  # [1, C]
            loss_pos = loss_pos * pw

        loss = loss_pos + loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────
#  Label Smoothing BCE wrapper
# ─────────────────────────────────────────────────────────────────
class LabelSmoothingBCE(nn.Module):
    """Applies label smoothing to binary targets before BCE."""
    def __init__(self, smoothing: float = 0.05, base_loss: nn.Module = None):
        super().__init__()
        self.smoothing  = smoothing
        self.base_loss  = base_loss or nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.base_loss(logits, targets_smooth)


# ─────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Computes:
      - Macro-AUROC  (across all classes with both labels present)
      - Micro-AUPRC  (across all instances)
      - Per-class AUROC dictionary
    """
    auroc_per_class: Dict[str, float] = {}
    auroc_list: List[float] = []

    for c, name in enumerate(MEDDRA_ADR_CLASSES):
        col = y_true[:, c]
        if len(np.unique(col)) > 1:
            score = roc_auc_score(col, y_pred[:, c])
            auroc_per_class[name] = round(score, 4)
            auroc_list.append(score)

    macro_auroc = float(np.mean(auroc_list)) if auroc_list else 0.5

    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    micro_auprc = float(auc(recall, precision))

    return macro_auroc, micro_auprc, auroc_per_class


# ─────────────────────────────────────────────────────────────────
#  Compute per-class positive weights for imbalance correction
# ─────────────────────────────────────────────────────────────────
def compute_pos_weights(dataset) -> torch.Tensor:
    """
    Computes pos_weight = (# negatives) / (# positives) per class,
    following PyTorch BCEWithLogitsLoss convention.
    """
    all_targets = torch.stack([s["target"] for s in dataset.samples])  # [N, 10]
    pos_counts  = all_targets.sum(0).clamp(min=1)
    neg_counts  = (len(dataset) - all_targets.sum(0)).clamp(min=1)
    return (neg_counts / pos_counts).clamp(max=10.0)   # cap at 10x


# ─────────────────────────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────────────────────────
def train_experiment(
    split_type: str = "scaffold",
    epochs:     int = 100,
    batch_size: int = 32,
    lr:         float = 5e-4,
    weight_decay: float = 1e-4,
    patience:   int = 15,
    save_path:  str = "model.pt",
) -> Dict[str, float]:
    """
    Full high-capacity training run for EpiADR-Net v3.

    Training Strategy:
      - AsymmetricFocalLoss (γ+ = 1.0, γ- = 4.0) + Label Smoothing (ε = 0.05)
      - AdamW with OneCycleLR scheduler (warmup + cosine decay)
      - Gradient clipping (max_norm = 1.0) for stable training
      - Early stopping on val AUROC with patience = 15 epochs
      - Stochastic Weight Averaging (SWA) over last 20 epochs
      - Best checkpoint saved to `save_path`
    """
    # ── Dataset ─────────────────────────────────────────────────
    dataset   = EpiADRDataset(repeat=10)
    smiles_ls = [s["smiles"] for s in dataset.samples]

    if split_type == "scaffold":
        train_idx, val_idx, test_idx = bemis_murcko_scaffold_split(dataset, smiles_ls)
    else:
        train_idx, val_idx, test_idx = random_split(dataset)

    train_sub = torch.utils.data.Subset(dataset, train_idx)
    val_sub   = torch.utils.data.Subset(dataset, val_idx)
    test_sub  = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,
        collate_fn=custom_collate_fn, drop_last=True,
    )
    val_loader  = DataLoader(val_sub,  batch_size=batch_size, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_sub, batch_size=batch_size, collate_fn=custom_collate_fn)

    print(f"\n  Dataset  : {len(dataset)} samples  "
          f"(train={len(train_sub)}, val={len(val_sub)}, test={len(test_sub)})")

    # ── Model ───────────────────────────────────────────────────
    model = EpiADRNet(
        in_features   = 17,
        hidden_dim    = 512,
        tissue_dim    = 256,
        num_classes   = 10,
        num_gat_layers= 6,
        num_heads     = 8,
        dropout       = 0.3,
        edge_dropout  = 0.1,
    )
    n_params = model.count_parameters()
    print(f"  Model    : EpiADR-Net v3  |  {n_params:,} trainable parameters")

    # ── Loss ────────────────────────────────────────────────────
    pos_weight = compute_pos_weights(dataset)
    base_afl   = AsymmetricFocalLoss(
        gamma_pos=1.0, gamma_neg=4.0, clip=0.05, pos_weight=pos_weight
    )
    criterion  = LabelSmoothingBCE(smoothing=0.05, base_loss=base_afl)

    # ── Optimiser ───────────────────────────────────────────────
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )

    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr     = lr,
        epochs     = epochs,
        steps_per_epoch = steps_per_epoch,
        pct_start  = 0.10,          # 10% warmup
        anneal_strategy = "cos",
        div_factor = 25.0,
        final_div_factor = 1e4,
    )

    # ── SWA ─────────────────────────────────────────────────────
    swa_model  = torch.optim.swa_utils.AveragedModel(model)
    swa_start  = int(epochs * 0.80)   # Start SWA at 80% of training
    swa_scheduler = torch.optim.swa_utils.SWALR(
        optimizer, swa_lr=1e-5, anneal_epochs=5
    )
    using_swa  = False

    # ── Training Loop ───────────────────────────────────────────
    print(f"\n{'─'*70}".encode('ascii', 'replace').decode())
    print(f"  EpiADR-Net v3  |  {split_type.upper()} Split  |  {epochs} Epochs")
    print(f"{'─'*70}".encode('ascii', 'replace').decode())

    best_val_auroc = 0.0
    best_state     = None
    patience_count = 0

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model(
                batch["x"], batch["edge_index"],
                batch["batch"], batch["tissue_vec"]
            )
            loss = criterion(logits, batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # SWA phase
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            using_swa = True
        else:
            scheduler.step()

        # ── Validation ──────────────────────────────────────
        eval_model = swa_model if using_swa else model
        eval_model.eval() if not using_swa else model.eval()

        y_true_v, y_pred_v = [], []
        with torch.no_grad():
            for batch in val_loader:
                lgt, _ = (swa_model if using_swa else model)(
                    batch["x"], batch["edge_index"],
                    batch["batch"], batch["tissue_vec"]
                )
                y_true_v.append(batch["y"].numpy())
                y_pred_v.append(torch.sigmoid(lgt).numpy())

        if y_true_v:
            y_tv = np.vstack(y_true_v)
            y_pv = np.vstack(y_pred_v)
            val_auroc, val_auprc, _ = calculate_metrics(y_tv, y_pv)
        else:
            val_auroc, val_auprc = 0.5, 0.0

        # ── Early stopping ───────────────────────────────────
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state     = {
                k: v.clone() for k, v in (
                    swa_model.module if using_swa else model
                ).state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        current_lr = optimizer.param_groups[0]["lr"]
        swa_tag    = "  [SWA]" if using_swa else ""

        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            print(
                f"  Epoch [{epoch:03d}/{epochs}]{swa_tag:<6} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val-AUROC: {val_auroc:.4f} | "
                f"Val-AUPRC: {val_auprc:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Best: {best_val_auroc:.4f}"
            )

        if patience_count >= patience and epoch > swa_start:
            print(f"\n  ⚡ Early stopping at epoch {epoch}  (patience={patience})")
            break

    # ── Restore best weights + update BN stats for SWA ─────
    if best_state is not None:
        (swa_model.module if using_swa else model).load_state_dict(best_state)

    # Update BN statistics for SWA averaged model (simple version)
    if using_swa:
        model.train()
        with torch.no_grad():
            for batch in list(
                DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                           collate_fn=custom_collate_fn)
            )[:10]:
                swa_model(
                    batch["x"], batch["edge_index"],
                    batch["batch"], batch["tissue_vec"]
                )

    # ── Test Evaluation ─────────────────────────────────────────
    final_model = swa_model if using_swa else model
    final_model.eval()

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            lgt, _ = (swa_model if using_swa else model)(
                batch["x"], batch["edge_index"],
                batch["batch"], batch["tissue_vec"]
            )
            y_true_list.append(batch["y"].numpy())
            y_pred_list.append(torch.sigmoid(lgt).numpy())

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    macro_auroc, micro_auprc, per_class = calculate_metrics(y_true, y_pred)

    # Save checkpoint
    if split_type == "scaffold":
        save_model = swa_model.module if using_swa else model
        torch.save(save_model.state_dict(), save_path)
        print(f"\n  ✅ Saved best model → '{save_path}'")

    print(f"\n{'─'*70}".encode('ascii', 'replace').decode())
    print(f"  RESULT [{split_type.upper()}]")
    print(f"    Test Macro-AUROC : {macro_auroc:.4f}")
    print(f"    Test Micro-AUPRC : {micro_auprc:.4f}")
    print(f"\n  Per-Class AUROC:")
    for cls_name, score in per_class.items():
        bar = "█" * int(score * 20)
        print(f"    {cls_name:<30} {score:.4f}  {bar}")
    print(f"{'─'*70}\n")

    return {
        "macro_auroc": macro_auroc,
        "micro_auprc": micro_auprc,
        "per_class":   per_class,
    }


# ─────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("   EpiADR-Net v3  —  High-Capacity Training & Benchmark   ")
    print("   Architecture: 6-Layer GAT | 8-Head | 512-dim | 17 feats")
    print("=" * 70)

    scaffold_res = train_experiment(
        split_type  = "scaffold",
        epochs      = 50,
        batch_size  = 32,
        lr          = 5e-4,
        weight_decay= 1e-4,
        patience    = 15,
        save_path   = "model.pt",
    )

    random_res = train_experiment(
        split_type  = "random",
        epochs      = 50,
        batch_size  = 32,
        lr          = 5e-4,
        weight_decay= 1e-4,
        patience    = 15,
        save_path   = "model_random.pt",
    )

    print("\n" + "=" * 70)
    print(f"{'Experiment Split':<30} | {'Test Macro-AUROC':<16} | {'Test Micro-AUPRC':<16}")
    print("-" * 70)
    print(f"{'Bemis-Murcko Scaffold Split':<30} | {scaffold_res['macro_auroc']:<16.4f} | {scaffold_res['micro_auprc']:<16.4f}")
    print(f"{'Random Split':<30} | {random_res['macro_auroc']:<16.4f} | {random_res['micro_auprc']:<16.4f}")
    print("=" * 70)
