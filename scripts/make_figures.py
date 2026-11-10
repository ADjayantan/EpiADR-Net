"""Render the results figures from results/*.json. Run after the experiment scripts."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

INK = "#1b1b1f"
GRID = "#d9d9de"
C_CHANCE = "#9aa0a6"
C_MOL = "#6b7fd7"
C_ONEHOT = "#e0953b"
C_TISSUE = "#2f9e6e"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})


def fig_baselines():
    with open("results/baselines.json", encoding="utf-8") as fh:
        d = json.load(fh)
    res = d["results"]
    labels = {
        "B0_pooled_prevalence": "pooled base rate",
        "B1_organ_prevalence": "per-organ base rate",
        "B2_morgan_logreg_per_organ": "Morgan + LogReg\n(per organ)",
        "B3_morgan_xgboost_per_organ": "Morgan + XGBoost\n(per organ)",
        "B4_morgan_onehot_shared_xgb": "Morgan + organ one-hot\n(shared)",
        "B5_morgan_tissue_shared_xgb": "Morgan + GTEx tissue\n(shared)",
        "B6_morgan_tissue_SHUFFLED_xgb": "Morgan + SHUFFLED tissue\n(ablation)",
    }
    keys = list(labels)
    macro = [res[k]["aggregate"]["macro_auroc"]["mean"] for k in keys]
    macro_e = [res[k]["aggregate"]["macro_auroc"]["std"] for k in keys]
    pooled = [res[k]["aggregate"]["pooled_auroc"]["mean"] for k in keys]
    pooled_e = [res[k]["aggregate"]["pooled_auroc"]["std"] for k in keys]

    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    h = 0.38
    ax.barh(y + h / 2, macro, h, xerr=macro_e, color=C_TISSUE, alpha=.9,
            error_kw=dict(ecolor=INK, lw=.9, capsize=2.5), label="per-organ (macro) AUROC")
    ax.barh(y - h / 2, pooled, h, xerr=pooled_e, color=C_CHANCE, alpha=.85,
            error_kw=dict(ecolor=INK, lw=.9, capsize=2.5), label="pooled AUROC")
    ax.axvline(0.5, color=INK, ls="--", lw=1)
    ax.text(0.502, len(keys) - .35, "chance", fontsize=7.5, color=INK, va="top")
    ax.set_yticks(y, [labels[k] for k in keys], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 0.78)
    ax.set_xlabel("AUROC  (molecule-level Bemis-Murcko scaffold 5-fold CV, mean ± s.d.)")
    ax.set_title("Pooled AUROC rewards organ base rates, not drug ranking",
                 fontsize=10.5, loc="left", weight="bold", pad=10)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    out = f"{FIGDIR}/fig1_baselines.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_zeroshot():
    with open("results/zeroshot.json", encoding="utf-8") as fh:
        d = json.load(fh)
    s = d["settings"]
    settings = [
        ("S1\nunseen drug\nseen organ", "S1_unseen_drug_seen_organ"),
        ("S2\nseen drug\nunseen organ", "S2_seen_drug_unseen_organ"),
        ("S3\nunseen drug\nunseen organ", "S3_unseen_drug_unseen_organ"),
    ]
    modes = [("molecule only", "mol", C_MOL),
             ("+ organ one-hot", "onehot", C_ONEHOT),
             ("+ GTEx tissue profile", "tissue", C_TISSUE)]

    x = np.arange(len(settings))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, (lab, key, col) in enumerate(modes):
        vals = [s[sk][key]["macro_auroc"]["mean"] for _, sk in settings]
        errs = [s[sk][key]["macro_auroc"]["std"] for _, sk in settings]
        pos = x + (i - 1) * w
        ax.bar(pos, vals, w, yerr=errs, color=col, alpha=.92, label=lab,
               error_kw=dict(ecolor=INK, lw=.9, capsize=3))
        for p, v, e in zip(pos, vals, errs, strict=False):
            ax.text(p, v + e + .008, f"{v:.3f}", ha="center", fontsize=7.3, color=INK)

    ax.axhline(0.5, color=INK, ls="--", lw=1)
    ax.text(len(settings) - .45, 0.504, "chance", fontsize=7.5, color=INK)
    ax.set_xticks(x, [lab for lab, _ in settings], fontsize=8.3)
    ax.set_ylim(0.48, 0.79)
    ax.set_ylabel("per-organ (macro) AUROC")
    ax.set_title("Tissue conditioning does not beat an organ indicator in any setting",
                 fontsize=10.5, loc="left", weight="bold", pad=10)
    ax.grid(axis="y", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout()
    out = f"{FIGDIR}/fig2_zeroshot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_per_organ():
    with open("results/baselines.json", encoding="utf-8") as fh:
        d = json.load(fh)
    agg = d["results"]["B3_morgan_xgboost_per_organ"]["aggregate"]["per_organ"]
    prev = d["config"]["prevalence_by_organ"]
    organs = list(prev)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(organs))
    auroc = [agg[o]["auroc"]["mean"] for o in organs]
    err = [agg[o]["auroc"]["std"] for o in organs]
    ax.bar(x, auroc, 0.52, yerr=err, color=C_TISSUE, alpha=.9,
           error_kw=dict(ecolor=INK, lw=.9, capsize=3))
    ax.axhline(0.5, color=INK, ls="--", lw=1)
    for xi, (a, o) in enumerate(zip(auroc, organs, strict=False)):
        ax.text(xi, a + .012, f"{a:.3f}", ha="center", fontsize=7.6)
        ax.text(xi, 0.513, f"{prev[o]*100:.1f}% pos", ha="center", fontsize=6.9,
                color="white", weight="bold")
    adr_short = ["Hepatotox.", "Arrhythmia", "Seizure", "Nephrotox.", "Pneumonitis"]
    ax.set_xticks(x, [f"{o}\n{d}" for o, d in zip(organs, adr_short, strict=False)],
                  fontsize=8)
    ax.set_ylim(0.48, 0.78)
    ax.set_ylabel("AUROC")
    ax.set_title("Best honest model, per organ  —  Morgan + XGBoost, scaffold CV",
                 fontsize=10.5, loc="left", weight="bold", pad=10)
    ax.grid(axis="y", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = f"{FIGDIR}/fig3_per_organ.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_mechanism():
    with open("results/mechanism.json", encoding="utf-8") as fh:
        d = json.load(fh)
    r = d["results"]
    order = [
        ("M0_morgan_only", "Morgan only", C_MOL),
        ("M1_morgan_onehot", "+ organ one-hot", C_ONEHOT),
        ("M2_morgan_gtex_pca", "+ GTEx PCA profile", C_CHANCE),
        ("M4_morgan_binding_onehot", "+ drug-target binding", C_TISSUE),
        ("M8_morgan_binding_onehot_MECHANISM", "+ mechanism coupling", "#1f7a55"),
    ]
    vals = [r[k]["aggregate"]["macro_auroc"]["mean"] for k, _, _ in order]
    errs = [r[k]["aggregate"]["macro_auroc"]["std"] for k, _, _ in order]
    cols = [c for _, _, c in order]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(order))
    ax.bar(x, vals, 0.6, yerr=errs, color=cols, alpha=.93,
           error_kw=dict(ecolor=INK, lw=.9, capsize=3))
    for xi, (v, e) in enumerate(zip(vals, errs, strict=True)):
        ax.text(xi, v + e + .004, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0.5, color=INK, ls="--", lw=1)
    ax.text(len(order) - .45, 0.504, "chance", fontsize=7.5, color=INK)
    ax.set_xticks(x, [lab for _, lab, _ in order], fontsize=8.2)
    ax.set_ylim(0.48, 0.70)
    ax.set_ylabel("per-organ (macro) AUROC")
    ax.set_title("Drug-target data helps; the gain is not tissue-specific",
                 fontsize=10.5, loc="left", weight="bold", pad=10)
    ax.grid(axis="y", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = f"{FIGDIR}/fig4_mechanism.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_permutation():
    with open("results/ablation_test.json", encoding="utf-8") as fh:
        d = json.load(fh)
    null = np.asarray(d["null_macro_auroc"])
    real = d["real_macro_auroc"]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.scatter(null, np.zeros_like(null) + 0.02 * np.arange(len(null)) - 0.1 * 0,
               s=46, color=C_CHANCE, alpha=.85, zorder=3,
               label=f"deranged organ→expression (n={len(null)})")
    ax.axvline(real, color="#c2402f", lw=2.2, zorder=4,
               label=f"real assignment ({real:.4f})")
    ax.axvline(float(null.mean()), color=INK, ls="--", lw=1.1, zorder=2,
               label=f"null mean ({null.mean():.4f})")
    ax.set_yticks([])
    ax.set_xlabel("per-organ (macro) AUROC, M8 model")
    ax.set_title(
        f"Permutation test: the real organ→expression map is not special "
        f"(p = {d['p_value']:.2f})",
        fontsize=10.5, loc="left", weight="bold", pad=10)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.tight_layout()
    out = f"{FIGDIR}/fig5_permutation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    figs = [fig_baselines(), fig_zeroshot(), fig_per_organ()]
    if os.path.exists("results/mechanism.json"):
        figs.append(fig_mechanism())
    if os.path.exists("results/ablation_test.json"):
        figs.append(fig_permutation())
    for f in figs:
        print("wrote", f)
