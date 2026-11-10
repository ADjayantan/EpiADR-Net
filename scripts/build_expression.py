"""
Extract per-gene, per-organ expression for the pre-specified panel.

Source: Human Protein Atlas consensus tissue RNA (nTPM), which aggregates HPA
and GTEx samples per tissue. It is used here instead of the PCA-reduced
data/tissue_profiles.csv because the mechanism feature needs expression of
SPECIFIC genes, and 128 principal components cannot be inverted back to genes.

Download once:
  curl -L -o data/external/rna_tissue_consensus.tsv.zip \
    https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip

Output: data/gene_tissue_expression.csv with one row per panel gene and columns
  <organ>_ntpm       raw nTPM in that organ
  <organ>_frac       that organ's share of the gene's total expression across
                     the five organs -- the tissue-SELECTIVITY of the target,
                     which is the quantity the mechanistic hypothesis is about

Run:  python scripts/build_expression.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from epiadr.gene_panel import ORGAN_TO_HPA_TISSUE, PANEL  # noqa: E402
from epiadr.task import ORGANS  # noqa: E402

SRC = "data/external/rna_tissue_consensus.tsv.zip"
OUT = "data/gene_tissue_expression.csv"


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC} — see the download command in this file's docstring")

    wanted_genes = set(PANEL)
    tissue_to_organ = {t: o for o, t in ORGAN_TO_HPA_TISSUE.items()}

    expr: dict[str, dict[str, float]] = {}
    with zipfile.ZipFile(SRC) as z, z.open(z.namelist()[0]) as fh:
        reader = csv.DictReader(io.TextIOWrapper(fh, "utf-8", "replace"), delimiter="\t")
        for row in reader:
            gene = row["Gene name"]
            if gene not in wanted_genes:
                continue
            organ = tissue_to_organ.get(row["Tissue"])
            if organ is None:
                continue
            expr.setdefault(gene, {})[organ] = float(row["nTPM"])

    missing = sorted(wanted_genes - set(expr))
    incomplete = sorted(g for g, d in expr.items() if len(d) != len(ORGANS))

    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gene"] + [f"{o}_ntpm" for o in ORGANS] + [f"{o}_frac" for o in ORGANS])
        for gene in PANEL:
            vals = expr.get(gene)
            if not vals or len(vals) != len(ORGANS):
                continue
            ntpm = [vals[o] for o in ORGANS]
            total = sum(ntpm)
            frac = [v / total for v in ntpm] if total > 0 else [0.0] * len(ORGANS)
            w.writerow([gene] + [f"{v:.4f}" for v in ntpm] + [f"{v:.6f}" for v in frac])

    kept = sum(1 for g in PANEL if g in expr and len(expr[g]) == len(ORGANS))
    print(f"wrote {OUT}")
    print(f"  {kept}/{len(PANEL)} panel genes have expression in all {len(ORGANS)} organs")
    if missing:
        print(f"  not found in HPA ({len(missing)}): {', '.join(missing)}")
    if incomplete:
        print(f"  incomplete ({len(incomplete)}): {', '.join(incomplete)}")

    # Sanity: the panel should show the textbook selectivity patterns.
    print("\n  sanity check — organ with the highest share of each gene's expression:")
    for gene in ("KCNH2", "CYP3A4", "ABCB11", "SLC22A6", "GRIN1", "PTGS1"):
        vals = expr.get(gene)
        if not vals:
            continue
        total = sum(vals.values()) or 1.0
        best = max(ORGANS, key=lambda o: vals[o])
        print(f"    {gene:<9} -> {best:<7} ({100 * vals[best] / total:.0f}% of panel-organ total,"
              f" {vals[best]:.1f} nTPM)")


if __name__ == "__main__":
    main()
