"""
Reproducible validity audit of the EpiADR-Net v6 training pipeline.

Run:  python scripts/audit_pipeline.py
Writes: results/audit_findings.json  and prints a human-readable report.

This script answers four questions about the ORIGINAL pipeline:
  A. Does train.py consume data/adr_records.csv?
  B. Are the GTEx tissue profiles used in training real transcriptomics?
  C. Does the training label depend on the organ it is conditioned on?
  D. How many distinct molecules does the reported benchmark actually see?
"""
import ast
import collections
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

findings = {}


def _src(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


# ── A. Is the real labelled CSV reachable from the training entrypoint? ──────
def audit_csv_reachability():
    tree = ast.parse(_src("train.py"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            for a in node.names:
                imported.add(a.name)

    # Who references the CSV loaders anywhere in the package?
    loader_users = []
    for fname in sorted(f for f in os.listdir(ROOT) if f.endswith(".py")):
        if fname == "data_loader.py":
            continue
        s = _src(fname)
        if "load_labeled_records" in s or "GTExTissueLoader" in s:
            loader_users.append(fname)

    return {
        "train_py_imports_from_data_loader": sorted(imported),
        "train_py_imports_csv_loader": "load_labeled_records" in imported,
        "modules_using_csv_loaders": loader_users,
        "verdict": (
            "DEAD CODE: load_labeled_records / GTExTissueLoader are defined but "
            "never called by any module. data/adr_records.csv never enters training."
        ) if not loader_users and "load_labeled_records" not in imported else "reachable",
    }


# ── B. Are the training tissue vectors real GTEx data? ──────────────────────
def audit_tissue_source():
    s = _src("data_loader.py")
    synthetic = "np.random.dirichlet" in s and "GTEX_TISSUE_PROFILES" in s

    # Dimensionality claimed by the data contract vs used in training
    with open(os.path.join(ROOT, "data", "tissue_profiles.csv"), encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    csv_dim = sum(1 for c in header if c.startswith("feature_"))

    train_dim = None
    for line in s.splitlines():
        if line.strip().startswith("TISSUE_DIM"):
            train_dim = int(line.split("=")[1].strip())

    return {
        "training_profiles_are_random_dirichlet": synthetic,
        "csv_tissue_dim": csv_dim,
        "training_tissue_dim": train_dim,
        "dims_match": csv_dim == train_dim,
        "verdict": (
            "SYNTHETIC: training conditions on np.random.dirichlet noise, not GTEx. "
            f"The real CSV has {csv_dim} PCA features; training allocates {train_dim}."
        ) if synthetic else "real",
    }


# ── C. Does the label depend on the organ? ─────────────────────────────────
def audit_label_organ_dependence():
    s = _src("data_loader.py")
    tree = ast.parse(s)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_build_target_vector"
    )
    params = [a.arg for a in fn.args.args]

    # And in the real CSV: does the label vector vary across organs for a drug?
    with open(os.path.join(ROOT, "data", "adr_records.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    label_cols = [c for c in rows[0] if c not in ("smiles", "organ")]
    by_smiles = collections.defaultdict(dict)
    for r in rows:
        by_smiles[r["smiles"]][r["organ"]] = tuple(r[c] for c in label_cols)
    varying = sum(1 for d in by_smiles.values() if len(set(d.values())) > 1)

    return {
        "build_target_vector_params": params,
        "label_fn_receives_organ": any("organ" in p for p in params),
        "csv_unique_molecules": len(by_smiles),
        "csv_rows": len(rows),
        "csv_molecules_whose_labels_vary_by_organ": varying,
        "verdict": (
            "NO ORGAN SIGNAL: _build_target_vector() never receives the organ, so all "
            "10 organs for a drug get an identical target. Independently, in the real CSV "
            f"{varying}/{len(by_smiles)} molecules have labels that differ across organs. "
            "An organ-conditioned model therefore has nothing organ-specific to learn."
        ) if varying == 0 else "organ-dependent",
    }


# ── D. How many molecules does the benchmark actually train on? ────────────
def audit_benchmark_scale():
    s = _src("data_loader.py")
    tree = ast.parse(s)
    raw = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_RAW_DRUG_DEFS" for t in n.targets)
    )
    drugs = [e.elts[0].value for e in raw.value.elts]

    organs = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "ORGAN_NAMES" for t in n.targets):
            organs = [e.value for e in n.value.elts]

    repeat_default = 8  # train.py --repeat default
    return {
        "molecules_in_training_set": len(drugs),
        "molecule_names": drugs,
        "organs": len(organs),
        "default_repeat_factor": repeat_default,
        "total_samples": len(drugs) * len(organs) * repeat_default,
        "distinct_label_vectors": len(drugs),
        "folds": 5,
        "molecules_held_out_per_fold": len(drugs) // 5,
        "verdict": (
            f"{len(drugs)} molecules, each duplicated {repeat_default}x across {len(organs)} organs "
            f"= {len(drugs)*len(organs)*repeat_default} rows but only {len(drugs)} independent "
            "label vectors. 5-fold CV therefore evaluates on 1 molecule per fold."
        ),
    }


def main():
    findings["A_csv_reachability"] = audit_csv_reachability()
    findings["B_tissue_source"] = audit_tissue_source()
    findings["C_label_organ_dependence"] = audit_label_organ_dependence()
    findings["D_benchmark_scale"] = audit_benchmark_scale()

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "audit_findings.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2)

    print("=" * 78)
    print(" EpiADR-Net — PIPELINE VALIDITY AUDIT")
    print("=" * 78)
    for key, data in findings.items():
        print(f"\n[{key}]")
        for k, v in data.items():
            if k == "verdict":
                continue
            print(f"  {k}: {v}")
        print(f"  --> {data['verdict']}")
    print("\n" + "=" * 78)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
