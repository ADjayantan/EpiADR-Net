"""
Fetch drug-target interactions from PubChem for the pre-specified gene panel.

Two PubChem calls, both cached to data/external/ so reruns are free:

  1. InChIKey -> CID   (batched; InChIKeys are computed locally with RDKit)
  2. gene symbol -> CIDs active in any assay against that target  (one per gene)

Output: data/drug_target_matrix.csv -- rows are the task's molecules, columns are
panel genes, values are 1 when PubChem records that compound as ACTIVE in at
least one assay whose target is that gene.

"Active" here is PubChem's own assay outcome. It pools curated pharmacology with
high-throughput screens, so it is noisier than a binding database; the honest
reading is "has measured activity against", not "binds with affinity X".

Run:  python scripts/fetch_targets.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from epiadr.gene_panel import PANEL  # noqa: E402
from epiadr.task import load_task  # noqa: E402

EXT = "data/external"
CID_CACHE = f"{EXT}/smiles_to_cid.json"
GENE_CACHE = f"{EXT}/gene_active_cids.json"
OUT = "data/drug_target_matrix.csv"

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
BATCH = 100
PAUSE = 0.25          # PubChem asks for <=5 requests/second
RETRIES = 4


def _post(url: str, fields: dict) -> bytes:
    body = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    return _call(req)


def _call(req_or_url, timeout: int = 90) -> bytes:
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req_or_url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""                      # no records for this target
            last = e
            time.sleep(2 ** attempt)
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after {RETRIES} attempts: {last}")


# ── 1. molecules -> PubChem CIDs ────────────────────────────────────────────
def resolve_cids(molecules: list[str]) -> dict[str, int]:
    cache: dict[str, int] = {}
    if os.path.exists(CID_CACHE):
        with open(CID_CACHE, encoding="utf-8") as fh:
            cache = json.load(fh)

    todo = [s for s in molecules if s not in cache]
    if not todo:
        print(f"  CIDs: {len(cache)} cached, nothing to fetch")
        return cache

    key_to_smiles: dict[str, str] = {}
    for smi in todo:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            key_to_smiles.setdefault(Chem.MolToInchiKey(mol), smi)
        except Exception:
            continue

    keys = list(key_to_smiles)
    print(f"  CIDs: {len(cache)} cached, resolving {len(keys)} InChIKeys "
          f"in {(len(keys) + BATCH - 1) // BATCH} batches")

    for i in range(0, len(keys), BATCH):
        chunk = keys[i:i + BATCH]
        raw = _post(f"{PUBCHEM}/compound/inchikey/property/InChIKey/JSON",
                    {"inchikey": ",".join(chunk)})
        if raw:
            for prop in json.loads(raw).get("PropertyTable", {}).get("Properties", []):
                smi = key_to_smiles.get(prop.get("InChIKey", ""))
                if smi and prop.get("CID"):
                    cache[smi] = int(prop["CID"])
        print(f"    batch {i // BATCH + 1}: {len(cache)} resolved", flush=True)
        time.sleep(PAUSE)

    os.makedirs(EXT, exist_ok=True)
    with open(CID_CACHE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
    return cache


# ── 2. gene -> CIDs with measured activity against it ───────────────────────
def fetch_gene_actives(genes: list[str]) -> dict[str, list[int]]:
    cache: dict[str, list[int]] = {}
    if os.path.exists(GENE_CACHE):
        with open(GENE_CACHE, encoding="utf-8") as fh:
            cache = json.load(fh)

    todo = [g for g in genes if g not in cache]
    print(f"  targets: {len(cache)} cached, fetching {len(todo)} genes")

    for n, gene in enumerate(todo, 1):
        url = (f"{PUBCHEM}/assay/target/genesymbol/{urllib.parse.quote(gene)}"
               f"/cids/JSON?cids_type=active")
        raw = _call(url)
        cids: set[int] = set()
        if raw:
            for entry in json.loads(raw).get("InformationList", {}).get("Information", []):
                cids.update(int(c) for c in entry.get("CID", []))
        cache[gene] = sorted(cids)
        print(f"    [{n}/{len(todo)}] {gene}: {len(cids)} active compounds", flush=True)
        time.sleep(PAUSE)

        if n % 10 == 0 or n == len(todo):
            with open(GENE_CACHE, "w", encoding="utf-8") as fh:
                json.dump(cache, fh)

    with open(GENE_CACHE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
    return cache


def main():
    t0 = time.time()
    os.makedirs(EXT, exist_ok=True)
    task = load_task("data/adr_records.csv")
    molecules = task.molecules
    print(f"molecules: {len(molecules)}   panel genes: {len(PANEL)}")

    smiles_to_cid = resolve_cids(molecules)
    resolved = [s for s in molecules if s in smiles_to_cid]
    print(f"  -> {len(resolved)}/{len(molecules)} molecules have a PubChem CID "
          f"({100 * len(resolved) / len(molecules):.1f}%)")

    gene_actives = fetch_gene_actives(PANEL)

    # ── build the matrix ────────────────────────────────────────────────
    active_sets = {g: set(c) for g, c in gene_actives.items()}
    rows, n_hits = [], 0
    for smi in molecules:
        cid = smiles_to_cid.get(smi)
        hits = [1 if (cid is not None and cid in active_sets.get(g, ())) else 0
                for g in PANEL]
        n_hits += sum(hits)
        rows.append((smi, cid if cid is not None else "", hits))

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("smiles,cid," + ",".join(PANEL) + "\n")
        for smi, cid, hits in rows:
            fh.write(f'"{smi}",{cid},' + ",".join(map(str, hits)) + "\n")

    per_mol = n_hits / max(len(molecules), 1)
    with_any = sum(1 for _, _, h in rows if any(h))
    print(f"\nwrote {OUT}")
    print(f"  {n_hits} drug-target hits, {per_mol:.2f} per molecule")
    print(f"  {with_any}/{len(molecules)} molecules hit at least one panel gene "
          f"({100 * with_any / len(molecules):.1f}%)")
    cover = sorted(((sum(h[i] for _, _, h in rows), g) for i, g in enumerate(PANEL)),
                   reverse=True)
    print("  most-hit genes:", ", ".join(f"{g} ({n})" for n, g in cover[:10]))
    print(f"  dead genes (0 hits): {sum(1 for n, _ in cover if n == 0)}/{len(PANEL)}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
