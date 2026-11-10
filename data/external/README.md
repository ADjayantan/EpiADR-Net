# External data cache

Files here are fetched from public sources and cached so the experiment scripts
can be rerun without hitting the network. The derived tables that the models
actually consume — `data/drug_target_matrix.csv` and
`data/gene_tissue_expression.csv` — are committed; the bulk downloads are not.

| file | committed | source | refetch |
|:--|:--|:--|:--|
| `smiles_to_cid.json` | yes (132 KB) | PubChem PUG REST, InChIKey → CID | `make targets` |
| `gene_active_cids.json` | no (4.9 MB) | PubChem BioAssay, active CIDs per panel gene | `make targets` |
| `rna_tissue_consensus.tsv.zip` | no (5.1 MB) | Human Protein Atlas consensus tissue RNA | see below |

```bash
curl -L -o data/external/rna_tissue_consensus.tsv.zip \
  https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip
make expression
```

## Licences

* **PubChem** — public domain (NCBI). <https://pubchem.ncbi.nlm.nih.gov>
* **Human Protein Atlas** — CC BY-SA 4.0. Please cite the HPA when reusing.
  <https://www.proteinatlas.org/about/licence>

`gene_active_cids.json` is excluded from the repository for size only; it is
fully reproducible from the gene panel in `epiadr/gene_panel.py` and is
regenerated automatically by `scripts/fetch_targets.py` when absent.
