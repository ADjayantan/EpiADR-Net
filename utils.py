import re
import random
from typing import List, Tuple, Dict, Any, Optional
import torch
import numpy as np

# Suppress RDKit C++ stderr noise
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────
#  Atom & Bond Lookup Tables
# ─────────────────────────────────────────────────────────────────
ATOM_NUMBERS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53,
    'Si': 14, 'B': 5, 'Se': 34, 'Na': 11, 'K': 19, 'Pt': 78,
}
ATOM_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
    'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'Br': 79.904, 'I': 126.90,
    'Si': 28.085, 'B': 10.811, 'Se': 78.971, 'Na': 22.990, 'K': 39.098, 'Pt': 195.08,
}
ELECTRONEGATIVITY = {
    'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
    'Si': 1.90, 'B': 2.04, 'Se': 2.55, 'Na': 0.93, 'K': 0.82, 'Pt': 2.28,
}
VDW_RADIUS = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
    'Si': 2.10, 'B': 1.92, 'Se': 1.90, 'Na': 2.27, 'K': 2.75, 'Pt': 1.75,
}

HALOGENS = {'F', 'Cl', 'Br', 'I'}
HETEROATOMS = {'P', 'S', 'Se', 'B', 'Si', 'Pt'}

def smiles_to_graph(smiles: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Converts a SMILES string into a rich molecular graph with 24-dimensional atom features x_i in R^24
    and edge_index E in R^(2 x M).

    Node Features x_i in R^24:
      [0]  Atomic Number / 100.0
      [1]  Atomic Mass / 200.0
      [2]  Degree / 6.0
      [3]  Formal Charge clamped [-2, +2] / 2.0
      [4]  Hybridization code (1=sp, 2=sp2, 3=sp3, 4=sp3d, 5=sp3d2, 0=other) / 5.0
      [5]  Is Aromatic (binary)
      [6]  Implicit Valence / 6.0
      [7]  Is In Ring (binary)
      [8]  Is Stereocenter / Chiral Flag (binary)
      [9]  Total Hydrogen Count / 4.0
      [10-13] Element Group One-Hot: [C/N/O, Halogens, Heteroatoms, Other]
      [14] Pauling Electronegativity / 4.0
      [15] vdW Radius / 3.0
      [16] Is Ring Size 3 (binary)
      [17] Is Ring Size 4 (binary)
      [18] Is Ring Size 5 (binary)
      [19] Is Ring Size 6 (binary)
      [20] Is Ring Size 7 (binary)
      [21] Is Ring Size 8 (binary)
      [22] Is Conjugated Atom (binary)
      [23] Gasteiger Charge Proxy (normalized [-1, +1])
    """
    def _atom_to_features(symbol, num, deg, chg, hyb, aromatic,
                           imp_val, in_ring, mass, chiral, h_count,
                           ring_sizes, conjugated, charge_proxy) -> List[float]:
        group = [0.0, 0.0, 0.0, 0.0]
        if symbol in {'C', 'N', 'O'}:
            group[0] = 1.0
        elif symbol in HALOGENS:
            group[1] = 1.0
        elif symbol in HETEROATOMS:
            group[2] = 1.0
        else:
            group[3] = 1.0

        en  = ELECTRONEGATIVITY.get(symbol, 2.0) / 4.0
        vdw = VDW_RADIUS.get(symbol, 1.7) / 3.0

        r3 = 1.0 if 3 in ring_sizes else 0.0
        r4 = 1.0 if 4 in ring_sizes else 0.0
        r5 = 1.0 if 5 in ring_sizes else 0.0
        r6 = 1.0 if 6 in ring_sizes else 0.0
        r7 = 1.0 if 7 in ring_sizes else 0.0
        r8 = 1.0 if 8 in ring_sizes else 0.0

        return [
            float(num) / 100.0,                               # [0]
            float(mass) / 200.0,                              # [1]
            min(float(deg), 6.0) / 6.0,                       # [2]
            max(-2.0, min(2.0, float(chg))) / 2.0,            # [3]
            float(hyb) / 5.0,                                 # [4]
            float(aromatic),                                  # [5]
            min(float(imp_val), 6.0) / 6.0,                   # [6]
            float(in_ring),                                   # [7]
            float(chiral),                                    # [8]
            min(float(h_count), 4.0) / 4.0,                   # [9]
            *group,                                           # [10-13]
            en,                                               # [14]
            vdw,                                              # [15]
            r3, r4, r5, r6, r7, r8,                          # [16-21]
            float(conjugated),                                # [22]
            max(-1.0, min(1.0, float(charge_proxy))),         # [23]
        ]

    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
            atoms, atom_symbols = [], []

            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                num = atom.GetAtomicNum()
                deg = atom.GetDegree()
                chg = atom.GetFormalCharge()
                hyb_val = int(atom.GetHybridization())
                hyb = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5}.get(hyb_val, 0)
                aromatic = 1.0 if atom.GetIsAromatic() else 0.0

                try:
                    imp_val = float(atom.GetValence(Chem.ValenceType.IMPLICIT))
                except Exception:
                    imp_val = float(atom.GetImplicitValence())

                in_ring = 1.0 if atom.IsInRing() else 0.0
                mass = float(atom.GetMass())
                chiral = 1.0 if (atom.HasProp('_ChiralityPossible') or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED) else 0.0
                h_count = float(atom.GetTotalNumHs())

                ring_sizes = [size for size in range(3, 9) if atom.IsInRingSize(size)]
                conjugated = 1.0 if atom.GetIsAromatic() or any(b.GetIsConjugated() for b in atom.GetBonds()) else 0.0
                charge_proxy = float(chg) + (0.1 if symbol in {'N', 'O'} else (-0.1 if symbol in {'C'} else 0.0))

                feats = _atom_to_features(
                    symbol, num, deg, chg, hyb, aromatic,
                    imp_val, in_ring, mass, chiral, h_count,
                    ring_sizes, conjugated, charge_proxy
                )
                atoms.append(feats)
                atom_symbols.append(symbol)

            edges = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edges.extend([[i, j], [j, i]])

            if not edges:
                edges = [[0, 0]]

            node_feats = torch.tensor(atoms, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            return node_feats, edge_index, atom_symbols

    except Exception:
        pass

    # ── Regex Fallback Parser ───────────────────────────────────
    tokens = re.findall(r'Cl|Br|Si|Se|Pt|[A-Z][a-z]?|[a-z]|[\=\#\-\+\(\)]', smiles)
    atoms, atom_symbols, edges = [], [], []
    stack, prev_idx = [], None

    for tok in tokens:
        sym = tok.upper() if tok.isalpha() else tok
        if sym in ATOM_NUMBERS or tok in ATOM_NUMBERS:
            key = sym if sym in ATOM_NUMBERS else tok
            num = ATOM_NUMBERS.get(key, 6)
            mass = ATOM_MASSES.get(key, 12.0)
            aromatic = 1.0 if tok.islower() else 0.0
            idx = len(atoms)

            feats = _atom_to_features(
                key, num, deg=2 if aromatic else 1, chg=0, hyb=2 if aromatic else 3,
                aromatic=aromatic, imp_val=0.0, in_ring=aromatic, mass=mass,
                chiral=0.0, h_count=1.0, ring_sizes=[6] if aromatic else [],
                conjugated=aromatic, charge_proxy=0.0
            )
            atoms.append(feats)
            atom_symbols.append(key)

            if prev_idx is not None:
                edges.extend([[prev_idx, idx], [idx, prev_idx]])
            prev_idx = idx
        elif tok == '(':
            if prev_idx is not None:
                stack.append(prev_idx)
        elif tok == ')':
            if stack:
                prev_idx = stack.pop()

    if not atoms:
        atoms = [_atom_to_features('C', 6, 1, 0, 3, 0, 0, 0, 12.011, 0, 1, [], 0, 0.0)]
        atom_symbols = ['C']
        edges = [[0, 0]]

    if not edges:
        edges = [[0, 0]]

    node_feats = torch.tensor(atoms, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return node_feats, edge_index, atom_symbols


# ─────────────────────────────────────────────────────────────────
#  Bemis-Murcko Scaffold Splitter
# ─────────────────────────────────────────────────────────────────
def get_bemis_murcko_scaffold(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        pass

    rings = re.findall(r'c1[a-z0-9\=\#\-]+1|C1[A-Za-z0-9\=\#\-]+1', smiles)
    if rings:
        return "-".join(sorted(rings))
    c_count = smiles.upper().count('C')
    return f"Framework_C{c_count}"


def bemis_murcko_scaffold_split(
    dataset,
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    scaffolds: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles_list):
        sc = get_bemis_murcko_scaffold(smi)
        scaffolds.setdefault(sc, []).append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    rng = random.Random(seed)
    rng.shuffle(scaffold_sets)

    total = len(dataset)
    train_cut = int(frac_train * total)
    val_cut = int((frac_train + frac_val) * total)
    train_idx, val_idx, test_idx = [], [], []

    for cluster in scaffold_sets:
        if len(train_idx) + len(cluster) <= train_cut:
            train_idx.extend(cluster)
        elif len(train_idx) + len(val_idx) + len(cluster) <= val_cut:
            val_idx.extend(cluster)
        else:
            test_idx.extend(cluster)

    return train_idx, val_idx, test_idx


def random_split(
    dataset,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    n = len(indices)
    train_cut = int(frac_train * n)
    val_cut = int((frac_train + frac_val) * n)
    return indices[:train_cut], indices[train_cut:val_cut], indices[val_cut:]


# ─────────────────────────────────────────────────────────────────
#  XAI — Toxic Hotspot Highlighting
# ─────────────────────────────────────────────────────────────────
def highlight_toxic_subgraph(
    smiles: str,
    attention_scores: torch.Tensor,
    top_k: int = 3,
) -> Dict[str, Any]:
    node_feats, edge_index, atom_symbols = smiles_to_graph(smiles)
    num_nodes = len(atom_symbols)

    if attention_scores is None or len(attention_scores) == 0:
        scores = np.ones(num_nodes) / max(num_nodes, 1)
    else:
        scores = attention_scores.detach().cpu().numpy()
        if len(scores) < num_nodes:
            scores = np.pad(scores, (0, num_nodes - len(scores)), 'constant')
        elif len(scores) > num_nodes:
            scores = scores[:num_nodes]

    max_s = np.max(scores) if np.max(scores) > 0 else 1.0
    norm = scores / max_s
    top_idxs = np.argsort(norm)[::-1][:min(top_k, num_nodes)].tolist()

    hotspots = [
        {
            "atom_index": int(i),
            "atom_symbol": atom_symbols[i],
            "attention_score": round(float(norm[i]), 4),
            "is_toxic_hotspot": True,
        }
        for i in top_idxs
    ]

    return {
        "smiles": smiles,
        "total_atoms": num_nodes,
        "atom_symbols": atom_symbols,
        "attention_weights": [round(float(s), 4) for s in norm],
        "top_toxic_hotspots": hotspots,
    }
