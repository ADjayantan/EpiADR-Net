import re
import random
from typing import List, Tuple, Dict, Any
import torch
import numpy as np

# Suppress RDKit C++ stderr (parse warnings flood the terminal)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────
#  Atom element lookup tables
# ─────────────────────────────────────────────────────────────────
ATOM_NUMBERS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53,
    'Si': 14, 'B': 5, 'Se': 34, 'Na': 11, 'K': 19,
}
ATOM_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
    'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'Br': 79.904, 'I': 126.90,
    'Si': 28.085, 'B': 10.811, 'Se': 78.971, 'Na': 22.990, 'K': 39.098,
}

# One-hot encoding helpers (safe — returns zero vector for unknown)
def _one_hot(value, choices):
    """Returns a fixed-length one-hot list; last element = 'other'."""
    vec = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    vec[idx] = 1
    return vec


def smiles_to_graph(smiles: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Converts a SMILES string into a molecular graph with 17-dimensional atom features.

    Node Feature Vector x_i ∈ ℝ^17 per atom:
    ─────────────────────────────────────────────
     [0]    Atomic Number  (normalized / 100)
     [1]    Atomic Mass    (normalized / 100)
     [2]    Atom Degree    (normalized / 6)
     [3]    Formal Charge  (clamped to [-2, +2], normalized / 2)
     [4]    Hybridization code  (sp=1, sp2=2, sp3=3, sp3d=4, sp3d2=5, other=0)
     [5]    Is Aromatic    (binary)
     [6]    Implicit Valence (normalized / 6)
     [7]    Is In Ring     (binary)
     [8]    Normalized Atomic Mass / 100  (duplicate with richer context)
     [9]    Is Stereocenter  (binary)
     [10]   Total Hydrogen Count (normalized / 4)
     [11-14] Element group one-hot: [C/N/O, Halogens, Heteroatoms(P/S/Se/B/Si), Other]
     [15]   Electronegativity proxy  (Pauling scale / 4.0, normalized)
     [16]   Van der Waals radius proxy (normalized / 2.5)

    Bond information is encoded implicitly via the edge_index.

    Returns:
        node_feats : torch.Tensor [N, 17]
        edge_index : torch.Tensor [2, E]
        atom_symbols: List[str]
    """
    # Electronegativity (Pauling) and vdW radius lookup tables
    ELECTRONEGATIVITY = {
        'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
        'Si': 1.90, 'B': 2.04, 'Se': 2.55, 'Na': 0.93, 'K': 0.82,
    }
    VDW_RADIUS = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
        'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
        'Si': 2.10, 'B': 1.92, 'Se': 1.90, 'Na': 2.27, 'K': 2.75,
    }
    HALOGENS = {'F', 'Cl', 'Br', 'I'}
    HETEROATOMS = {'P', 'S', 'Se', 'B', 'Si'}

    def atom_to_features(symbol, num, deg, chg, hyb, aromatic,
                          imp_val, in_ring, mass, chiral, h_count) -> List[float]:
        # Group one-hot: [C/N/O group, Halogen, Heteroatom, Other]
        group = [0.0, 0.0, 0.0, 0.0]
        if symbol in {'C', 'N', 'O'}:
            group[0] = 1.0
        elif symbol in HALOGENS:
            group[1] = 1.0
        elif symbol in HETEROATOMS:
            group[2] = 1.0
        else:
            group[3] = 1.0

        en   = ELECTRONEGATIVITY.get(symbol, 2.0) / 4.0        # normalized [0,1]
        vdw  = VDW_RADIUS.get(symbol, 1.7) / 2.5               # normalized [0,1]

        return [
            float(num)   / 100.0,                              # [0]
            float(mass)  / 100.0,                              # [1]
            min(float(deg), 6.0) / 6.0,                       # [2]
            max(-2.0, min(2.0, float(chg))) / 2.0,            # [3]
            float(hyb)   / 5.0,                               # [4]
            float(aromatic),                                   # [5]
            min(float(imp_val), 6.0) / 6.0,                   # [6]
            float(in_ring),                                    # [7]
            float(mass)  / 100.0,                              # [8] (reused for richer context)
            float(chiral),                                     # [9]
            min(float(h_count), 4.0) / 4.0,                   # [10]
            *group,                                            # [11-14]
            en,                                                # [15]
            vdw,                                               # [16]
        ]

    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)   # ensure aromaticity / valence is set
            atoms        = []
            atom_symbols = []

            for atom in mol.GetAtoms():
                symbol   = atom.GetSymbol()
                num      = atom.GetAtomicNum()
                deg      = atom.GetDegree()
                chg      = atom.GetFormalCharge()
                hyb_val  = int(atom.GetHybridization())
                # Map RDKit hybridization enum values to 1-5 codes
                hyb = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5}.get(hyb_val, 0)
                aromatic = 1.0 if atom.GetIsAromatic() else 0.0

                try:
                    imp_val = float(atom.GetValence(Chem.ValenceType.IMPLICIT))
                except Exception:
                    imp_val = float(atom.GetImplicitValence())

                in_ring = 1.0 if atom.IsInRing() else 0.0
                mass    = float(atom.GetMass())
                chiral  = (
                    1.0 if (
                        atom.HasProp('_ChiralityPossible') or
                        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                    ) else 0.0
                )
                h_count = float(atom.GetTotalNumHs())

                feats = atom_to_features(
                    symbol, num, deg, chg, hyb, aromatic,
                    imp_val, in_ring, mass, chiral, h_count
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

    # ── Regex fallback (no RDKit) ────────────────────────────────
    tokens = re.findall(r'Cl|Br|Si|Se|[A-Z][a-z]?|[a-z]|[\=\#\-\+\(\)]', smiles)
    atoms, atom_symbols, edges = [], [], []
    stack, prev_idx = [], None

    for tok in tokens:
        sym = tok.upper() if tok.isalpha() else tok
        if sym in ATOM_NUMBERS or tok in ATOM_NUMBERS:
            key = sym if sym in ATOM_NUMBERS else tok
            num      = ATOM_NUMBERS.get(key, 6)
            mass     = ATOM_MASSES.get(key, 12.0)
            aromatic = 1.0 if tok.islower() else 0.0
            idx      = len(atoms)

            feats = atom_to_features(
                key, num,
                deg=2 if aromatic else 1,
                chg=0, hyb=2 if aromatic else 3,
                aromatic=aromatic, imp_val=0.0,
                in_ring=aromatic, mass=mass,
                chiral=0.0, h_count=1.0
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
        atoms        = [atom_to_features('C', 6, 1, 0, 3, 0, 0, 0, 12.011, 0, 1)]
        atom_symbols = ['C']
        edges        = [[0, 0]]

    if not edges:
        edges = [[0, 0]]

    node_feats = torch.tensor(atoms, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return node_feats, edge_index, atom_symbols


# ─────────────────────────────────────────────────────────────────
#  Bemis-Murcko Scaffold Split
# ─────────────────────────────────────────────────────────────────
def get_bemis_murcko_scaffold(smiles: str) -> str:
    """Extracts the Bemis-Murcko scaffold from a SMILES string."""
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
    frac_val:   float = 0.1,
    frac_test:  float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Scaffold-based dataset split ensuring zero structural overlap
    between train / val / test sets (Bemis & Murcko, 1996).
    """
    scaffolds: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles_list):
        sc = get_bemis_murcko_scaffold(smi)
        scaffolds.setdefault(sc, []).append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    rng = random.Random(seed)
    rng.shuffle(scaffold_sets)

    total      = len(dataset)
    train_cut  = int(frac_train * total)
    val_cut    = int((frac_train + frac_val) * total)
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
    frac_val:   float = 0.1,
    frac_test:  float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Random IID dataset split (baseline)."""
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    n         = len(indices)
    train_cut = int(frac_train * n)
    val_cut   = int((frac_train + frac_val) * n)
    return indices[:train_cut], indices[train_cut:val_cut], indices[val_cut:]


# ─────────────────────────────────────────────────────────────────
#  XAI — Toxic Hotspot Highlighting
# ─────────────────────────────────────────────────────────────────
def highlight_toxic_subgraph(
    smiles: str,
    attention_scores: torch.Tensor,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Identifies the top-k toxic atomic hotspots from Layer-6 GAT attention weights.
    Used for Explainable AI (XAI) visualisation in the Gradio / Streamlit UI.
    """
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

    max_s      = np.max(scores) if np.max(scores) > 0 else 1.0
    norm       = scores / max_s
    top_idxs   = np.argsort(norm)[::-1][:min(top_k, num_nodes)].tolist()

    hotspots = [
        {
            "atom_index":      int(i),
            "atom_symbol":     atom_symbols[i],
            "attention_score": round(float(norm[i]), 4),
            "is_toxic_hotspot": True,
        }
        for i in top_idxs
    ]

    return {
        "smiles":            smiles,
        "total_atoms":       num_nodes,
        "atom_symbols":      atom_symbols,
        "attention_weights": [round(float(s), 4) for s in norm],
        "top_toxic_hotspots": hotspots,
    }
