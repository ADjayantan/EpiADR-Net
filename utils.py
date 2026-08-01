import re
import random
from typing import List, Tuple, Dict, Any
import torch
import numpy as np

ATOM_NUMBERS = {'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'H': 1}
ATOM_MASSES = {'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'Br': 79.904, 'I': 126.90, 'H': 1.008}

def smiles_to_graph(smiles: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Converts a SMILES string into molecular graph node features X (N x 10) and edge index E (2 x M).
    
    Node Features per Atom x_i in R^10:
      1. Atomic Number
      2. Atom Degree (# explicit bonds)
      3. Formal Charge
      4. Hybridization Code (1=sp, 2=sp2, 3=sp3, 4=sp3d, 5=sp3d2, 0=other)
      5. Is Aromatic (1.0 or 0.0)
      6. Implicit Valence
      7. Is In Ring (1.0 or 0.0)
      8. Atomic Mass / 100.0 (normalized)
      9. Is Stereocenter / Chiral Flag (1.0 or 0.0)
      10. Total Hydrogen Count
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            atoms = []
            atom_symbols = []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                num = atom.GetAtomicNum()
                deg = atom.GetDegree()
                chg = atom.GetFormalCharge()
                hyb_val = int(atom.GetHybridization())
                hyb = 1 if hyb_val == 2 else (2 if hyb_val == 3 else (3 if hyb_val == 4 else (4 if hyb_val == 5 else 0)))
                aromatic = 1.0 if atom.GetIsAromatic() else 0.0
                imp_val = float(atom.GetImplicitValence())
                in_ring = 1.0 if atom.IsInRing() else 0.0
                mass = float(atom.GetMass()) / 100.0
                chiral = 1.0 if atom.HasProp('_ChiralityPossible') or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else 0.0
                h_count = float(atom.GetTotalNumHs())
                
                atoms.append([
                    float(num), float(deg), float(chg), float(hyb), aromatic,
                    imp_val, in_ring, mass, chiral, h_count
                ])
                atom_symbols.append(symbol)
            
            edges = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edges.append([i, j])
                edges.append([j, i])
            
            if len(edges) == 0:
                edges = [[0, 0]]
                
            node_feats = torch.tensor(atoms, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            return node_feats, edge_index, atom_symbols
    except Exception:
        pass

    # Regex fallback parser providing 10 feature values per atom
    tokens = re.findall(r'Cl|Br|[A-Z][a-z]?|[a-z]|[\=\#\-\+\(\)]', smiles)
    atoms = []
    atom_symbols = []
    stack = []
    edges = []
    prev_idx = None

    for tok in tokens:
        if tok in ATOM_NUMBERS or tok.upper() in ATOM_NUMBERS:
            symbol = tok.upper()
            num = ATOM_NUMBERS.get(symbol, 6)
            is_aromatic = 1.0 if tok.islower() else 0.0
            mass = ATOM_MASSES.get(symbol, 12.0) / 100.0
            idx = len(atoms)
            
            atoms.append([
                float(num),
                2.0 if is_aromatic else 1.0,
                0.0,
                2.0 if is_aromatic else 3.0,
                is_aromatic,
                0.0,
                is_aromatic,
                mass,
                0.0,
                1.0
            ])
            atom_symbols.append(symbol)

            if prev_idx is not None:
                edges.append([prev_idx, idx])
                edges.append([idx, prev_idx])
            prev_idx = idx
        elif tok == '(':
            if prev_idx is not None:
                stack.append(prev_idx)
        elif tok == ')':
            if stack:
                prev_idx = stack.pop()

    if not atoms:
        atoms = [[6.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.12, 0.0, 1.0]]
        atom_symbols = ['C']
        edges = [[0, 0]]

    if not edges:
        edges = [[0, 0]]

    node_feats = torch.tensor(atoms, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return node_feats, edge_index, atom_symbols


def get_bemis_murcko_scaffold(smiles: str) -> str:
    """
    Extracts the Bemis-Murcko scaffold from a SMILES string.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            return scaffold
    except Exception:
        pass

    rings = re.findall(r'c1[a-z0-9\=\#\-]+1|C1[A-Za-z0-9\=\#\-]+1', smiles)
    if rings:
        return "-".join(sorted(rings))
    c_count = smiles.upper().count('C')
    return f"Framework_C{c_count}"


def bemis_murcko_scaffold_split(
    dataset: List[Any],
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Bemis-Murcko scaffold split ensuring zero structural overlap.
    """
    scaffolds = {}
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_bemis_murcko_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
    
    rng = random.Random(seed)
    rng.shuffle(scaffold_sets)

    total_len = len(dataset)
    train_cutoff = int(frac_train * total_len)
    val_cutoff = int((frac_train + frac_val) * total_len)

    train_idx, val_idx, test_idx = [], [], []

    for cluster in scaffold_sets:
        if len(train_idx) + len(cluster) <= train_cutoff:
            train_idx.extend(cluster)
        elif len(train_idx) + len(val_idx) + len(cluster) <= val_cutoff:
            val_idx.extend(cluster)
        else:
            test_idx.extend(cluster)

    return train_idx, val_idx, test_idx


def random_split(
    dataset: List[Any],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Random dataset split baseline.
    """
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    total_len = len(dataset)
    train_cutoff = int(frac_train * total_len)
    val_cutoff = int((frac_train + frac_val) * total_len)

    train_idx = indices[:train_cutoff]
    val_idx = indices[train_cutoff:val_cutoff]
    test_idx = indices[val_cutoff:]

    return train_idx, val_idx, test_idx


def highlight_toxic_subgraph(smiles: str, attention_scores: torch.Tensor, top_k: int = 3) -> Dict[str, Any]:
    """
    Extracts top-k toxic atomic subgraphs based on Layer 4 GAT attention weights.
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
            
    max_score = np.max(scores) if np.max(scores) > 0 else 1.0
    norm_scores = scores / max_score

    top_indices = np.argsort(norm_scores)[::-1][:min(top_k, num_nodes)].tolist()
    
    toxic_subgraphs = [
        {
            "atom_index": int(idx),
            "atom_symbol": atom_symbols[idx],
            "attention_score": float(round(norm_scores[idx], 4)),
            "is_toxic_hotspot": True
        }
        for idx in top_indices
    ]

    return {
        "smiles": smiles,
        "total_atoms": num_nodes,
        "atom_symbols": atom_symbols,
        "attention_weights": [float(round(s, 4)) for s in norm_scores],
        "top_toxic_hotspots": toxic_subgraphs
    }
