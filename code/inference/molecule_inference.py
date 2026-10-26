import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from torch_geometric.data import Data

# Import constants from config
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import ATOMIC_NUMBERS


def _supported_atoms(mol):
    supported_atomic_numbers = set(ATOMIC_NUMBERS.values())
    unsupported = sorted({
        atom.GetAtomicNum() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in supported_atomic_numbers
    })
    if unsupported:
        symbols = ', '.join(Chem.GetPeriodicTable().GetElementSymbol(number) for number in unsupported)
        raise ValueError(
            f"Unsupported element(s): {symbols}. Supported elements are H, C, N, O, F, P, S, Cl, Br, and I."
        )


def _node_features(mol):
    features = []
    for atom in mol.GetAtoms():
        hybridization = str(atom.GetHybridization())
        if 'SP3' in hybridization:
            hybridization_value = 3.0
        elif 'SP2' in hybridization:
            hybridization_value = 2.0
        elif 'SP' in hybridization:
            hybridization_value = 1.0
        else:
            hybridization_value = 0.0
        features.append([
            float(atom.GetAtomicNum()),
            float(atom.GetFormalCharge()),
            hybridization_value,
            float(atom.GetDegree()),
            float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRingSize(6)),
            float(atom.IsInRingSize(5)),
            float(atom.GetTotalValence()),
        ])
    return np.asarray(features, dtype=np.float32)


def _bond_features(mol):
    edges = []
    edge_features = []
    display_bonds = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feature = [
            float(bond.GetBondType()),
            float(bond.GetIsAromatic()),
            float(bond.IsInRing()),
            float(bond.GetIsConjugated()),
        ]
        edges.extend(((start, end), (end, start)))
        edge_features.extend((feature, feature))
        display_bonds.append({
            'source': start,
            'target': end,
            'bond_type': float(bond.GetBondType()),
            'is_aromatic': bool(bond.GetIsAromatic()),
            'is_in_ring': bool(bond.IsInRing()),
            'stereo': str(bond.GetStereo()),
        })
    return edges, edge_features, display_bonds

def _display_atoms(mol):
    atoms = []
    for atom in mol.GetAtoms():
        isotope = atom.GetIsotope()
        charge = atom.GetFormalCharge()
        label = atom.GetSymbol()
        if isotope:
            label = f"{isotope}{label}"
        if charge:
            sign = '+' if charge > 0 else '-'
            magnitude = abs(charge)
            label += sign if magnitude == 1 else f"{magnitude}{sign}"
        atoms.append({
            'id': atom.GetIdx(),
            'atomic_num': atom.GetAtomicNum(),
            'symbol': atom.GetSymbol(),
            'label': label,
            'formal_charge': charge,
            'isotope': isotope,
            'is_aromatic': bool(atom.GetIsAromatic()),
            'chiral_tag': str(atom.GetChiralTag()),
            'explicit_hydrogens': atom.GetNumExplicitHs(),
            'implicit_hydrogens': atom.GetNumImplicitHs(),
        })
    return atoms


def smiles_to_qm9_graph(smiles):
    """
    Safely converts a SMILES string to a PyTorch Geometric Data object
    matching exactly the QM9 training representation from Layer 2.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES input cannot be empty.")

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    _supported_atoms(mol)
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=42)
    if res != 0:
        raise ValueError("3D conformer generation failed in RDKit.")
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        pass

    # The model uses explicit hydrogens to match the QM9 atom representation.
    # The 2D conformer is kept separately for chemistry-aware display only.
    display_mol = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(display_mol)
    display_positions = display_mol.GetConformer().GetPositions()[:, :2]
    positions = mol.GetConformer().GetPositions()
    atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edges, edge_features, display_bonds = _bond_features(mol)
    display_heavy_atoms = [atom for atom in _display_atoms(mol) if atom['atomic_num'] != 1]
    display_heavy_ids = {atom['id'] for atom in display_heavy_atoms}
    display_heavy_bonds = [
        bond for bond in display_bonds
        if bond['source'] in display_heavy_ids and bond['target'] in display_heavy_ids
    ]
    display_heavy_positions = [
        display_positions[atom['id']].tolist() for atom in display_heavy_atoms
    ]

    x = torch.tensor(_node_features(mol), dtype=torch.float)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float)
        
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(positions, dtype=torch.float),
        display_pos=torch.tensor(display_positions, dtype=torch.float),
        display_bonds=display_bonds,
            display_atoms=_display_atoms(mol),
            display_heavy_atoms=display_heavy_atoms,
            display_heavy_bonds=display_heavy_bonds,
            display_heavy_pos=display_heavy_positions,
        smiles=smiles.strip(),
    )
    return data, atomic_nums
