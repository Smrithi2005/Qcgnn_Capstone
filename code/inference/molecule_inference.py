import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

# Import constants from config
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import ATOMIC_NUMBERS, VDW_RADII, BOND_THRESHOLD_FACTOR

def smiles_to_qm9_graph(smiles):
    """
    Safely converts a SMILES string to a PyTorch Geometric Data object
    matching exactly the QM9 training representation from Layer 2.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
        
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=42)
    if res != 0:
        raise ValueError("3D conformer generation failed in RDKit.")
        
    AllChem.MMFFOptimizeMolecule(mol)
    
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = mol.GetConformer().GetPositions()
    
    n_atoms = len(symbols)
    atomic_nums = []
    
    # Rebuild molecular connectivity based strictly on 3D vdW radii
    # to perfectly match training distribution logic
    built_mol = Chem.RWMol()
    for symbol in symbols:
        atomic_num = ATOMIC_NUMBERS.get(symbol, 6)
        atomic_nums.append(atomic_num)
        atom = Chem.Atom(atomic_num)
        built_mol.AddAtom(atom)

    edges = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r_i = VDW_RADII.get(atomic_nums[i], 1.7)
            r_j = VDW_RADII.get(atomic_nums[j], 1.7)
            distance = np.linalg.norm(positions[i] - positions[j])
            threshold = BOND_THRESHOLD_FACTOR * (r_i + r_j)
            if distance < threshold:
                built_mol.AddBond(i, j, Chem.BondType.SINGLE)
                edges.append((i, j))
                
    try:
        Chem.SanitizeMol(built_mol)
    except Exception:
        pass
        
    built_mol = built_mol.GetMol()
    
    # Extract Node Features [N, 9]
    node_features = []
    for i in range(built_mol.GetNumAtoms()):
        atom = built_mol.GetAtomWithIdx(i)
        hyb = str(atom.GetHybridization())
        if 'SP3' in hyb: hyb_type = 3.0
        elif 'SP2' in hyb: hyb_type = 2.0
        elif 'SP' in hyb: hyb_type = 1.0
        else: hyb_type = 0.0

        feat = np.array([
            float(atom.GetAtomicNum()), float(atom.GetFormalCharge()), hyb_type,
            float(atom.GetDegree()), float(atom.GetTotalNumHs()), float(atom.GetIsAromatic()),
            float(atom.IsInRingSize(6)), float(atom.IsInRingSize(5)), float(atom.GetTotalValence()),
        ], dtype=np.float32)
        node_features.append(feat)
        
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    # Extract Edge Indices and Features
    edge_index = []
    edge_features = []
    for i, j in edges:
        # Add bidirectional edges
        edge_index.append([i, j])
        edge_index.append([j, i])
        
        bond = built_mol.GetBondBetweenAtoms(i, j)
        feat = [
            float(bond.GetBondType()), float(bond.GetIsAromatic()),
            float(bond.IsInRing()), float(bond.GetIsConjugated())
        ] if bond is not None else [0.0, 0.0, 0.0, 0.0]
        
        edge_features.append(feat)
        edge_features.append(feat)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float)
        
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, atomic_nums
