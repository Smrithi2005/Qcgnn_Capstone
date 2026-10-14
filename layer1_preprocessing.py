"""
LAYER 1: G-SchNet QM9 Preprocessing
With detailed CSV logging and statistics
"""

import os
import sys
from pathlib import Path
import numpy as np
import logging
import pickle
import pandas as pd
from tqdm import tqdm

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable

#
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    LAYER1_INPUT, LAYER1_OUTPUT_PKL, LAYER1_OUTPUT_CSV, LAYER1_OUTPUT_STATS,
    ATOMIC_NUMBERS, VDW_RADII, BOND_THRESHOLD_FACTOR,
    NODE_FEATURE_DIM, EDGE_FEATURE_DIM, LOG_FORMAT, LOG_LEVEL
)

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
      logging.StreamHandler(sys.stdout),
      logging.FileHandler("layer1_output.txt", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("LAYER 1: G-SchNet QM9 Preprocessing")
logger.info("=" * 80)


# ==============================================================================
# G-SchNet Preprocessing Class
# ==============================================================================

class GSchNetPreprocessor:


    def __init__(self):
        self.atomic_nums = ATOMIC_NUMBERS
        self.vdw_radii = VDW_RADII
        self.bond_threshold_factor = BOND_THRESHOLD_FACTOR

    @staticmethod
    def read_xyz(filepath):

        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            n_atoms = int(lines[0].strip())
            symbols = []
            positions = []

            for i in range(n_atoms):
                parts = lines[2 + i].split()
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])

            return np.array(symbols), np.array(positions), True

        except Exception as e:
            
            return None, None, False

    def build_graph(self, symbols, positions):

        n_atoms = len(symbols)
        mol = Chem.RWMol()
        atomic_nums = []

        for symbol in symbols:
            atomic_num = self.atomic_nums.get(symbol, 6)
            atomic_nums.append(atomic_num)
            atom = Chem.Atom(atomic_num)
            mol.AddAtom(atom)

        edges = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_i = self.vdw_radii.get(atomic_nums[i], 1.7)
                r_j = self.vdw_radii.get(atomic_nums[j], 1.7)

                distance = np.linalg.norm(positions[i] - positions[j])
                threshold = self.bond_threshold_factor * (r_i + r_j)

                if distance < threshold:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    edges.append((i, j))

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        mol = mol.GetMol()
        return mol, edges, atomic_nums

    @staticmethod
    def extract_node_features(mol, symbols):

        node_features = []

        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)

            hyb = str(atom.GetHybridization())
            if 'SP3' in hyb:
                hyb_type = 3.0
            elif 'SP2' in hyb:
                hyb_type = 2.0
            elif 'SP' in hyb:
                hyb_type = 1.0
            else:
                hyb_type = 0.0

            feat = np.array([
                float(atom.GetAtomicNum()),
                float(atom.GetFormalCharge()),
                hyb_type,
                float(atom.GetDegree()),
                float(atom.GetTotalNumHs()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRingSize(6)),
                float(atom.IsInRingSize(5)),
                float(atom.GetTotalValence()),
            ], dtype=np.float32)

            node_features.append(feat)

        return np.array(node_features)

    @staticmethod
    def extract_edge_features(mol, edges):

        edge_features = []

        for i, j in edges:
            bond = mol.GetBondBetweenAtoms(i, j)

            if bond is None:
                continue

            feat = np.array([
                float(bond.GetBondType()),
                float(bond.GetIsAromatic()),
                float(bond.IsInRing()),
                float(bond.GetIsConjugated()),
            ], dtype=np.float32)

            edge_features.append(feat)

        return np.array(edge_features)

    def process_molecule(self, xyz_file, mol_id):

        try:
            symbols, positions, success = self.read_xyz(xyz_file)

            if not success:
                return None, "Failed to read XYZ file"

            mol, edges, atomic_nums = self.build_graph(symbols, positions)

            node_features = self.extract_node_features(mol, symbols)
            edge_features = self.extract_edge_features(mol, edges)

            edge_index = []
            for i, j in edges:
                edge_index.append([i, j])
                edge_index.append([j, i])

            edge_index = (
                np.array(edge_index, dtype=np.int64)
                if edge_index
                else np.zeros((0, 2), dtype=np.int64)
            )

            return {
                'mol_id':        mol_id,
                'filename':      Path(xyz_file).name,
                'symbols':       list(symbols),
                'positions':     positions.astype(np.float32),
                'atomic_numbers': np.array(atomic_nums, dtype=np.int32),
                'node_features': node_features,    # (n_atoms, 9)
                'edge_index':    edge_index,        # (n_edges*2, 2) bidirectional
                'edge_features': edge_features,     # (n_bonds, 4)
                'n_atoms':       len(symbols),
                'n_bonds':       len(edges),
            }, "Success"

        except Exception as e:
            return None, str(e)




logger.info("\n[STEP 1] Finding XYZ files...")
logger.info(f"Input directory: {LAYER1_INPUT}")

if not LAYER1_INPUT.exists():
    logger.error(f"Input directory does not exist: {LAYER1_INPUT}")
    sys.exit(1)

xyz_files = sorted(LAYER1_INPUT.glob("*.xyz"))
logger.info(f"✓ Found {len(xyz_files)} XYZ files")

if len(xyz_files) == 0:
    logger.error("No XYZ files found. Please check LAYER1_INPUT in config.py")
    sys.exit(1)

if LAYER1_OUTPUT_PKL.exists():
    logger.info(f"✓ Output already exists: {LAYER1_OUTPUT_PKL}")
    logger.info("Skipping preprocessing. Delete the file if you want to reprocess.")
    sys.exit(0)




logger.info("\n[STEP 2] Processing molecules with G-SchNet...")

layer1_data = []
csv_records = []
preprocessor = GSchNetPreprocessor()

for idx, xyz_file in enumerate(tqdm(xyz_files, desc="Processing XYZ files")):
    result, status = preprocessor.process_molecule(str(xyz_file), idx)

    if result is not None:
        layer1_data.append(result)

        csv_records.append({
            'mol_id':        result['mol_id'],
            'filename':      result['filename'],
            'n_atoms':       result['n_atoms'],
            'n_bonds':       result['n_bonds'],
            'status':        'SUCCESS',
            'error_message': '',
            'atom_types':    ','.join(result['symbols']),
        })
    else:
        csv_records.append({
            'mol_id':        idx,
            'filename':      Path(xyz_file).name,
            'n_atoms':       0,
            'n_bonds':       0,
            'status':        'FAILED',
            'error_message': status,
            'atom_types':    '',
        })

logger.info(f"\n✓ Processed successfully : {len(layer1_data)}")
logger.info(f"✗ Failed                 : {len(xyz_files) - len(layer1_data)}")




logger.info("\n[STEP 3] Saving Layer 1 output...")

LAYER1_OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)

with open(LAYER1_OUTPUT_PKL, 'wb') as f:
    pickle.dump(layer1_data, f)
logger.info(f"✓ Saved pickle    : {LAYER1_OUTPUT_PKL}  "
            f"({LAYER1_OUTPUT_PKL.stat().st_size / 1e6:.2f} MB)")

csv_df = pd.DataFrame(csv_records)
csv_df.to_csv(LAYER1_OUTPUT_CSV, index=False)
logger.info(f"✓ Saved details   : {LAYER1_OUTPUT_CSV}")




logger.info("\n[STEP 4] Computing statistics...")

success_df   = csv_df[csv_df['status'] == 'SUCCESS'].copy()
n_atoms_list = success_df['n_atoms'].tolist()
n_bonds_list = success_df['n_bonds'].tolist()

stats = {
    'Total Molecules Processed': len(xyz_files),
    'Successfully Processed':    len(layer1_data),
    'Failed':                    len(xyz_files) - len(layer1_data),
    'Success Rate (%)':          round(100 * len(layer1_data) / len(xyz_files), 2) if xyz_files else 0,
    'Avg Atoms per Molecule':    round(float(np.mean(n_atoms_list)), 2) if n_atoms_list else 0,
    'Avg Bonds per Molecule':    round(float(np.mean(n_bonds_list)), 2) if n_bonds_list else 0,
    'Min Atoms':                 int(np.min(n_atoms_list))  if n_atoms_list else 0,
    'Max Atoms':                 int(np.max(n_atoms_list))  if n_atoms_list else 0,
    'Min Bonds':                 int(np.min(n_bonds_list))  if n_bonds_list else 0,
    'Max Bonds':                 int(np.max(n_bonds_list))  if n_bonds_list else 0,
    'Node Feature Dim':          NODE_FEATURE_DIM,
    'Edge Feature Dim':          EDGE_FEATURE_DIM,
}

stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
stats_df.to_csv(LAYER1_OUTPUT_STATS)
logger.info(f"✓ Saved statistics: {LAYER1_OUTPUT_STATS}")




logger.info("\n" + "=" * 80)
logger.info("LAYER 1 PREPROCESSING COMPLETE!")
logger.info("=" * 80)

logger.info("\n[STATISTICS]")
logger.info(f"  Total molecules         : {len(xyz_files)}")
logger.info(f"  Successfully processed  : {len(layer1_data)}")
logger.info(f"  Failed                  : {len(xyz_files) - len(layer1_data)}")
logger.info(f"  Success rate            : {stats['Success Rate (%)']:.1f}%")

logger.info("\n[MOLECULE PROPERTIES]")
logger.info(f"  Avg atoms per molecule  : {stats['Avg Atoms per Molecule']:.1f}")
logger.info(f"  Avg bonds per molecule  : {stats['Avg Bonds per Molecule']:.1f}")
logger.info(f"  Atom count range        : {stats['Min Atoms']} – {stats['Max Atoms']}")
logger.info(f"  Bond count range        : {stats['Min Bonds']} – {stats['Max Bonds']}")

logger.info("\n[FEATURE DIMENSIONS]")
logger.info(f"  Node features (per atom): {NODE_FEATURE_DIM}D")
logger.info(f"  Edge features (per bond): {EDGE_FEATURE_DIM}D")

logger.info("\n[OUTPUT FILES]")
logger.info(f"  Pickle data  : {LAYER1_OUTPUT_PKL}")
logger.info(f"  Details CSV  : {LAYER1_OUTPUT_CSV}")
logger.info(f"  Statistics   : {LAYER1_OUTPUT_STATS}")

logger.info("\n[NEXT STEP]")
logger.info("  Run Layer 2:  python code/layer2_qcgnn/qcgnn_preprocessing.py")
logger.info("\n" + "=" * 80)