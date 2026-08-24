"""
LAYER 2 (FIXED): QCGNN Preprocessing
======================================
Extracts QM9 target properties DIRECTLY from your XYZ files.
No PyG QM9 download needed — your XYZ files already have the targets on line 2.

QM9 XYZ line 2 format (17 values):
    tag  A  B  C  mu  alpha  homo  lumo  gap  r2  zpve  U0  U  H  G  Cv
    idx: 0  1  2  3   4      5     6     7    8   9     10  11 12 13 14 15

We want: homo=index 6, lumo=index 7, gap=index 8
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
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    LAYER1_OUTPUT_PKL,
    LAYER2_OUTPUT_PKL,
    LAYER2_OUTPUT_TRAIN,
    LAYER2_OUTPUT_VAL,
    LAYER2_OUTPUT_TEST,
    LAYER2_OUTPUT_CSV,
    LAYER2_OUTPUT_STATS,
    QM9_TARGET_INDICES,
    QM9_TARGET_NAMES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    MAX_SAMPLES,
    RANDOM_SEED,
    RAW_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("layer2_output.txt", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("LAYER 2 (FIXED): QCGNN Preprocessing - Direct XYZ Target Extraction")
logger.info("=" * 80)


# ==============================================================================
# QM9 Property positions in line 2 of each XYZ file
# ==============================================================================

QM9_XYZ_PROPERTY_INDEX = {
    'mu':    4,
    'alpha': 5,
    'homo':  6,
    'lumo':  7,
    'gap':   8,
    'r2':    9,
    'zpve':  10,
    'u0':    11,
    'u298':  12,
    'h298':  13,
    'g298':  14,
    'cv':    15,
}


# ==============================================================================
# Target Extractor
# ==============================================================================

def extract_targets_from_xyz(xyz_dir: Path, target_names: list) -> dict:
    """
    Read line 2 of every XYZ file and extract scalar target values.
    QM9 quirk: replaces '*^' with 'e' for scientific notation parsing.

    Returns:
        Dict: filename -> np.array([homo, lumo, gap]) or None on failure
    """
    xyz_files = sorted(xyz_dir.glob("*.xyz"))
    logger.info(f"  Found {len(xyz_files):,} XYZ files in {xyz_dir}")

    prop_positions = [QM9_XYZ_PROPERTY_INDEX[name] for name in target_names]
    logger.info(f"  Extracting positions {prop_positions} -> {target_names}")

    targets = {}
    failed  = 0

    for xyz_file in tqdm(xyz_files, desc="Reading XYZ targets"):
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()

            # Line 2 (index 1) — replace QM9 scientific notation quirk
            prop_line = lines[1].replace('*^', 'e').strip()
            parts     = prop_line.split()

            values = np.array(
                [float(parts[i]) for i in prop_positions],
                dtype=np.float32
            )
            targets[xyz_file.name] = values

        except Exception:
            targets[xyz_file.name] = None
            failed += 1

    logger.info(f"  [OK] Extracted : {len(targets) - failed:,}")
    logger.info(f"  [!!] Failed    : {failed:,}")
    return targets


# ==============================================================================
# PyG Converter
# ==============================================================================

class QCGNNPreprocessor:

    def convert(self, mol_dict: dict, target_values: np.ndarray) -> Data:
        x = torch.tensor(mol_dict['node_features'], dtype=torch.float)

        edge_idx_raw = mol_dict['edge_index']
        if len(edge_idx_raw) > 0:
            edge_index = torch.tensor(
                edge_idx_raw, dtype=torch.long
            ).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        ef = mol_dict['edge_features']
        if len(ef) > 0:
            ef_bidir  = np.concatenate([ef, ef], axis=0)
            edge_attr = torch.tensor(ef_bidir, dtype=torch.float)
        else:
            edge_attr = torch.zeros((0, 4), dtype=torch.float)

        pos = torch.tensor(mol_dict['positions'], dtype=torch.float)
        y   = torch.tensor(target_values, dtype=torch.float).unsqueeze(0)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=y,
            mol_id=mol_dict['mol_id'],
            n_atoms=mol_dict['n_atoms'],
            filename=mol_dict['filename'],
        )

    def validate(self, data: Data) -> bool:
        if data.x is None or data.x.shape[0] == 0:
            return False
        if torch.isnan(data.x).any():
            return False
        if torch.isnan(data.y).any():
            return False
        if data.x.shape[1] != 9:
            return False
        return True


# ==============================================================================
# STEP 1 — Load Layer 1 Data
# ==============================================================================

logger.info(f"\n[STEP 1] Loading Layer 1 data...")

if not LAYER1_OUTPUT_PKL.exists():
    logger.error(f"Layer 1 pickle not found: {LAYER1_OUTPUT_PKL}")
    sys.exit(1)

with open(LAYER1_OUTPUT_PKL, 'rb') as f:
    layer1_data = pickle.load(f)

logger.info(f"  [OK] Loaded {len(layer1_data):,} molecules")

if MAX_SAMPLES is not None and MAX_SAMPLES < len(layer1_data):
    np.random.seed(RANDOM_SEED)
    idx = sorted(np.random.choice(len(layer1_data), MAX_SAMPLES, replace=False))
    layer1_data = [layer1_data[i] for i in idx]
    logger.info(f"  [OK] Subsampled to {len(layer1_data):,} (MAX_SAMPLES={MAX_SAMPLES})")
else:
    logger.info(f"  [OK] Using full dataset ({len(layer1_data):,} molecules)")


# ==============================================================================
# STEP 2 — Extract Targets from XYZ Files
# ==============================================================================

logger.info(f"\n[STEP 2] Extracting targets directly from XYZ files...")

targets_by_filename = extract_targets_from_xyz(RAW_DIR, QM9_TARGET_NAMES)


# ==============================================================================
# STEP 3 — Convert to PyG Data Objects
# ==============================================================================

logger.info(f"\n[STEP 3] Converting to PyG Data objects...")

preprocessor    = QCGNNPreprocessor()
pyg_data_list   = []
csv_records     = []
skipped         = 0
missing_targets = 0

for mol_dict in tqdm(layer1_data, desc="Converting"):
    filename    = mol_dict['filename']
    target_vals = targets_by_filename.get(filename, None)

    if target_vals is None:
        missing_targets += 1
        skipped += 1
        csv_records.append({
            'mol_id': mol_dict['mol_id'], 'filename': filename,
            'n_atoms': mol_dict['n_atoms'], 'n_bonds': mol_dict['n_bonds'],
            'n_edges': 0, 'status': 'NO_TARGET',
            **{n: None for n in QM9_TARGET_NAMES},
        })
        continue

    data = preprocessor.convert(mol_dict, target_vals)

    if preprocessor.validate(data):
        pyg_data_list.append(data)
        csv_records.append({
            'mol_id': mol_dict['mol_id'], 'filename': filename,
            'n_atoms': mol_dict['n_atoms'], 'n_bonds': mol_dict['n_bonds'],
            'n_edges': data.edge_index.shape[1], 'status': 'SUCCESS',
            **{n: float(target_vals[j]) for j, n in enumerate(QM9_TARGET_NAMES)},
        })
    else:
        skipped += 1
        csv_records.append({
            'mol_id': mol_dict['mol_id'], 'filename': filename,
            'n_atoms': mol_dict['n_atoms'], 'n_bonds': mol_dict['n_bonds'],
            'n_edges': 0, 'status': 'INVALID',
            **{n: None for n in QM9_TARGET_NAMES},
        })

logger.info(f"  [OK] Converted     : {len(pyg_data_list):,}")
logger.info(f"  [!!] No target     : {missing_targets:,}")
logger.info(f"  [!!] Invalid       : {skipped - missing_targets:,}")


# ==============================================================================
# STEP 4 — Normalize Targets
# ==============================================================================

logger.info(f"\n[STEP 4] Normalizing targets (z-score)...")

all_y       = torch.cat([d.y for d in pyg_data_list], dim=0)
target_mean = all_y.mean(dim=0)
target_std  = all_y.std(dim=0).clamp(min=1e-6)

for i, name in enumerate(QM9_TARGET_NAMES):
    logger.info(f"  {name:<6} : mean={target_mean[i].item():>10.5f}  "
                f"std={target_std[i].item():>8.5f}")

if target_std.min().item() < 1e-4:
    logger.warning("  [WARN]  Std near zero - target extraction may have failed!")
else:
    logger.info("  [OK] Targets are healthy (non-zero mean & std)")

for d in pyg_data_list:
    d.y = (d.y - target_mean) / target_std

norm_stats = {
    'mean': target_mean.tolist(),
    'std':  target_std.tolist(),
    'targets': QM9_TARGET_NAMES,
}


# ==============================================================================
# STEP 5 — Split
# ==============================================================================

logger.info(f"\n[STEP 5] Train / Val / Test split...")

np.random.seed(RANDOM_SEED)
n      = len(pyg_data_list)
perm   = np.random.permutation(n)
n_tr   = int(TRAIN_RATIO * n)
n_val  = int(VAL_RATIO   * n)

train_data = [pyg_data_list[i] for i in perm[:n_tr]]
val_data   = [pyg_data_list[i] for i in perm[n_tr : n_tr + n_val]]
test_data  = [pyg_data_list[i] for i in perm[n_tr + n_val:]]

logger.info(f"  Train ({int(TRAIN_RATIO*100)}%) : {len(train_data):,}")
logger.info(f"  Val   ({int(VAL_RATIO*100)}%)  : {len(val_data):,}")
logger.info(f"  Test  ({int(TEST_RATIO*100)}%)  : {len(test_data):,}")


# ==============================================================================
# STEP 6 — Save
# ==============================================================================

logger.info(f"\n[STEP 6] Saving outputs...")
LAYER2_OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)

with open(LAYER2_OUTPUT_PKL, 'wb') as f:
    pickle.dump({'data': pyg_data_list, 'norm_stats': norm_stats}, f)
logger.info(f"  [OK] Full PKL    : {LAYER2_OUTPUT_PKL}  "
            f"({LAYER2_OUTPUT_PKL.stat().st_size / 1e6:.1f} MB)")

with open(LAYER2_OUTPUT_TRAIN, 'wb') as f:
    pickle.dump(train_data, f)
logger.info(f"  [OK] Train PKL   : {LAYER2_OUTPUT_TRAIN}")

with open(LAYER2_OUTPUT_VAL, 'wb') as f:
    pickle.dump(val_data, f)
logger.info(f"  [OK] Val PKL     : {LAYER2_OUTPUT_VAL}")

with open(LAYER2_OUTPUT_TEST, 'wb') as f:
    pickle.dump(test_data, f)
logger.info(f"  [OK] Test PKL    : {LAYER2_OUTPUT_TEST}")

norm_path = LAYER2_OUTPUT_PKL.parent / "norm_stats.pkl"
with open(norm_path, 'wb') as f:
    pickle.dump(norm_stats, f)
logger.info(f"  [OK] Norm stats  : {norm_path}")

pd.DataFrame(csv_records).to_csv(LAYER2_OUTPUT_CSV, index=False)
logger.info(f"  [OK] Details CSV : {LAYER2_OUTPUT_CSV}")

stats = {
    'Total input':           len(layer1_data),
    'Converted':             len(pyg_data_list),
    'Skipped':               skipped,
    'Train':                 len(train_data),
    'Val':                   len(val_data),
    'Test':                  len(test_data),
    'Node feature dim':      9,
    'Edge feature dim':      4,
    'Targets':               ', '.join(QM9_TARGET_NAMES),
    **{f'Mean {QM9_TARGET_NAMES[i]}': round(target_mean[i].item(), 6)
       for i in range(len(QM9_TARGET_NAMES))},
    **{f'Std  {QM9_TARGET_NAMES[i]}': round(target_std[i].item(), 6)
       for i in range(len(QM9_TARGET_NAMES))},
}
pd.DataFrame.from_dict(stats, orient='index', columns=['Value']).to_csv(LAYER2_OUTPUT_STATS)
logger.info(f"  [OK] Stats CSV   : {LAYER2_OUTPUT_STATS}")


# ==============================================================================
# Final Summary
# ==============================================================================

logger.info("\n" + "=" * 80)
logger.info("LAYER 2 COMPLETE!")
logger.info("=" * 80)
logger.info(f"\n  Converted : {len(pyg_data_list):,} molecules")
logger.info(f"  Train     : {len(train_data):,}")
logger.info(f"  Val       : {len(val_data):,}")
logger.info(f"  Test      : {len(test_data):,}")
logger.info(f"\n  TARGET PROPERTIES (should NOT be zero):")
for i, name in enumerate(QM9_TARGET_NAMES):
    logger.info(f"    {name} -> mean={target_mean[i].item():.5f}  "
                f"std={target_std[i].item():.5f}")
logger.info(f"\n[NEXT STEP]")
logger.info(f"  python code/layer3_qcgnn_model/qcgnn_train.py")
logger.info("\n" + "=" * 80)