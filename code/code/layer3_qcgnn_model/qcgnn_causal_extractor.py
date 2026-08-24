"""
LAYER 3: Causal SCM - Extract Minimal Sufficient Subgraphs (G_c)
================================================================

This layer:
1. Loads pre-computed Layer 2 PyG data (no reprocessing)
2. Extracts causal subgraph G_c for each target (HOMO, LUMO, gap)
3. Uses gradient-based atom importance masking
4. Saves causal embeddings as PKL for Layer 4

Optimized for RTX A2000 with 8.6GB VRAM.
"""

import os
import sys
from pathlib import Path
import numpy as np
import logging
import pickle
import pandas as pd
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    LAYER2_OUTPUT_PKL,
    LAYER3_OUTPUT_DIR,
    LAYER3_OUTPUT_PKL,
    LAYER3_OUTPUT_CAUSAL_STATS,
    QM9_TARGET_NAMES,
    CAUSAL_MIN_ATOMS,
    CAUSAL_MAX_ATOMS,
    CAUSAL_AUTO_FRACTION,
    GCN_HIDDEN_DIM,
    DEVICE,
    RANDOM_SEED,
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
        logging.FileHandler("layer3_output.txt", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("LAYER 3: Causal SCM - Extract Minimal Sufficient Subgraphs")
logger.info("=" * 80)
logger.info(f"Device: {DEVICE}")
logger.info(f"Causal atoms range: {CAUSAL_MIN_ATOMS} - {CAUSAL_MAX_ATOMS}")


# ==============================================================================
# STEP 1: Load Pre-computed Layer 2 Data
# ==============================================================================

logger.info(f"\n[STEP 1] Loading Layer 2 data...")

if not LAYER2_OUTPUT_PKL.exists():
    logger.error(f"Layer 2 PKL not found: {LAYER2_OUTPUT_PKL}")
    sys.exit(1)

with open(LAYER2_OUTPUT_PKL, 'rb') as f:
    layer2_data = pickle.load(f)
    pyg_data_list = layer2_data['data']
    norm_stats = layer2_data['norm_stats']

logger.info(f"  [OK] Loaded {len(pyg_data_list):,} PyG Data objects")
logger.info(f"  [OK] Targets: {QM9_TARGET_NAMES}")


# ==============================================================================
# Causal Subgraph Extractor (Gradient-Based)
# ==============================================================================

class CausalSubgraphExtractor(torch.nn.Module):
    """
    Extracts causal atoms via gradient-based importance masking.
    
    How it works:
    1. Build a simple GCN model
    2. Forward pass: predict target from node features
    3. Backward pass: compute gradient of loss w.r.t. node features
    4. High gradient magnitude = important atom
    5. Keep top-k atoms as G_c
    """
    
    def __init__(self, node_feature_dim: int = 9, hidden_dim: int = 32, device: str = 'cpu'):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Simple 2-layer GCN
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Readout: simple linear layer to predict 1D value
        self.readout = torch.nn.Linear(hidden_dim, 1)
        
        self.to(device)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through GCN"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global mean pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        x = self.readout(x)
        return x
    
    def compute_node_importance(self, data: Data, target_idx: int) -> np.ndarray:
        """
        Compute importance score for each atom via gradient.
        
        Args:
            data: PyG Data object
            target_idx: Which target (0=homo, 1=lumo, 2=gap)
        
        Returns:
            importance: (n_atoms,) array of importance scores
        """
        # Detach and clone node features for gradient computation
        x = data.x.clone().detach().requires_grad_(True)
        edge_index = data.edge_index.to(self.device)
        
        # Forward pass
        pred = self.forward(x, edge_index)
        target = data.y[0, target_idx].to(self.device)
        
        # MSE loss
        loss = F.mse_loss(pred, target.unsqueeze(0))
        
        # Backward: compute gradient of loss w.r.t. input features
        loss.backward()
        
        # Sum absolute gradients across feature dimensions
        importance = x.grad.abs().sum(dim=1).detach().cpu().numpy()
        
        return importance
    
    def extract_subgraph(
        self, 
        data: Data, 
        target_idx: int,
        top_k: int = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract causal subgraph G_c by keeping top-k important atoms.
        """
        importance = self.compute_node_importance(data, target_idx)
        n_atoms = len(importance)
        
        # Auto-select k based on molecule size
        if top_k is None:
            top_k = max(CAUSAL_MIN_ATOMS, min(CAUSAL_MAX_ATOMS, int(CAUSAL_AUTO_FRACTION * n_atoms)))
        
        # Keep top-k atoms
        important_idx = np.argsort(importance)[-top_k:]
        important_mask = np.zeros(n_atoms, dtype=bool)
        important_mask[important_idx] = True
        
        # Extract subgraph features
        subgraph_x = data.x[important_mask]
        
        return subgraph_x, important_mask


# ==============================================================================
# STEP 2: Extract Causal Subgraphs
# ==============================================================================

logger.info(f"\n[STEP 2] Extracting causal subgraphs G_c...")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

extractor = CausalSubgraphExtractor(
    node_feature_dim=9, 
    hidden_dim=GCN_HIDDEN_DIM,
    device=DEVICE
)

causal_data_dict = {}

for target_idx, target_name in enumerate(QM9_TARGET_NAMES):
    logger.info(f"\n  Extracting G_c for: {target_name}")
    subgraph_records = []
    
    for idx, data in enumerate(tqdm(pyg_data_list, desc=f"    {target_name}", leave=False)):
        try:
            data = data.to(DEVICE)
            
            subgraph_x, important_mask = extractor.extract_subgraph(
                data, 
                target_idx,
            )
            
            # Store result
            subgraph_records.append({
                'mol_id': idx,
                'n_atoms_full': data.x.shape[0],
                'n_atoms_causal': important_mask.sum(),
                'causal_ratio': float(important_mask.sum()) / len(important_mask),
                'subgraph_x': subgraph_x.detach().cpu().numpy(),
                'important_mask': important_mask,
                'target_value': float(data.y[0, target_idx].cpu()),
            })
            
            # Clear GPU memory periodically
            if (idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
        
        except Exception as e:
            logger.warning(f"    Failed for mol {idx}: {e}")
            continue
    
    causal_data_dict[target_name] = subgraph_records
    logger.info(f"  [OK] Extracted {len(subgraph_records):,} G_c for {target_name}")


# ==============================================================================
# STEP 3: Save Outputs
# ==============================================================================

logger.info(f"\n[STEP 3] Saving outputs...")
LAYER3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save causal data
with open(LAYER3_OUTPUT_PKL, 'wb') as f:
    pickle.dump({
        'causal_data': causal_data_dict,
        'norm_stats': norm_stats,
        'target_names': QM9_TARGET_NAMES,
    }, f)
logger.info(f"  [OK] Saved: {LAYER3_OUTPUT_PKL}")

# Compute statistics
causal_stats = {}
for target_name, records in causal_data_dict.items():
    if records:
        causal_ratios = np.array([r['causal_ratio'] for r in records])
        causal_atoms = np.array([r['n_atoms_causal'] for r in records])
        
        causal_stats[target_name] = {
            'n_subgraphs': len(records),
            'avg_causal_ratio': float(causal_ratios.mean()),
            'min_causal_atoms': int(causal_atoms.min()),
            'max_causal_atoms': int(causal_atoms.max()),
            'mean_causal_atoms': float(causal_atoms.mean()),
        }

stats_df = pd.DataFrame.from_dict(causal_stats, orient='index')
stats_df.to_csv(LAYER3_OUTPUT_CAUSAL_STATS)
logger.info(f"  [OK] Saved: {LAYER3_OUTPUT_CAUSAL_STATS}")


# ==============================================================================
# Final Summary
# ==============================================================================

logger.info("\n" + "=" * 80)
logger.info("LAYER 3 COMPLETE!")
logger.info("=" * 80)

logger.info(f"\n[CAUSAL SUBGRAPH EXTRACTION SUMMARY]")
for target_name, stats in causal_stats.items():
    logger.info(f"\n  {target_name}:")
    logger.info(f"    Subgraphs extracted    : {stats['n_subgraphs']:,}")
    logger.info(f"    Avg causal atoms       : {stats['mean_causal_atoms']:.1f}")
    logger.info(f"    Range                  : {stats['min_causal_atoms']} – {stats['max_causal_atoms']}")
    logger.info(f"    Avg importance ratio   : {stats['avg_causal_ratio']:.1%}")

logger.info(f"\n[KEY DESIGN DECISION]")
logger.info(f"  Causal SCM BEFORE Quantum Circuit (Option A)")
logger.info(f"  → Filter in classical domain (formal SCM guarantees)")
logger.info(f"  → Then quantum transforms only causally relevant atoms")
logger.info(f"  → Prevents CNOT entanglement from exploiting scaffold bias")

logger.info(f"\n[NEXT STEP]")
logger.info(f"  python code/layer4_quantum_circuit/qcgnn_train.py")
logger.info("\n" + "=" * 80)

torch.cuda.empty_cache()