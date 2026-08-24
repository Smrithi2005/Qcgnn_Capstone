"""
LAYER 3 QUICK START - Causal SCM Extraction
============================================
Runs the complete Layer 3 pipeline:
1. Load Layer 2 data (PKL files)
2. Extract causal subgraphs using gradient-based masking
3. Save Layer 3 outputs (PKL + statistics)

Optimized for RTX A2000 with 8GB VRAM
"""

import sys
import os
from pathlib import Path

# Add code folder to path
sys.path.insert(0, str(Path(__file__).parent / "code"))

import torch
import numpy as np
import pickle
import logging
from config import *

# ==============================================================================
# SETUP LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CHECK 1: Verify all required files exist
# ==============================================================================

print("\n" + "="*80)
print("LAYER 3: CAUSAL SCM - QUICK START")
print("="*80)

print("\n[CHECK 1] Verifying Layer 1 & 2 PKL files...")

# Check Layer 2 files
if LAYER2_OUTPUT_PKL.exists():
    print(f"  ✓ Layer 2 PKL found: {LAYER2_OUTPUT_PKL}")
else:
    print(f"  ✗ ERROR: Layer 2 PKL NOT found: {LAYER2_OUTPUT_PKL}")
    print("  Please run Layer 1 & 2 first!")
    sys.exit(1)

if LAYER1_OUTPUT_PKL.exists():
    print(f"  ✓ Layer 1 PKL found: {LAYER1_OUTPUT_PKL}")
else:
    print(f"  ✗ ERROR: Layer 1 PKL NOT found: {LAYER1_OUTPUT_PKL}")
    sys.exit(1)

# ==============================================================================
# CHECK 2: Verify GPU
# ==============================================================================

print("\n[CHECK 2] Verifying GPU...")
if torch.cuda.is_available():
    print(f"  ✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  ✓ Compute Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("  ✗ WARNING: GPU NOT available - will use CPU (SLOW)")
    DEVICE = 'cpu'

# ==============================================================================
# CHECK 3: Verify Layer 3 code file
# ==============================================================================

print("\n[CHECK 3] Verifying Layer 3 code file...")
layer3_code_path = Path(__file__).parent / "code" / "layer3_qcgnn_model" / "qcgnn_causal_extractor.py"
if layer3_code_path.exists():
    print(f"  ✓ Layer 3 code found: {layer3_code_path}")
else:
    print(f"  ✗ ERROR: Layer 3 code NOT found: {layer3_code_path}")
    sys.exit(1)

# ==============================================================================
# CHECK 4: Create Layer 3 output directories
# ==============================================================================

print("\n[CHECK 4] Creating Layer 3 output directories...")
LAYER3_DIR.mkdir(parents=True, exist_ok=True)
print(f"  ✓ Layer 3 output directory ready: {LAYER3_DIR}")

# ==============================================================================
# LOAD LAYER 2 DATA
# ==============================================================================

print("\n[STEP 1] Loading Layer 2 data...")
try:
    with open(LAYER2_OUTPUT_PKL, 'rb') as f:
        layer2_data = pickle.load(f)
    print(f"  ✓ Loaded Layer 2 PKL: {len(layer2_data)} molecules")
except Exception as e:
    print(f"  ✗ ERROR loading Layer 2 data: {e}")
    sys.exit(1)

# ==============================================================================
# IMPORT LAYER 3 MODULE
# ==============================================================================

print("\n[STEP 2] Importing Layer 3 causal extractor...")
try:
    from code.layer3_qcgnn_model.qcgnn_causal_extractor import (
        CausalExtractor,
        extract_causal_subgraphs
    )
    print("  ✓ Successfully imported causal extractor")
except ImportError as e:
    print(f"  ✗ ERROR importing Layer 3 module: {e}")
    sys.exit(1)

# ==============================================================================
# INITIALIZE CAUSAL EXTRACTOR
# ==============================================================================

print("\n[STEP 3] Initializing Causal Extractor...")
try:
    extractor = CausalExtractor(
        hidden_dim=GCN_HIDDEN_DIM,
        use_pretrained=USE_PRETRAINED_GCN,
        importance_smoothing=IMPORTANCE_SMOOTHING,
        device=DEVICE
    )
    print(f"  ✓ Causal Extractor initialized")
    print(f"  ✓ Hidden dim: {GCN_HIDDEN_DIM}")
    print(f"  ✓ Device: {DEVICE}")
except Exception as e:
    print(f"  ✗ ERROR initializing extractor: {e}")
    sys.exit(1)

# ==============================================================================
# EXTRACT CAUSAL SUBGRAPHS
# ==============================================================================

print("\n" + "="*80)
print("[RUNNING LAYER 3 - Causal Subgraph Extraction]")
print("="*80)
print(f"Config:")
print(f"  • Batch size: {LAYER3_BATCH_SIZE}")
print(f"  • Min atoms: {CAUSAL_MIN_ATOMS}")
print(f"  • Max atoms: {CAUSAL_MAX_ATOMS}")
print(f"  • Auto fraction: {CAUSAL_AUTO_FRACTION}")
print(f"  • Filter mode: {CAUSAL_FILTER_MODE}")
print(f"Expected time: 5-15 minutes on RTX A2000\n")

try:
    causal_data = extract_causal_subgraphs(
        layer2_data,
        extractor=extractor,
        batch_size=LAYER3_BATCH_SIZE,
        min_atoms=CAUSAL_MIN_ATOMS,
        max_atoms=CAUSAL_MAX_ATOMS,
        auto_fraction=CAUSAL_AUTO_FRACTION,
        filter_mode=CAUSAL_FILTER_MODE,
        device=DEVICE
    )
    print("\n  ✓ Causal subgraph extraction completed!")
except Exception as e:
    print(f"\n  ✗ ERROR during causal extraction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================================================
# SAVE LAYER 3 OUTPUTS
# ==============================================================================

print("\n[STEP 4] Saving Layer 3 outputs...")
try:
    # Save main causal data
    with open(LAYER3_OUTPUT_PKL, 'wb') as f:
        pickle.dump(causal_data, f)
    print(f"  ✓ Saved: {LAYER3_OUTPUT_PKL}")
    
    # Generate and save statistics
    if 'stats' in causal_data:
        stats_df = causal_data['stats']
        stats_df.to_csv(LAYER3_OUTPUT_CAUSAL_STATS, index=False)
        print(f"  ✓ Saved: {LAYER3_OUTPUT_CAUSAL_STATS}")
    
    print("\n✓ ALL LAYER 3 OUTPUTS SAVED SUCCESSFULLY!")
    
except Exception as e:
    print(f"  ✗ ERROR saving outputs: {e}")
    sys.exit(1)

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("LAYER 3 COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nOutputs saved to: {LAYER3_DIR}")
print(f"\nNext steps:")
print(f"  1. Check data/qm9/layer3/qm9_layer3_causal_stats.csv")
print(f"  2. Verify mean_causal_atoms ≈ 7 (for each target)")
print(f"  3. Run Layer 4-5 quantum circuit training:")
print(f"     python code/layer4_quantum_circuit/qcgnn_train.py")
print("\n" + "="*80 + "\n")