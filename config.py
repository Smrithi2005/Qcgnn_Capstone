"""
config.py
=========
Central configuration for the entire QCGNN project.
All paths, hyperparameters, and constants live here.

Project structure assumed:
    qcgnn_project/
        code/
            config.py               ← this file
            layer1_gschnet/
            layer2_qcgnn/
            layer3_qcgnn_model/
        data/
            qm9/
                raw/                ← put your .xyz files here
                processed/          ← Layer 1 outputs go here
                layer2/             ← Layer 2 outputs go here
                layer3/             ← Layer 3 outputs go here
                layer4/             ← Layer 4-5 outputs go here
        models/
        results/
        logs/
"""

import logging
import os
from pathlib import Path

# ==============================================================================
# Logging
# ==============================================================================

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==============================================================================
# Project Root (always resolves correctly from any run location)
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent 
DATA_ROOT = PROJECT_ROOT / "data" / "qm9"

# ==============================================================================
# Directory Paths
# ==============================================================================

RAW_DIR = DATA_ROOT / "raw"                    # where your .xyz files live
PROCESSED_DIR = DATA_ROOT / "processed"        # Layer 1 outputs
LAYER2_DIR = DATA_ROOT / "layer2"              # Layer 2 outputs
LAYER3_DIR = DATA_ROOT / "layer3"              # Layer 3 outputs
LAYER4_DIR = DATA_ROOT / "layer4"              # Layer 4-5 outputs
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for _dir in [RAW_DIR, PROCESSED_DIR, LAYER2_DIR, LAYER3_DIR, LAYER4_DIR, 
             MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Layer 1 — G-SchNet Preprocessing
# ==============================================================================

LAYER1_INPUT = RAW_DIR
LAYER1_OUTPUT_PKL = PROCESSED_DIR / "qm9_layer1.pkl"
LAYER1_OUTPUT_CSV = PROCESSED_DIR / "qm9_layer1_details.csv"
LAYER1_OUTPUT_STATS = PROCESSED_DIR / "qm9_layer1_statistics.csv"

# Atom type → atomic number mapping
ATOMIC_NUMBERS = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Br': 35,
    'I': 53,
}

# Van der Waals radii (Angstroms) by atomic number
VDW_RADII = {
    1: 1.20,    # H
    6: 1.70,    # C
    7: 1.55,    # N
    8: 1.52,    # O
    9: 1.47,    # F
    15: 1.80,   # P
    16: 1.80,   # S
    17: 1.75,   # Cl
    35: 1.85,   # Br
    53: 1.98,   # I
}

BOND_THRESHOLD_FACTOR = 1.1   # multiply sum of vdW radii by this for bond cutoff

NODE_FEATURE_DIM = 9    # per-atom feature vector size
EDGE_FEATURE_DIM = 4    # per-bond feature vector size

# ==============================================================================
# Layer 2 — QCGNN Preprocessing
# ==============================================================================

# Output files
LAYER2_OUTPUT_PKL = LAYER2_DIR / "qm9_layer2.pkl"
LAYER2_OUTPUT_TRAIN = LAYER2_DIR / "qm9_train.pkl"
LAYER2_OUTPUT_VAL = LAYER2_DIR / "qm9_val.pkl"
LAYER2_OUTPUT_TEST = LAYER2_DIR / "qm9_test.pkl"
LAYER2_OUTPUT_CSV = LAYER2_DIR / "qm9_layer2_details.csv"
LAYER2_OUTPUT_STATS = LAYER2_DIR / "qm9_layer2_statistics.csv"

# QM9 target properties to predict
# Full list:
#   0=mu, 1=alpha, 2=homo, 3=lumo, 4=gap, 5=r2, 6=zpve,
#   7=u0, 8=u298, 9=h298, 10=g298, 11=cv
QM9_TARGET_INDICES = [2, 3, 4]                  # HOMO, LUMO, Gap (eV)
QM9_TARGET_NAMES = ['homo', 'lumo', 'gap']      # must match indices above

# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# MAX_SAMPLES: set to an integer to use a subset, or None for full 133k dataset
# Recommended values:
#   None      → full ~133k  (20–30 hrs training on RTX A2000)
#   50000     → 50k subset  (6–10 hrs)
#   25000     → 25k subset  (2–4 hrs)   ← good for research/testing
#   1000      → quick test  (~5 min)
MAX_SAMPLES = 25000

# Random seed for reproducible splits
RANDOM_SEED = 42

# ==============================================================================
# Layer 3 — Causal SCM Output Paths
# ==============================================================================

LAYER3_OUTPUT_PKL = LAYER3_DIR / "qm9_layer3.pkl"
LAYER3_OUTPUT_CAUSAL_STATS = LAYER3_DIR / "qm9_layer3_causal_stats.csv"
LAYER3_OUTPUT_TRAIN = LAYER3_DIR / "qm9_causal_train.pkl"
LAYER3_OUTPUT_VAL = LAYER3_DIR / "qm9_causal_val.pkl"
LAYER3_OUTPUT_TEST = LAYER3_DIR / "qm9_causal_test.pkl"

# ==============================================================================
# Layer 4-5 — Quantum Circuit + Training Paths
# ==============================================================================

LAYER4_CHECKPOINT_DIR = LAYER4_DIR / "checkpoints"
LAYER4_BEST_MODEL = LAYER4_CHECKPOINT_DIR / "best_model.pt"
LAYER4_TRAINING_LOG = LAYER4_DIR / "training_log.csv"
LAYER4_PREDICTIONS = LAYER4_DIR / "predictions.pkl"

# ==============================================================================
# GPU CONFIGURATION FOR RTX A2000 WITH 8.6GB VRAM
# ==============================================================================

import torch
import numpy as np

# Auto-detect device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize GPU
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    print(f"\n[GPU SETUP]")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")

# BATCH SIZES - Optimized for 8.6GB VRAM
# Memory estimate: ~80-100 MB per batch of 32 molecules
BATCH_SIZE = 32                 # Large batch for better GPU utilization
LAYER3_BATCH_SIZE = 16          # Layer 3 (gradient computation)
LAYER4_BATCH_SIZE = 32          # Layer 4-5 (training)

# Memory optimization
TORCH_EMPTY_CACHE_EVERY = 20    # Clear cache every 20 batches
TORCH_CUDA_DEVICE = 0           # Use GPU 0
USE_AUTOMATIC_MIXED_PRECISION = False  # Disable AMP (not needed with 8.6GB)
LAYER3_GRADIENT_CHECKPOINT = False     # Disable gradient checkpointing (faster)
PIN_MEMORY = True
NUM_WORKERS = 0

# ==============================================================================
# Causal SCM Hyperparameters (Layer 3)
# ==============================================================================

CAUSAL_MIN_ATOMS = 3
CAUSAL_MAX_ATOMS = 12           # Use full range (more atoms = more accurate)
CAUSAL_AUTO_FRACTION = 0.2      # Auto-select ~20% of atoms
CAUSAL_FILTER_MODE = 'gradient' # Gradient-based importance

# ==============================================================================
# Quantum Circuit Hyperparameters (Layer 4)
# ==============================================================================

N_QUBITS_BASE = 8
N_ROTATION_LAYERS = 2
N_ENTANGLE_LAYERS = 1
ENTANGLE_MODE = 'linear'

# ==============================================================================
# Training Hyperparameters (Layers 4-5 Joint)
# ==============================================================================

LEARNING_RATE = 1e-3
LR_DECAY_FACTOR = 0.95
LR_DECAY_STEPS = 10
N_EPOCHS = 100                  # Can run more epochs with better GPU
EARLY_STOPPING_PATIENCE = 15
GRADIENT_CLIP_MAX_NORM = 1.0
WEIGHT_DECAY = 0.0
LAMBDA_CAUSAL = 0.1

# ==============================================================================
# Causal Extractor Configuration
# ==============================================================================

GCN_HIDDEN_DIM = 64             # Increased from 32
USE_PRETRAINED_GCN = False
IMPORTANCE_SMOOTHING = 1e-6

# ==============================================================================
# Random Seeds
# ==============================================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True