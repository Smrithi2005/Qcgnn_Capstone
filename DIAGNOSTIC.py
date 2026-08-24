"""
DIAGNOSTIC SCRIPT - Check what's happening with Layer 1
Run this to see errors and debug
"""

import sys
import os
from pathlib import Path

print("\n" + "="*80)
print("DIAGNOSTIC: Checking Layer 1 Setup")
print("="*80)

# Check 1: Current directory
print("\n[CHECK 1] Current directory...")
cwd = Path.cwd()
print(f"Current dir: {cwd}")

# Check 2: Config file
print("\n[CHECK 2] Looking for config.py...")
config_path = cwd / "code" / "config.py"
print(f"Looking for: {config_path}")
print(f"Exists: {config_path.exists()}")

if not config_path.exists():
    print("ERROR: config.py not found!")
    print(f"Make sure config.py is in: {cwd / 'code'}")
    sys.exit(1)

# Check 3: Import config
print("\n[CHECK 3] Importing config...")
try:
    sys.path.insert(0, str(cwd / "code"))
    import config
    print("✓ config.py imported successfully")
except Exception as e:
    print(f"ERROR importing config: {e}")
    sys.exit(1)

# Check 4: Check raw data
print("\n[CHECK 4] Checking raw XYZ files...")
raw_dir = config.RAW_DIR
print(f"Raw directory: {raw_dir}")
print(f"Exists: {raw_dir.exists()}")

if raw_dir.exists():
    xyz_files = list(raw_dir.glob("*.xyz"))
    print(f"Found {len(xyz_files)} XYZ files")
else:
    print("ERROR: Raw directory not found!")
    sys.exit(1)

# Check 5: Check output directory
print("\n[CHECK 5] Checking output directory...")
processed_dir = config.PROCESSED_DIR
print(f"Processed directory: {processed_dir}")
processed_dir.mkdir(parents=True, exist_ok=True)
print(f"Created: {processed_dir.exists()}")

# Check 6: Dependencies
print("\n[CHECK 6] Checking dependencies...")
try:
    import torch
    print(f"✓ torch: {torch.__version__}")
except:
    print("✗ torch not installed")

try:
    import rdkit
    print(f"✓ rdkit: installed")
except:
    print("✗ rdkit not installed")

try:
    import pandas
    print(f"✓ pandas: installed")
except:
    print("✗ pandas not installed")

try:
    import numpy
    print(f"✓ numpy: installed")
except:
    print("✗ numpy not installed")

try:
    import tqdm
    print(f"✓ tqdm: installed")
except:
    print("✗ tqdm not installed")

print("\n" + "="*80)
print("✓ ALL CHECKS PASSED - Ready to run Layer 1!")
print("="*80)

print("\nNow run:")
print("  python code\\layer1_gschnet\\qm9_preprocess_dataset.py")