import sys
from pathlib import Path

# Setup paths to ensure we can import the existing codebase
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent)) # qcgnn_project
sys.path.insert(0, str(_HERE.parent.parent.parent))        # qcgnn_project/code

# Re-use the existing, validated QCGNN architecture identically
from layer4_quantum.quantum_model import HybridQuantumGNN as FullQCGNN
