import sys
from pathlib import Path

# Resolve path to project root relative to this file
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import config

def check_paths():
    project_root = config.PROJECT_ROOT
    model_path = config.MODELS_DIR / "best_qcgnn.pt"
    history_path = config.RESULTS_DIR / "quantum_training_history.json"
    metrics_path = config.RESULTS_DIR / "quantum_test_metrics.json"

    # Ensure directories exist (config.py already does this, but explicitly verifying)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Project root:\n{project_root}")
    print(f"Model path:\n{model_path}")
    print(f"Training history:\n{history_path}")
    print(f"Evaluation metrics:\n{metrics_path}")
    
    # Verify exact path resolution and preservation
    if model_path.exists():
        print(f"\nThe existing checkpoint is preserved exactly at:\n{model_path}")
    else:
        print("\nNo existing checkpoint found at the target location.")

if __name__ == "__main__":
    check_paths()
