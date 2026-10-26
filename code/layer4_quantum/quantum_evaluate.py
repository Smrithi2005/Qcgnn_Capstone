import torch
from torch_geometric.loader import DataLoader
import pickle
import json
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Add project root to path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent))

from layer4_quantum.quantum_model import HybridQuantumGNN
import config

def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def evaluate():
    device = config.DEVICE
    test_path = config.LAYER3_DIR / "qm9_causal_homo_test.pkl"
    if not test_path.exists():
        test_path = config.LAYER3_DIR / "qm9_causal_test.pkl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at {config.LAYER3_DIR / 'qm9_causal_homo_test.pkl'} "
            f"or {config.LAYER3_DIR / 'qm9_causal_test.pkl'}."
        )

    model_save_path = config.MODELS_DIR / "best_qcgnn_corrected.pt"
    if not model_save_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_save_path}")

    norm_path = config.LAYER2_DIR / "norm_stats.pkl"
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization statistics not found: {norm_path}")

    print(f"Evaluating on device: {device}")
    print(f"Loading test data from {test_path}")
    test_dataset = load_data(test_path)
    if not test_dataset:
        raise ValueError(f"Test dataset is empty: {test_path}")

    test_loader = DataLoader(test_dataset, batch_size=config.LAYER4_BATCH_SIZE, shuffle=False)

    model = HybridQuantumGNN(
        node_dim=config.NODE_FEATURE_DIM,
        n_qubits=config.N_QUBITS_BASE,
        n_rotation_layers=config.N_ROTATION_LAYERS,
        n_entangle_layers=config.N_ENTANGLE_LAYERS,
        entangle_mode=config.ENTANGLE_MODE,
        num_targets=len(config.QM9_TARGET_NAMES)
    ).to(device)

    print(f"Loading model weights from {model_save_path}")
    state_dict = torch.load(model_save_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a model state dictionary: {model_save_path}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with open(norm_path, "rb") as f:
        norm = pickle.load(f)
    expected_targets = ["homo", "lumo", "gap"]
    if norm.get("targets") != expected_targets:
        raise ValueError(
            f"Normalization target order must be {expected_targets}, "
            f"got {norm.get('targets')}."
        )
    mean_tgt = np.asarray(norm.get("mean"), dtype=np.float64)
    std_tgt = np.asarray(norm.get("std"), dtype=np.float64)
    if mean_tgt.shape != (3,) or std_tgt.shape != (3,):
        raise ValueError("Normalization mean and std must each contain three targets.")
    if not np.isfinite(mean_tgt).all() or not np.isfinite(std_tgt).all() or (std_tgt <= 0).any():
        raise ValueError("Normalization mean/std contain non-finite values or non-positive standard deviations.")

    preds = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            target = batch.y.view(out.shape).float()

            preds.append(out.cpu().numpy())
            targets.append(target.cpu().numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)
    if not np.isfinite(preds).all() or not np.isfinite(targets).all():
        raise ValueError("Non-finite predictions or targets encountered during evaluation.")

    metrics = {}
    target_names = ['HOMO', 'LUMO', 'GAP']
    HARTREE_TO_EV = 27.211386245988
    for i, name in enumerate(target_names):
        # Layer 4 trains on standardized Layer 3 targets. The stored statistics
        # are in Hartree, so reverse standardization once, then convert to eV.
        y_true = targets[:, i] * std_tgt[i] + mean_tgt[i]
        y_pred = preds[:, i] * std_tgt[i] + mean_tgt[i]
        y_true_ev = y_true * HARTREE_TO_EV
        y_pred_ev = y_pred * HARTREE_TO_EV
        if not np.isfinite(y_true_ev).all() or not np.isfinite(y_pred_ev).all():
            raise ValueError(f"Non-finite denormalized values encountered for {name}.")

        mae = mean_absolute_error(y_true_ev, y_pred_ev)
        rmse = np.sqrt(mean_squared_error(y_true_ev, y_pred_ev))
        r2 = r2_score(y_true_ev, y_pred_ev)

        metrics[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "units": "eV"
        }

        print(f"\n{name} Metrics (eV):")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")

    metrics_path = config.RESULTS_DIR / "quantum_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    evaluate()
