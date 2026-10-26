import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn import global_max_pool, global_mean_pool

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent))

from layer4_quantum.quantum_model import HybridQuantumGNN
import config


HARTREE_TO_EV = 27.211386245988
DATA_PATH = config.LAYER3_DIR / "qm9_causal_homo_test.pkl"
CHECKPOINT_PATH = config.MODELS_DIR / "best_qcgnn_corrected.pt"
NORM_PATH = config.LAYER2_DIR / "norm_stats.pkl"
REPORT_PATH = config.RESULTS_DIR / "quantum_simulator_baseline_corrected.json"


def main():
    for path, label in (
        (DATA_PATH, "Layer 3 dataset"),
        (CHECKPOINT_PATH, "corrected checkpoint"),
        (NORM_PATH, "normalization statistics"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if REPORT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite baseline report: {REPORT_PATH}")

    with open(DATA_PATH, "rb") as handle:
        dataset = pickle.load(handle)
    if not dataset:
        raise ValueError(f"Layer 3 dataset is empty: {DATA_PATH}")
    graph = dataset[0]

    with open(NORM_PATH, "rb") as handle:
        norm = pickle.load(handle)
    if norm.get("targets") != ["homo", "lumo", "gap"]:
        raise ValueError(f"Unexpected normalization target order: {norm.get('targets')}")
    mean = np.asarray(norm["mean"], dtype=np.float64)
    std = np.asarray(norm["std"], dtype=np.float64)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization statistics must contain three finite values each.")

    device = config.DEVICE
    model = HybridQuantumGNN(
        node_dim=config.NODE_FEATURE_DIM,
        n_qubits=config.N_QUBITS_BASE,
        n_rotation_layers=config.N_ROTATION_LAYERS,
        n_entangle_layers=config.N_ENTANGLE_LAYERS,
        entangle_mode=config.ENTANGLE_MODE,
        num_targets=len(config.QM9_TARGET_NAMES),
    ).to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    graph = graph.to(device)
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        pooled_mean = global_mean_pool(graph.x.float(), batch)
        pooled_max = global_max_pool(graph.x.float(), batch)
        pooled = torch.cat([pooled_mean, pooled_max], dim=1)
        angles = model.classical_projection(pooled)
        measurements = model.qlayer(angles)
        standardized = model.classical_readout(measurements)

    angles_np = angles[0].cpu().numpy()
    measurements_np = measurements[0].cpu().numpy()
    standardized_np = standardized[0].cpu().numpy()
    predictions_ev = (standardized_np * std + mean) * HARTREE_TO_EV
    arrays = (angles_np, measurements_np, standardized_np, predictions_ev)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Non-finite values encountered in simulator baseline.")

    report = {
        "dataset": str(DATA_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(device),
        "graph_index": 0,
        "graph_identifier": getattr(graph, "mol_id", None),
        "shapes": {
            "x": list(graph.x.shape),
            "edge_index": list(graph.edge_index.shape),
            "edge_attr": list(graph.edge_attr.shape) if hasattr(graph, "edge_attr") else None,
            "y": list(graph.y.shape) if hasattr(graph, "y") else None,
        },
        "target_order": ["HOMO", "LUMO", "GAP"],
        "input_angles": angles_np.tolist(),
        "quantum_measurements": measurements_np.tolist(),
        "standardized_predictions": standardized_np.tolist(),
        "predictions_eV": predictions_ev.tolist(),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "hartree_to_ev": HARTREE_TO_EV,
    }
    with open(REPORT_PATH, "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Baseline report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
