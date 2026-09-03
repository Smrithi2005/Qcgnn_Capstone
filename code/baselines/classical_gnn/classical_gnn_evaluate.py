import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pickle
import json
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Setup paths
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent)) # qcgnn_project
sys.path.insert(0, str(_HERE.parent.parent.parent))        # qcgnn_project/code

import config
from baselines.classical_gnn.classical_gnn_model import ClassicalGNN

def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def evaluate():
    device = config.DEVICE
    test_path = config.LAYER2_DIR / "qm9_test.pkl"
        
    print(f"Evaluating on device: {device}")
    print(f"Loading test data from {test_path}")
    test_dataset = load_data(test_path)
    
    test_loader = DataLoader(test_dataset, batch_size=config.LAYER4_BATCH_SIZE, shuffle=False)
    
    model = ClassicalGNN(
        node_dim=config.NODE_FEATURE_DIM,
        edge_dim=config.EDGE_FEATURE_DIM,
        hidden_dim=64,
        num_layers=3,
        num_targets=len(config.QM9_TARGET_NAMES)
    ).to(device)
    
    model_save_path = config.MODELS_DIR / "best_classical_gnn.pt"
    print(f"Loading model weights from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    # Optional denormalization stats from Layer 2
    norm_path = config.LAYER2_DIR / "norm_stats.pkl"
    if norm_path.exists():
        with open(norm_path, "rb") as f:
            norm = pickle.load(f)
        mean_tgt = torch.tensor(norm["mean"]).numpy()
        std_tgt = torch.tensor(norm["std"]).numpy()
    else:
        mean_tgt = np.zeros(3)
        std_tgt = np.ones(3)
        
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
            target = batch.y.view(out.shape).float()
            
            preds.append(out.cpu().numpy())
            targets.append(target.cpu().numpy())
            
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    
    metrics = {}
    HARTREE_TO_EV = 27.211386245988
    
    for i, name in enumerate(['HOMO', 'LUMO', 'GAP']):
        # Denormalize predictions and targets to standard units (Hartree, then eV)
        y_true = targets[:, i] * std_tgt[i] + mean_tgt[i]
        y_pred = preds[:, i] * std_tgt[i] + mean_tgt[i]
        
        y_true_ev = y_true * HARTREE_TO_EV
        y_pred_ev = y_pred * HARTREE_TO_EV
        
        mae = mean_absolute_error(y_true_ev, y_pred_ev)
        rmse = np.sqrt(mean_squared_error(y_true_ev, y_pred_ev))
        r2 = r2_score(y_true_ev, y_pred_ev)
        
        metrics[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2)
        }
        
        print(f"\n{name} Metrics (eV):")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")
        
    metrics_path = config.RESULTS_DIR / "classical_gnn_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    evaluate()
