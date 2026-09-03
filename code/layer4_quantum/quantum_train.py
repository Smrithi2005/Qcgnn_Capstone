import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pickle
import json
import time
import sys
from pathlib import Path

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

def integration_test():
    print("--- LAYER 3 TO 4 INTEGRATION TEST ---")
    train_path = config.LAYER3_DIR / "qm9_causal_homo_train.pkl"
    if not train_path.exists():
        # Fallback if only the non-prefixed version exists
        train_path = config.LAYER3_DIR / "qm9_causal_train.pkl"
        
    print(f"Loading data from {train_path}")
    dataset = load_data(train_path)
    sample = dataset[0]
    
    print(f"Original graph node count: {len(sample.kept_index)}")
    print(f"Causal graph node count: {sample.x.size(0)}")
    print(f"Node feature shape: {list(sample.x.shape)}")
    print(f"Edge index shape: {list(sample.edge_index.shape)}")
    if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
        print(f"Edge feature shape: {list(sample.edge_attr.shape)}")
    else:
        print("Edge feature shape: None")
    print(f"Target shape: {list(sample.y.shape)}")
    
    model = HybridQuantumGNN(
        node_dim=config.NODE_FEATURE_DIM,
        n_qubits=config.N_QUBITS_BASE,
        n_rotation_layers=config.N_ROTATION_LAYERS,
        n_entangle_layers=config.N_ENTANGLE_LAYERS,
        entangle_mode=config.ENTANGLE_MODE,
        num_targets=len(config.QM9_TARGET_NAMES)
    )
    
    # Run the model manually step by step to print shapes
    batch = torch.zeros(sample.x.size(0), dtype=torch.long)
    from torch_geometric.nn import global_mean_pool, global_max_pool
    h_mean = global_mean_pool(sample.x, batch)
    h_max = global_max_pool(sample.x, batch)
    h_c = torch.cat([h_mean, h_max], dim=1)
    
    print(f"Projected feature (graph representation) shape: {list(h_c.shape)}")
    
    z = model.classical_projection(h_c)
    print(f"Number of qubits: {config.N_QUBITS_BASE}")
    print(f"Quantum input (z) shape: {list(z.shape)}")
    
    q_out = model.qlayer(z)
    print(f"Quantum output shape: {list(q_out.shape)}")
    
    final_out = model.classical_readout(q_out)
    print(f"Final prediction shape: {list(final_out.shape)}")
    print("Integration test passed.\n")

def train():
    integration_test()
    
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.N_EPOCHS)
    args = ap.parse_args()
    
    device = config.DEVICE
    print(f"Training on device: {device}")
    
    train_path = config.LAYER3_DIR / "qm9_causal_homo_train.pkl"
    val_path = config.LAYER3_DIR / "qm9_causal_homo_val.pkl"
    if not train_path.exists():
        train_path = config.LAYER3_DIR / "qm9_causal_train.pkl"
        val_path = config.LAYER3_DIR / "qm9_causal_val.pkl"
        
    train_dataset = load_data(train_path)
    val_dataset = load_data(val_path)
    
    # Use config batch size
    batch_size = config.LAYER4_BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = HybridQuantumGNN(
        node_dim=config.NODE_FEATURE_DIM,
        n_qubits=config.N_QUBITS_BASE,
        n_rotation_layers=config.N_ROTATION_LAYERS,
        n_entangle_layers=config.N_ENTANGLE_LAYERS,
        entangle_mode=config.ENTANGLE_MODE,
        num_targets=len(config.QM9_TARGET_NAMES)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss() # Targets are already normalized
    
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "epochs": 0}
    
    model_save_path = config.MODELS_DIR / "best_qcgnn.pt"
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Predict all targets
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            target = batch.y.view(out.shape).float()
            
            loss = criterion(out, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_MAX_NORM)
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch:03d} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x.float(), batch.edge_index, batch.batch)
                target = batch.y.view(out.shape).float()
                loss = criterion(out, target)
                val_loss += loss.item() * batch.num_graphs
        val_loss /= len(val_loader.dataset)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"] += 1
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  -> New best model! Saving to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
            
    history_path = config.RESULTS_DIR / "quantum_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    train()
