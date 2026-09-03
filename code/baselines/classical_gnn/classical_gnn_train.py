import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pickle
import json
import time
import sys
from pathlib import Path

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

def train():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.N_EPOCHS)
    args = ap.parse_args()
    
    device = config.DEVICE
    print(f"Training on device: {device}")
    
    # Load FULL graphs from Layer 2
    train_path = config.LAYER2_DIR / "qm9_train.pkl"
    val_path = config.LAYER2_DIR / "qm9_val.pkl"
    
    print(f"Loading training data from {train_path}")
    train_dataset = load_data(train_path)
    print(f"Loading validation data from {val_path}")
    val_dataset = load_data(val_path)
    
    # Use standard batch size, possibly smaller if memory is an issue with full graphs, 
    # but 32 (config.LAYER4_BATCH_SIZE) should be fine for GENConv.
    batch_size = config.LAYER4_BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ClassicalGNN(
        node_dim=config.NODE_FEATURE_DIM,
        edge_dim=config.EDGE_FEATURE_DIM,
        hidden_dim=64,
        num_layers=3,
        num_targets=len(config.QM9_TARGET_NAMES)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "epochs": 0}
    
    model_save_path = config.MODELS_DIR / "best_classical_gnn.pt"
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Predict
            out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
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
                out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
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
            
    history_path = config.RESULTS_DIR / "classical_gnn_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    train()
