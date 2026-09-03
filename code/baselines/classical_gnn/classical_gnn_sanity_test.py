import torch
import sys
import pickle
from pathlib import Path

# Setup paths to ensure we can import the existing codebase
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent)) # qcgnn_project
sys.path.insert(0, str(_HERE.parent.parent.parent))        # qcgnn_project/code

import config
from baselines.classical_gnn.classical_gnn_model import ClassicalGNN

def run_sanity_test():
    print("--- CLASSICAL GNN SANITY TEST ---")
    
    try:
        # 1. Dataset loads correctly
        test_file = config.LAYER2_DIR / "qm9_train.pkl"
        print(f"Loading FULL molecular graph from: {test_file.name}")
        
        with open(test_file, "rb") as f:
            dataset = pickle.load(f)
            
        # 2. A full molecular graph is loaded
        sample = dataset[0]
        
        x = sample.x.float().clone().detach().requires_grad_(True)
        edge_index = sample.edge_index
        edge_attr = sample.edge_attr.float()
        y = sample.y
        batch = torch.zeros(x.size(0), dtype=torch.long)
        
        print(f"Number of nodes (FULL graph): {x.size(0)}")
        
        # 3. Node features have expected dimensions
        assert x.shape[1] == config.NODE_FEATURE_DIM, f"Expected node dim {config.NODE_FEATURE_DIM}, got {x.shape[1]}"
        print(f"x shape: {list(x.shape)}")
        
        # 4. Edge index is valid
        assert edge_index.shape[0] == 2, "Edge index should have shape [2, num_edges]"
        print(f"edge_index shape: {list(edge_index.shape)}")
        
        # 5. Edge attributes are valid
        assert edge_attr.shape[1] == config.EDGE_FEATURE_DIM, f"Expected edge dim {config.EDGE_FEATURE_DIM}, got {edge_attr.shape[1]}"
        print(f"edge_attr shape: {list(edge_attr.shape)}")
        print(f"y shape: {list(y.shape)}")
        
        # Initialize Model
        model = ClassicalGNN(
            node_dim=config.NODE_FEATURE_DIM,
            edge_dim=config.EDGE_FEATURE_DIM,
            hidden_dim=64,
            num_layers=3,
            num_targets=len(config.QM9_TARGET_NAMES)
        )
        
        # 6. Model forward pass works
        # 7. Output shape is [batch_size, 3]
        out = model(x, edge_index, edge_attr, batch)
        print(f"Output shape: {list(out.shape)}")
        assert out.shape == (1, 3), f"Expected output shape [1, 3], got {list(out.shape)}"
        
        # 8. Loss can be calculated
        loss = torch.nn.functional.mse_loss(out, y.view(out.shape).float())
        print(f"Loss calculated: {loss.item():.4f}")
        
        # 9. Backpropagation works
        loss.backward()
        
        # 10. Model parameters actually receive gradients
        grad_valid = (x.grad is not None) and torch.isfinite(x.grad).all() and \
                     (model.node_encoder.weight.grad is not None) and \
                     torch.isfinite(model.node_encoder.weight.grad).all()
                     
        if not grad_valid:
            raise ValueError("Gradients were not calculated properly or contained non-finite values.")
            
        print("\nSANITY TEST: PASS")
        
    except Exception as e:
        print(f"\nSANITY TEST: FAIL")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_sanity_test()
