import torch
import sys
import pickle
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent))

import config
from experiments.experiment_D_full_qcgnn.full_qcgnn_model import FullQCGNN

def run_sanity_test():
    print("--- FULL GRAPH QCGNN SANITY TEST (EXPERIMENT D) ---")
    
    try:
        # 1. Full graph loads (From Layer 2)
        test_file = config.LAYER2_DIR / "qm9_train.pkl"
        print(f"Loading FULL molecular graph from: {test_file.name}")
        
        with open(test_file, "rb") as f:
            dataset = pickle.load(f)
            
        sample = dataset[0]
        
        # 2. Node features load
        x = sample.x.float().clone().detach().requires_grad_(True)
        # 3. Edge attributes load
        edge_index = sample.edge_index
        edge_attr = sample.edge_attr.float() if sample.edge_attr is not None else None
        
        y = sample.y
        batch = torch.zeros(x.size(0), dtype=torch.long)
        
        print(f"Number of nodes (FULL graph): {x.size(0)}")
        assert x.shape[1] == config.NODE_FEATURE_DIM, f"Expected node dim {config.NODE_FEATURE_DIM}"
        print(f"x shape: {list(x.shape)}")
        
        assert edge_index.shape[0] == 2, "Edge index should have shape [2, num_edges]"
        print(f"edge_index shape: {list(edge_index.shape)}")
        print(f"y shape: {list(y.shape)}")
        
        # Initialize Model directly using the validated QCGNN architecture
        model = FullQCGNN(
            node_dim=config.NODE_FEATURE_DIM,
            n_qubits=config.N_QUBITS_BASE,
            n_rotation_layers=config.N_ROTATION_LAYERS,
            n_entangle_layers=config.N_ENTANGLE_LAYERS,
            entangle_mode=config.ENTANGLE_MODE,
            num_targets=len(config.QM9_TARGET_NAMES)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # Manually trace representation and quantum shapes
        from torch_geometric.nn import global_mean_pool, global_max_pool
        h_mean = global_mean_pool(x, batch)
        h_max = global_max_pool(x, batch)
        h_c = torch.cat([h_mean, h_max], dim=1)
        z = model.classical_projection(h_c)
        q_out = model.qlayer(z)
        print(f"Graph representation shape: {list(h_c.shape)}")
        print(f"Quantum input shape: {list(z.shape)}")
        print(f"Configured number of qubits: {config.N_QUBITS_BASE}")
        print(f"Quantum output shape: {list(q_out.shape)}")
        
        # 4. Model forward pass works
        # 5. Output shape is [batch_size, 3]
        out = model(x, edge_index, batch)
        print(f"Final prediction shape: {list(out.shape)}")
        assert out.shape == (1, 3), f"Expected output shape [1, 3], got {list(out.shape)}"
        
        # 6. Loss can be calculated
        loss = torch.nn.functional.mse_loss(out, y.view(out.shape).float())
        print(f"Loss calculated: {loss.item():.4f}")
        
        # 7. Gradients exist
        loss.backward()
        
        grad_valid = (x.grad is not None) and torch.isfinite(x.grad).all() and \
                     (model.classical_projection.weight.grad is not None) and \
                     torch.isfinite(model.classical_projection.weight.grad).all()
                     
        if not grad_valid:
            raise ValueError("Gradients were not calculated properly or contained non-finite values.")
            
        # 8. One optimization step works
        optimizer.step()
            
        print("\nSANITY TEST: PASS")
        
    except Exception as e:
        print(f"\nSANITY TEST: FAIL")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_sanity_test()
