import torch
import pickle
import sys
from pathlib import Path

# Setup paths to ensure we can import the existing codebase
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent)) # qcgnn_project
sys.path.insert(0, str(_HERE.parent.parent))        # qcgnn_project/code

import config
from layer4_quantum.quantum_model import HybridQuantumGNN
from torch_geometric.nn import global_mean_pool, global_max_pool

def run_integration_test():
    print("--- LAYER 3 TO LAYER 4 REAL DATA INTEGRATION TEST ---")
    
    try:
        # 1. Find/load one real Layer 3 .pkl file
        test_file = config.LAYER3_DIR / "qm9_causal_homo_train.pkl"
        if not test_file.exists():
            test_file = config.LAYER3_DIR / "qm9_causal_train.pkl"
            
        print(f"Source file: {test_file.name}")
        
        with open(test_file, "rb") as f:
            dataset = pickle.load(f)
            
        # 2. Load one real PyTorch Geometric Data object
        sample = dataset[0]
        
        # Enable gradients on the input features to verify end-to-end backward pass
        x = sample.x.float().clone().detach().requires_grad_(True)
        edge_index = sample.edge_index
        edge_attr = getattr(sample, 'edge_attr', None)
        pos = getattr(sample, 'pos', None)
        y = sample.y
        batch = torch.zeros(x.size(0), dtype=torch.long)
        
        # 3. Print the actual shapes
        print(f"Number of causal nodes: {x.size(0)}")
        print(f"x.shape: {list(x.shape)}")
        print(f"edge_index.shape: {list(edge_index.shape)}")
        print(f"edge_attr.shape: {list(edge_attr.shape) if edge_attr is not None else 'None'}")
        print(f"pos.shape: {list(pos.shape) if pos is not None else 'None'}")
        print(f"y.shape: {list(y.shape)}")
        
        # 4. Use the EXISTING Layer 4 model
        model = HybridQuantumGNN(
            node_dim=config.NODE_FEATURE_DIM,
            n_qubits=config.N_QUBITS_BASE,
            n_rotation_layers=config.N_ROTATION_LAYERS,
            n_entangle_layers=config.N_ENTANGLE_LAYERS,
            entangle_mode=config.ENTANGLE_MODE,
            num_targets=len(config.QM9_TARGET_NAMES)
        )
        
        # 5. Run ONE forward pass (and extract intermediates to print shapes safely)
        h_mean = global_mean_pool(x, batch)
        h_max = global_max_pool(x, batch)
        h_c = torch.cat([h_mean, h_max], dim=1)
        z = model.classical_projection(h_c)
        q_out = model.qlayer(z)
        
        out = model(x, edge_index, batch)
        
        # 6. Print intermediate and final shapes
        print(f"Graph representation shape: {list(h_c.shape)}")
        print(f"Configured number of qubits: {config.N_QUBITS_BASE}")
        print(f"Quantum input shape: {list(z.shape)}")
        print(f"Quantum output shape: {list(q_out.shape)}")
        print(f"Final prediction shape: {list(out.shape)}")
        print(f"Target shape: {list(y.shape)}")
        
        # 7. Verify all tensors are finite
        if not torch.isfinite(out).all():
            raise ValueError("Output contains NaN or infinite values.")
            
        # 8 & 9. Verify tensor dimensions and 3 output values
        if out.shape[-1] != 3:
            raise ValueError(f"Final prediction has {out.shape[-1]} values, expected 3 (HOMO, LUMO, GAP).")
        if y.shape[-1] != 3:
            raise ValueError(f"Target 'y' has {y.shape[-1]} values, expected 3.")
            
        # 10. Perform backward/gradient check through the real-data path
        loss = out.sum()
        loss.backward()
        
        grad_valid = (x.grad is not None) and torch.isfinite(x.grad).all() and \
                     (model.classical_projection.weight.grad is not None) and \
                     torch.isfinite(model.classical_projection.weight.grad).all()
                     
        if not grad_valid:
            raise ValueError("Gradient calculation failed or produced non-finite values.")
            
        # 11. Print PASS
        print("\nINTEGRATION TEST: PASS")
        
    except Exception as e:
        # 12. Print FAIL and the real error
        print(f"\nINTEGRATION TEST: FAIL")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_integration_test()
