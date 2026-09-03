import torch
import numpy as np
from torch_geometric.data import Data

def prediction_gradient_attribution(model, data, min_atoms=3, max_atoms=12, auto_fraction=0.2):
    """
    Performs inference-time causal attribution using Prediction-Gradient Saliency.
    Instead of calculating the MSE loss against a known ground-truth (which doesn't exist at inference),
    this method calculates the gradient of the trained model's scalar prediction magnitude
    with respect to the input node features.
    
    This identifies the "Model-Attributed Important Atoms".
    """
    model.eval()
    
    # 1. Prepare input features for gradient tracking
    x = data.x.clone().detach().requires_grad_(True)
    edge_index = data.edge_index
    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    
    # 2. Forward pass through trained QCGNN
    pred = model(x, edge_index, batch)
    
    # 3. Sum the predictions to create a scalar loss-equivalent objective
    # (Summing HOMO + LUMO + GAP to find globally important topological features)
    P = pred.abs().sum()
    
    # 4. Backward pass to compute input saliency map
    P.backward()
    
    # 5. Calculate attribution score per atom (summing absolute gradients over the 9 features)
    attribution_scores = x.grad.abs().sum(dim=1).detach().cpu().numpy()
    
    # 6. Rank atoms by attribution score
    n_atoms = x.size(0)
    k = max(min_atoms, min(max_atoms, int(n_atoms * auto_fraction)))
    k = min(k, n_atoms)  # Ensure we don't pick more atoms than exist
    
    # Sort indices descending
    ranked_indices = np.argsort(attribution_scores)[::-1]
    kept_nodes = ranked_indices[:k].tolist()
    
    # 7. Construct the subgraph (G_c)
    kept_tensor = torch.tensor(kept_nodes, dtype=torch.long)
    subgraph_x = data.x[kept_tensor]
    
    # For QCGNN Layer 4, the edge_index is actually ignored via global_mean_pool, 
    # but we map it cleanly for scientific consistency if other layers need it.
    sub_edge_index = []
    for s, d in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()):
        if s in kept_nodes and d in kept_nodes:
            # Map original node IDs to subgraph node IDs
            sub_s = kept_nodes.index(s)
            sub_d = kept_nodes.index(d)
            sub_edge_index.append([sub_s, sub_d])
            
    if sub_edge_index:
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()
    else:
        sub_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
    causal_data = Data(x=subgraph_x, edge_index=sub_edge_index)
    
    return causal_data, kept_nodes, attribution_scores.tolist()
