import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
import pennylane as qml

from .quantum_circuit import build_qnode

class HybridQuantumGNN(nn.Module):
    def __init__(self, node_dim=9, n_qubits=8, n_rotation_layers=2, n_entangle_layers=1, entangle_mode='linear', num_targets=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_rotation_layers = n_rotation_layers
        self.n_entangle_layers = n_entangle_layers
        
        # Classical feature projection
        # We concatenate mean and max pooling, so input dim is 2 * node_dim
        self.classical_projection = nn.Linear(node_dim * 2, n_qubits)
        
        # Quantum parameters
        weight_shapes = {"weights": (n_rotation_layers, n_qubits)}
        self.qnode = build_qnode(n_qubits, n_rotation_layers, n_entangle_layers, entangle_mode)
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        
        # Classical readout
        # MLP: Linear -> ReLU -> Linear -> 3 outputs (HOMO, LUMO, GAP)
        self.classical_readout = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, num_targets)
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [N_c, 9]
            edge_index: Edge indices [2, E_c]
            batch: Batch indices for nodes
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # 1. Fixed-dimensional representation of causal graph
        h_mean = global_mean_pool(x, batch)
        h_max = global_max_pool(x, batch)
        h_c = torch.cat([h_mean, h_max], dim=1) # [batch_size, 18]
        
        # 2. Classical projection
        z = self.classical_projection(h_c) # [batch_size, N_QUBITS]
        
        # 3. Quantum encoding & circuit & measurement
        # TorchLayer automatically handles batching over the input `z`
        q_out = self.qlayer(z) # [batch_size, N_QUBITS]
        
        # 4. Classical readout
        out = self.classical_readout(q_out) # [batch_size, 3]
        return out
