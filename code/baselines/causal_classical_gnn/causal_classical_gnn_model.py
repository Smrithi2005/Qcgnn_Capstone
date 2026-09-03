import torch
import torch.nn as nn
from torch_geometric.nn import GENConv, global_mean_pool, global_max_pool

class CausalClassicalGNN(nn.Module):
    def __init__(self, node_dim=9, edge_dim=4, hidden_dim=64, num_layers=3, num_targets=3, dropout=0.1):
        super().__init__()
        
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # Using GENConv as per paper specification, supporting edge features
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax', t=1.0, learn_t=True, 
                           msg_norm=True, learn_msg_scale=True, edge_dim=edge_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.dropout = dropout
        self.act = nn.SiLU()
        
        # Readout processes the concatenated mean and max poolings (64 + 64 = 128)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets)
        )
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Initial node encoding (h_i^(0) = W_enc x_i + b_enc)
        h = self.node_encoder(x)
        
        # Message passing layers (Residual GENConv block)
        for conv, norm in zip(self.convs, self.norms):
            h_residual = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = self.act(h)
            h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)
            h = h + h_residual
            
        # Global pooling (Concat(Mean(h), Max(h)))
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=1)
        
        # Final MLP Readout (y_hat = MLP(h_G))
        out = self.readout(h_graph)
        return out
