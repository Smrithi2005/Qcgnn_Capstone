import torch
import sys
from pathlib import Path

# Add project root to path to import config
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent)) # qcgnn_project
sys.path.insert(0, str(_HERE.parent.parent))        # qcgnn_project/code

from layer4_quantum.quantum_model import HybridQuantumGNN
from layer4_quantum.quantum_circuit import build_qnode
import config
import pennylane as qml

def run_2_qubit_test():
    print("--- 2-QUBIT SANITY TEST ---")
    n_qubits = 2
    
    # Tiny circuit
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def tiny_circuit(inputs, weights):
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        
        # Parameterized layer
        qml.RY(weights[0, 0], wires=0)
        qml.RY(weights[0, 1], wires=1)
        
        qml.CNOT(wires=[0, 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    inputs = torch.tensor([[0.5, -0.3], [1.0, 0.0]], requires_grad=True)
    weights = torch.tensor([[0.1, 0.2]], requires_grad=True)
    
    try:
        # Create a torch layer to handle batching easily
        weight_shapes = {"weights": (1, 2)}
        qlayer = qml.qnn.TorchLayer(tiny_circuit, weight_shapes)
        # Initialize weights with our manual tensor to keep track
        qlayer.weights.data = weights
        
        output = qlayer(inputs)
        
        # Verify gradient
        loss = output.sum()
        loss.backward()
        
        grad_exists = (inputs.grad is not None) and (qlayer.weights.grad is not None)
        grad_finite = torch.isfinite(inputs.grad).all() and torch.isfinite(qlayer.weights.grad).all()
        
        print("Quantum simulator test: PASS")
        print(f"Qubit count: {n_qubits}")
        print(f"Measurement shape: {list(output.shape)}")
        if grad_exists and grad_finite:
            print("Gradient test: PASS")
        else:
            print("Gradient test: FAIL")
    except Exception as e:
        print("Quantum simulator test: FAIL")
        print(str(e))

def run_8_qubit_test():
    print("\n--- 8-QUBIT HYBRID MODEL TEST ---")
    n_qubits = config.N_QUBITS_BASE
    n_rot = config.N_ROTATION_LAYERS
    n_ent = config.N_ENTANGLE_LAYERS
    ent_mode = config.ENTANGLE_MODE
    
    model = HybridQuantumGNN(
        node_dim=config.NODE_FEATURE_DIM,
        n_qubits=n_qubits,
        n_rotation_layers=n_rot,
        n_entangle_layers=n_ent,
        entangle_mode=ent_mode,
        num_targets=len(config.QM9_TARGET_NAMES)
    )
    
    # Fake batch of 2 graphs, each with 5 atoms
    batch_size = 2
    n_atoms = 5
    x = torch.randn(batch_size * n_atoms, config.NODE_FEATURE_DIM, requires_grad=True)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ])
    # Duplicate edge_index for the second graph in the batch
    edge_index = torch.cat([edge_index, edge_index + n_atoms], dim=1)
    
    batch = torch.tensor([0]*n_atoms + [1]*n_atoms)
    
    try:
        output = model(x, edge_index, batch)
        
        loss = output.sum()
        loss.backward()
        
        grad_exists = (x.grad is not None)
        
        print("8-Qubit Model test: PASS")
        print(f"Output shape: {list(output.shape)}")
        if grad_exists:
            print("Gradient test: PASS")
        else:
            print("Gradient test: FAIL")
    except Exception as e:
        print("8-Qubit Model test: FAIL")
        print(str(e))

if __name__ == "__main__":
    run_2_qubit_test()
    run_8_qubit_test()
