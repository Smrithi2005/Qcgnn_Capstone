import pennylane as qml
import torch

def build_qnode(n_qubits, n_rotation_layers, n_entangle_layers, entangle_mode='linear'):
    """
    Constructs a PennyLane QNode for the hybrid QCGNN model.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_circuit(inputs, weights):
        """
        Args:
            inputs: Tensor of shape (n_qubits) containing angle encodings.
            weights: Tensor of shape (n_rotation_layers + 1, n_qubits) containing rotation angles.
        """
        # 1. Angle Encoding
        # Manual Ry gates per the specification: Ry(z_i)
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
            
        # 2. Parameterized Rotation and Entanglement Layers
        # The prompt says: "Input encoding -> Parameterized Ry/Rz layer -> Linear CNOT entanglement -> Parameterized Ry/Rz layer -> Measurement"
        # If N_ROTATION_LAYERS=2, we want exactly that.
        
        # First parameterized layer
        for i in range(n_qubits):
            qml.RY(weights[0, i], wires=i)
            
        # Optional additional layers depending on n_rotation_layers
        # To make it generalize to n_rotation_layers, we can interleave rotation and entanglement.
        # But for N_ROTATION_LAYERS=2, N_ENTANGLE_LAYERS=1, the standard hardware-efficient ansatz is:
        # for L in range(num_layers):
        #    Rotations
        #    Entanglement
        
        # Wait, the prompt specifically says:
        # Input encoding -> Parameterized Ry/Rz layer -> Linear CNOT entanglement -> Parameterized Ry/Rz layer -> Measurement
        
        layer_idx = 0
        while layer_idx < n_rotation_layers:
            # We already did one rotation layer if layer_idx == 0
            if layer_idx > 0:
                for i in range(n_qubits):
                    qml.RY(weights[layer_idx, i], wires=i)
            
            # Entanglement
            if layer_idx < n_entangle_layers:
                if entangle_mode == 'linear':
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                else:
                    raise ValueError(f"Unsupported entangle_mode: {entangle_mode}")
            
            layer_idx += 1
            
        # Wait, if layer_idx goes up to n_rotation_layers-1, we need to make sure we match the diagram exactly.
        # Let's simplify and follow the standard strongly entangling or basic entangling layers:
        # The weights will have shape (n_rotation_layers, n_qubits).
        
        # Let's rebuild the loop:
        for layer in range(n_rotation_layers):
            # Rotations
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
                
            # Entanglement (only apply up to n_entangle_layers)
            if layer < n_entangle_layers:
                if entangle_mode == 'linear':
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
        
        # 3. Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
    return quantum_circuit
