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
