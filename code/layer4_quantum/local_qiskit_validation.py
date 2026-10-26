import json
from pathlib import Path

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "results" / "quantum_simulator_baseline_corrected.json"
CHECKPOINT_PATH = ROOT / "models" / "best_qcgnn_corrected.pt"
RESULT_PATH = ROOT / "results" / "quantum_qiskit_equivalence_corrected.json"
N_QUBITS = 8
TOLERANCE = 1e-6


def finite(values, label):
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values.")
    return array


def main():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Simulator baseline not found: {BASELINE_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Corrected checkpoint not found: {CHECKPOINT_PATH}")
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite validation result: {RESULT_PATH}")

    with BASELINE_PATH.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    if baseline.get("target_order") != ["HOMO", "LUMO", "GAP"]:
        raise ValueError(f"Unexpected target order: {baseline.get('target_order')}")
    angles = finite(baseline["input_angles"], "baseline input angles")
    pennylane_measurements = finite(
        baseline["quantum_measurements"], "PennyLane measurements"
    )
    if angles.shape != (N_QUBITS,) or pennylane_measurements.shape != (N_QUBITS,):
        raise ValueError("The baseline must contain exactly eight angles and measurements.")

    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    required_keys = {
        "qlayer.weights",
        "classical_readout.0.weight",
        "classical_readout.0.bias",
        "classical_readout.2.weight",
        "classical_readout.2.bias",
    }
    missing_keys = required_keys.difference(state_dict)
    if missing_keys:
        raise KeyError(f"Checkpoint is missing required keys: {sorted(missing_keys)}")

    quantum_weights = state_dict["qlayer.weights"].detach().cpu().numpy()
    if quantum_weights.shape != (2, N_QUBITS):
        raise ValueError(f"Unexpected quantum weight shape: {quantum_weights.shape}")
    quantum_weights = finite(quantum_weights, "checkpoint quantum weights")

    circuit = QuantumCircuit(N_QUBITS)
    for qubit in range(N_QUBITS):
        circuit.ry(float(angles[qubit]), qubit)
    for qubit in range(N_QUBITS):
        circuit.ry(float(quantum_weights[0, qubit]), qubit)
    for qubit in range(N_QUBITS - 1):
        circuit.cx(qubit, qubit + 1)
    for qubit in range(N_QUBITS):
        circuit.ry(float(quantum_weights[1, qubit]), qubit)

    statevector = Statevector.from_instruction(circuit)
    qiskit_measurements = []
    pauli_labels = []
    for qubit in range(N_QUBITS):
        label = "I" * (N_QUBITS - 1 - qubit) + "Z" + "I" * qubit
        pauli_labels.append(label)
        qiskit_measurements.append(statevector.expectation_value(Pauli(label)).real)
    qiskit_measurements = finite(qiskit_measurements, "Qiskit measurements")

    absolute_differences = np.abs(qiskit_measurements - pennylane_measurements)
    max_difference = float(np.max(absolute_differences))
    equivalence_passed = bool(max_difference <= TOLERANCE)

    readout_0_weight = state_dict["classical_readout.0.weight"].numpy()
    readout_0_bias = state_dict["classical_readout.0.bias"].numpy()
    readout_2_weight = state_dict["classical_readout.2.weight"].numpy()
    readout_2_bias = state_dict["classical_readout.2.bias"].numpy()
    hidden = np.maximum(
        0.0, qiskit_measurements @ readout_0_weight.T + readout_0_bias
    )
    qiskit_standardized = finite(
        hidden @ readout_2_weight.T + readout_2_bias,
        "Qiskit standardized predictions",
    )
    baseline_standardized = finite(
        baseline["standardized_predictions"], "baseline standardized predictions"
    )
    prediction_difference = np.abs(qiskit_standardized - baseline_standardized)

    report = {
        "baseline": str(BASELINE_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "simulator": "Qiskit Statevector.from_instruction",
        "qiskit_version": __import__("qiskit").__version__,
        "n_qubits": N_QUBITS,
        "gate_sequence": [
            "RY(input_angles[qubit]) for qubit 0..7",
            "RY(quantum_weights[0, qubit]) for qubit 0..7",
            "CX(qubit, qubit + 1) for qubit 0..6",
            "RY(quantum_weights[1, qubit]) for qubit 0..7",
            "Pauli-Z expectation for qubit 0..7",
        ],
        "measurement_order": "logical qubit 0 through 7",
        "pauli_labels": pauli_labels,
        "tolerance": TOLERANCE,
        "pennylane_measurements": pennylane_measurements.tolist(),
        "qiskit_measurements": qiskit_measurements.tolist(),
        "absolute_differences": absolute_differences.tolist(),
        "max_absolute_difference": max_difference,
        "equivalence_passed": equivalence_passed,
        "baseline_standardized_predictions": baseline_standardized.tolist(),
        "qiskit_standardized_predictions": qiskit_standardized.tolist(),
        "standardized_prediction_absolute_differences": prediction_difference.tolist(),
        "target_order": ["HOMO", "LUMO", "GAP"],
    }
    with RESULT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Qiskit result saved to {RESULT_PATH}")
    print(f"Maximum absolute measurement difference: {max_difference:.12g}")
    print(f"Equivalence: {'PASS' if equivalence_passed else 'FAIL'}")
    if not equivalence_passed:
        raise RuntimeError(
            f"Qiskit/PennyLane equivalence failed tolerance {TOLERANCE:g}."
        )


if __name__ == "__main__":
    main()
