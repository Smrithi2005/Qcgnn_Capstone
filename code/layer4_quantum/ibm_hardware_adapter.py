import json
import pickle
from pathlib import Path

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister


ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "results" / "quantum_simulator_baseline_corrected.json"
CHECKPOINT_PATH = ROOT / "models" / "best_qcgnn_corrected.pt"
NORM_PATH = ROOT / "data" / "qm9" / "layer2" / "norm_stats.pkl"
N_QUBITS = 8
REGISTER_NAME = "meas"
HARTREE_TO_EV = 27.211386245988


def _finite(values, label):
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values.")
    return array


def load_trained_inputs(
    baseline_path=BASELINE_PATH, checkpoint_path=CHECKPOINT_PATH
):
    if not baseline_path.exists():
        raise FileNotFoundError(f"Simulator baseline not found: {baseline_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Corrected checkpoint not found: {checkpoint_path}")
    with baseline_path.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    angles = _finite(baseline["input_angles"], "input angles")
    if angles.shape != (N_QUBITS,):
        raise ValueError("Baseline must contain exactly eight input angles.")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = _finite(
        state_dict["qlayer.weights"].detach().cpu().numpy(),
        "trained quantum weights",
    )
    if weights.shape != (2, N_QUBITS):
        raise ValueError(f"Unexpected trained weight shape: {weights.shape}")
    readout = {
        key: state_dict[key].detach().cpu().numpy()
        for key in (
            "classical_readout.0.weight",
            "classical_readout.0.bias",
            "classical_readout.2.weight",
            "classical_readout.2.bias",
        )
    }
    return angles, weights, readout


def load_normalization_stats(norm_path=NORM_PATH):
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization statistics not found: {norm_path}")
    with norm_path.open("rb") as handle:
        norm = pickle.load(handle)
    if norm.get("targets") != ["homo", "lumo", "gap"]:
        raise ValueError(f"Unexpected normalization target order: {norm.get('targets')}")
    mean = _finite(norm["mean"], "normalization mean")
    std = _finite(norm["std"], "normalization standard deviation")
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Normalization statistics must each contain three values.")
    return mean, std


def build_qgnn_circuit(angles, weights):
    angles = _finite(angles, "input angles")
    weights = _finite(weights, "trained quantum weights")
    if angles.shape != (N_QUBITS,) or weights.shape != (2, N_QUBITS):
        raise ValueError("Expected angles shape (8,) and weights shape (2, 8).")

    circuit = QuantumCircuit(N_QUBITS)
    for qubit in range(N_QUBITS):
        circuit.ry(float(angles[qubit]), qubit)
    for qubit in range(N_QUBITS):
        circuit.ry(float(weights[0, qubit]), qubit)
    for qubit in range(N_QUBITS - 1):
        circuit.cx(qubit, qubit + 1)
    for qubit in range(N_QUBITS):
        circuit.ry(float(weights[1, qubit]), qubit)

    register = ClassicalRegister(N_QUBITS, REGISTER_NAME)
    circuit.add_register(register)
    circuit.measure(range(N_QUBITS), range(N_QUBITS))
    return circuit


def inspect_measurement_layout(circuit):
    layout = circuit.layout
    if layout is None:
        raise ValueError("Transpiled circuit did not provide a layout.")
    logical_to_physical = list(layout.final_index_layout(filter_ancillas=True))
    if len(logical_to_physical) != N_QUBITS:
        raise ValueError(f"Unexpected logical-to-physical layout: {logical_to_physical}")

    physical_to_classical = {}
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        physical = circuit.find_bit(instruction.qubits[0]).index
        classical = circuit.find_bit(instruction.clbits[0]).index
        if physical in physical_to_classical:
            raise ValueError(f"Physical qubit {physical} is measured more than once.")
        physical_to_classical[physical] = classical
    if set(physical_to_classical) != set(logical_to_physical):
        raise ValueError("Every logical qubit must have one measured physical qubit.")
    return logical_to_physical, physical_to_classical


def remap_counts_to_logical(counts, logical_to_physical, physical_to_classical):
    if len(logical_to_physical) != N_QUBITS:
        raise ValueError("Expected a mapping for all eight logical qubits.")
    logical_counts = {}
    for bitstring, count in counts.items():
        compact = bitstring.replace(" ", "")
        logical_bits = [
            compact[-1 - physical_to_classical[logical_to_physical[logical]]]
            for logical in range(N_QUBITS)
        ]
        logical_bitstring = "".join(reversed(logical_bits))
        logical_counts[logical_bitstring] = (
            logical_counts.get(logical_bitstring, 0) + count
        )
    return logical_counts


def predictions_from_measurements(measurements, readout, mean, std):
    measurements = _finite(measurements, "quantum measurements")
    if measurements.shape != (N_QUBITS,):
        raise ValueError("Expected exactly eight quantum measurements.")
    hidden = np.maximum(
        0.0,
        measurements @ readout["classical_readout.0.weight"].T
        + readout["classical_readout.0.bias"],
    )
    standardized = _finite(
        hidden @ readout["classical_readout.2.weight"].T
        + readout["classical_readout.2.bias"],
        "standardized predictions",
    )
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Prediction normalization statistics must have shape (3,).")
    predictions_ev = _finite(
        (standardized * std + mean) * HARTREE_TO_EV,
        "eV predictions",
    )
    return standardized, predictions_ev


def prepare_hardware_circuit(
    angles,
    weights,
    backend=None,
    coupling_map=None,
    basis_gates=None,
    optimization_level=1,
    seed_transpiler=11,
):
    if backend is None and coupling_map is None:
        raise ValueError("Provide a backend or an explicit local coupling_map.")
    circuit = build_qgnn_circuit(angles, weights)
    transpile_kwargs = {
        "optimization_level": optimization_level,
        "seed_transpiler": seed_transpiler,
    }
    if backend is not None:
        transpiled = transpile(circuit, backend=backend, **transpile_kwargs)
    else:
        transpile_kwargs["coupling_map"] = coupling_map
        if basis_gates is not None:
            transpile_kwargs["basis_gates"] = basis_gates
        transpiled = transpile(circuit, **transpile_kwargs)
    logical_to_physical, physical_to_classical = inspect_measurement_layout(transpiled)
    return {
        "circuit": transpiled,
        "logical_to_physical": logical_to_physical,
        "physical_to_classical": physical_to_classical,
        "shots": None,
    }


def submit_prepared_circuit(prepared, sampler, shots, confirmation):
    if confirmation != "SUBMIT":
        raise PermissionError("Hardware submission requires confirmation='SUBMIT'.")
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if sampler is None:
        raise ValueError("An explicitly supplied IBM Runtime SamplerV2 is required.")
    return sampler.run([prepared["circuit"]], shots=shots)
