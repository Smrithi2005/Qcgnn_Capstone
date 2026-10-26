import json
from pathlib import Path

import numpy as np
import qiskit
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Pauli, Statevector
from qiskit.transpiler import CouplingMap

from measurement_utils import counts_to_z_expectations


ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "results" / "quantum_simulator_baseline_corrected.json"
CHECKPOINT_PATH = ROOT / "models" / "best_qcgnn_corrected.pt"
RESULT_PATH = ROOT / "results" / "quantum_transpiled_layout_test.json"
N_QUBITS = 8
REGISTER_NAME = "meas"
SHOTS = 8192
STATEVECTOR_TOLERANCE = 1e-8
SAMPLE_TOLERANCE = 0.08


def finite(values, label):
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values.")
    return array


def build_circuit(angles, weights, with_measurements):
    circuit = QuantumCircuit(N_QUBITS)
    for qubit in range(N_QUBITS):
        circuit.ry(float(angles[qubit]), qubit)
    for qubit in range(N_QUBITS):
        circuit.ry(float(weights[0, qubit]), qubit)
    for qubit in range(N_QUBITS - 1):
        circuit.cx(qubit, qubit + 1)
    for qubit in range(N_QUBITS):
        circuit.ry(float(weights[1, qubit]), qubit)
    if with_measurements:
        register = ClassicalRegister(N_QUBITS, REGISTER_NAME)
        circuit.add_register(register)
        circuit.measure(range(N_QUBITS), range(N_QUBITS))
    return circuit


def extract_mapping(transpiled_circuit):
    layout = transpiled_circuit.layout
    if layout is None:
        raise ValueError("Transpiled circuit did not provide a layout.")
    logical_to_physical = list(layout.final_index_layout(filter_ancillas=True))
    if len(logical_to_physical) != N_QUBITS or sorted(logical_to_physical) != list(range(N_QUBITS)):
        raise ValueError(f"Unexpected logical-to-physical layout: {logical_to_physical}")
    measurement_map = {}
    for instruction in transpiled_circuit.data:
        if instruction.operation.name != "measure":
            continue
        physical = transpiled_circuit.find_bit(instruction.qubits[0]).index
        classical = transpiled_circuit.find_bit(instruction.clbits[0]).index
        if physical in measurement_map:
            raise ValueError(f"Multiple measurements found for physical qubit {physical}.")
        measurement_map[physical] = classical
    if set(measurement_map) != set(range(N_QUBITS)):
        raise ValueError(f"Incomplete physical measurement map: {measurement_map}")
    return logical_to_physical, measurement_map


def remap_counts_to_logical(counts, logical_to_physical, measurement_map):
    logical_counts = {}
    for bitstring, count in counts.items():
        logical_bits = [
            bitstring[-1 - measurement_map[logical_to_physical[logical]]]
            for logical in range(N_QUBITS)
        ]
        logical_bitstring = "".join(reversed(logical_bits))
        logical_counts[logical_bitstring] = logical_counts.get(logical_bitstring, 0) + count
    return logical_counts


def main():
    for path, label in (
        (BASELINE_PATH, "simulator baseline"),
        (CHECKPOINT_PATH, "corrected checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite validation result: {RESULT_PATH}")

    with BASELINE_PATH.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    angles = finite(baseline["input_angles"], "input angles")
    if angles.shape != (N_QUBITS,):
        raise ValueError("Baseline must contain exactly eight input angles.")

    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    weights = finite(
        state_dict["qlayer.weights"].detach().cpu().numpy(),
        "trained quantum weights",
    )
    if weights.shape != (2, N_QUBITS):
        raise ValueError(f"Unexpected trained weight shape: {weights.shape}")

    reference_circuit = build_circuit(angles, weights, with_measurements=False)
    reference_state = Statevector.from_instruction(reference_circuit)
    reference_expectations = finite(
        [
            reference_state.expectation_value(
                Pauli("I" * (N_QUBITS - 1 - qubit) + "Z" + "I" * qubit)
            ).real
            for qubit in range(N_QUBITS)
        ],
        "reference expectations",
    )

    measured_circuit = build_circuit(angles, weights, with_measurements=True)
    coupling_map = CouplingMap.from_line(N_QUBITS)
    transpiled_circuit = transpile(
        measured_circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        coupling_map=coupling_map,
        optimization_level=1,
        seed_transpiler=11,
    )
    logical_to_physical, measurement_map = extract_mapping(transpiled_circuit)

    no_measurements = transpiled_circuit.remove_final_measurements(inplace=False)
    transpiled_state = Statevector.from_instruction(no_measurements)
    transpiled_expectations = finite(
        [
            transpiled_state.expectation_value(
                Pauli("I" * (N_QUBITS - 1 - logical_to_physical[logical]) + "Z"
                      + "I" * logical_to_physical[logical])
            ).real
            for logical in range(N_QUBITS)
        ],
        "transpiled expectations",
    )
    statevector_differences = np.abs(transpiled_expectations - reference_expectations)

    sampler_result = StatevectorSampler(seed=1234).run(
        [transpiled_circuit], shots=SHOTS
    ).result()
    bit_array = getattr(sampler_result[0].data, REGISTER_NAME, None)
    if bit_array is None or not hasattr(bit_array, "get_counts"):
        raise ValueError(f"Transpiled result lacks register '{REGISTER_NAME}'.")
    physical_counts = bit_array.get_counts()
    logical_counts = remap_counts_to_logical(
        physical_counts, logical_to_physical, measurement_map
    )
    decoded_expectations = finite(
        counts_to_z_expectations(logical_counts, N_QUBITS),
        "decoded sampled expectations",
    )
    sample_differences = np.abs(decoded_expectations - reference_expectations)

    checks = {
        "layout_complete": True,
        "all_logical_qubits_measured": True,
        "statevector_mapping_matches": bool(
            np.max(statevector_differences) <= STATEVECTOR_TOLERANCE
        ),
        "counts_shots_match": int(sum(physical_counts.values())) == SHOTS,
        "decoded_shape_is_eight": decoded_expectations.shape == (N_QUBITS,),
        "decoded_values_finite": bool(np.isfinite(decoded_expectations).all()),
        "sample_within_tolerance": bool(np.max(sample_differences) <= SAMPLE_TOLERANCE),
    }
    all_passed = all(checks.values())
    report = {
        "qiskit_version": qiskit.__version__,
        "baseline": str(BASELINE_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "n_qubits": N_QUBITS,
        "shots": SHOTS,
        "coupling_map": [list(edge) for edge in coupling_map.get_edges()],
        "optimization_level": 1,
        "basis_gates": ["rz", "sx", "x", "cx"],
        "logical_to_physical": logical_to_physical,
        "physical_to_classical_bit": {
            str(physical): classical
            for physical, classical in measurement_map.items()
        },
        "reference_expectations": reference_expectations.tolist(),
        "transpiled_statevector_expectations": transpiled_expectations.tolist(),
        "statevector_absolute_differences": statevector_differences.tolist(),
        "physical_counts": physical_counts,
        "logical_counts": logical_counts,
        "decoded_expectations": decoded_expectations.tolist(),
        "sample_absolute_differences": sample_differences.tolist(),
        "statevector_tolerance": STATEVECTOR_TOLERANCE,
        "sample_tolerance": SAMPLE_TOLERANCE,
        "checks": checks,
        "all_passed": all_passed,
    }
    with RESULT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Transpiled layout result saved to {RESULT_PATH}")
    print(f"Logical-to-physical mapping: {logical_to_physical}")
    print(f"Maximum statevector difference: {np.max(statevector_differences):.12g}")
    print(f"Maximum sampled difference: {np.max(sample_differences):.12g}")
    print(f"Validation: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise RuntimeError("Transpiled layout validation failed.")


if __name__ == "__main__":
    main()
