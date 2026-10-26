import json
from pathlib import Path

import numpy as np
import qiskit
import torch
from qiskit.circuit import ClassicalRegister
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Pauli, Statevector
from qiskit.transpiler import CouplingMap

from ibm_hardware_adapter import (
    BASELINE_PATH,
    CHECKPOINT_PATH,
    N_QUBITS,
    load_trained_inputs,
    prepare_hardware_circuit,
)
from measurement_utils import counts_to_z_expectations


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "results" / "quantum_ibm_adapter_dry_run.json"
SHOTS = 8192
TOLERANCE = 1e-8
SAMPLE_TOLERANCE = 0.08


def main():
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite dry-run result: {RESULT_PATH}")
    angles, weights, readout = load_trained_inputs()
    coupling_map = CouplingMap.from_line(N_QUBITS)
    prepared = prepare_hardware_circuit(
        angles,
        weights,
        coupling_map=coupling_map,
        basis_gates=["rz", "sx", "x", "cx"],
    )
    circuit = prepared["circuit"]
    if not any(
        isinstance(register, ClassicalRegister) and register.name == "meas"
        for register in circuit.cregs
    ):
        raise ValueError("Prepared circuit is missing the named meas register.")

    reference = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    logical_to_physical = prepared["logical_to_physical"]
    reference_expectations = np.asarray(
        [
            reference.expectation_value(
                Pauli(
                    "I" * (N_QUBITS - 1 - logical_to_physical[logical])
                    + "Z"
                    + "I" * logical_to_physical[logical]
                )
            ).real
            for logical in range(N_QUBITS)
        ],
        dtype=np.float64,
    )
    result = StatevectorSampler(seed=1234).run([circuit], shots=SHOTS).result()
    bit_array = getattr(result[0].data, "meas", None)
    if bit_array is None or not hasattr(bit_array, "get_counts"):
        raise ValueError("Local sampler result lacks the meas register.")
    counts = bit_array.get_counts()
    decoded = counts_to_z_expectations(counts, N_QUBITS)
    sample_difference = np.abs(decoded - reference_expectations)

    hidden = np.maximum(
        0.0,
        reference_expectations @ readout["classical_readout.0.weight"].T
        + readout["classical_readout.0.bias"],
    )
    standardized = (
        hidden @ readout["classical_readout.2.weight"].T
        + readout["classical_readout.2.bias"]
    )
    checks = {
        "baseline_loaded": BASELINE_PATH.exists(),
        "checkpoint_loaded": CHECKPOINT_PATH.exists(),
        "layout_complete": len(logical_to_physical) == N_QUBITS,
        "named_measurement_register": True,
        "shot_total_matches": sum(counts.values()) == SHOTS,
        "decoded_shape_is_eight": decoded.shape == (N_QUBITS,),
        "decoded_values_finite": bool(np.isfinite(decoded).all()),
        "reference_finite": bool(np.isfinite(reference_expectations).all()),
        "sample_within_tolerance": bool(
            np.max(sample_difference) <= SAMPLE_TOLERANCE
        ),
        "submission_not_requested": True,
    }
    report = {
        "qiskit_version": qiskit.__version__,
        "runtime_version": "not installed in venv; inspected separately as 0.51.0 in ibm_test_venv",
        "mode": "local dry-run; no IBM Runtime service or sampler",
        "baseline": str(BASELINE_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "n_qubits": N_QUBITS,
        "shots": SHOTS,
        "logical_to_physical": logical_to_physical,
        "physical_to_classical": {
            str(key): value for key, value in prepared["physical_to_classical"].items()
        },
        "counts": counts,
        "reference_expectations": reference_expectations.tolist(),
        "decoded_expectations": decoded.tolist(),
        "sample_absolute_differences": sample_difference.tolist(),
        "sample_tolerance": SAMPLE_TOLERANCE,
        "classical_readout_input": reference_expectations.tolist(),
        "standardized_predictions": standardized.tolist(),
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    with RESULT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Adapter dry-run result saved to {RESULT_PATH}")
    print(f"Logical-to-physical mapping: {logical_to_physical}")
    print(f"Validation: {'PASS' if report['all_passed'] else 'FAIL'}")
    if not report["all_passed"]:
        raise RuntimeError("IBM adapter dry-run validation failed.")


if __name__ == "__main__":
    main()
