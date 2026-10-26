import json
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.primitives import StatevectorSampler

from measurement_utils import counts_to_z_expectations


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "results" / "quantum_sampler_result_test.json"
N_QUBITS = 8
REGISTER_NAME = "meas"
SHOTS = 128


def extract_counts(result, register_name):
    try:
        data = result[0].data
    except (IndexError, AttributeError, TypeError) as error:
        raise ValueError("Sampler result does not contain a first publication.") from error
    bit_array = getattr(data, register_name, None)
    if bit_array is None or not hasattr(bit_array, "get_counts"):
        raise ValueError(
            f"Sampler result does not contain a measurable register named '{register_name}'."
        )
    counts = bit_array.get_counts()
    if not counts:
        raise ValueError(f"Sampler register '{register_name}' contains no counts.")
    return counts


def main():
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite test result: {RESULT_PATH}")

    circuit = QuantumCircuit(N_QUBITS)
    register = ClassicalRegister(N_QUBITS, REGISTER_NAME)
    circuit.add_register(register)
    for qubit in (0, 1, 7):
        circuit.x(qubit)
    circuit.measure(range(N_QUBITS), range(N_QUBITS))

    sampler = StatevectorSampler(seed=1234)
    result = sampler.run([circuit], shots=SHOTS).result()
    counts = extract_counts(result, REGISTER_NAME)
    total_shots = int(sum(counts.values()))
    expected_bitstring = "10000011"
    expected_counts = {expected_bitstring: SHOTS}
    expected_expectations = np.array(
        [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        dtype=np.float64,
    )
    actual_expectations = counts_to_z_expectations(counts, N_QUBITS)
    differences = np.abs(actual_expectations - expected_expectations)

    checks = {
        "named_register_present": bool(hasattr(result[0].data, REGISTER_NAME)),
        "counts_match_deterministic_circuit": counts == expected_counts,
        "shot_total_matches_request": total_shots == SHOTS,
        "output_shape_is_eight": actual_expectations.shape == (N_QUBITS,),
        "output_is_finite": bool(np.isfinite(actual_expectations).all()),
        "expectations_match": bool(np.max(differences) == 0.0),
    }

    malformed_results = []
    for name, malformed_result, expected_error in (
        (
            "missing_register",
            result,
            f"Sampler result does not contain a measurable register named 'missing'.",
        ),
        (
            "malformed_result",
            object(),
            "Sampler result does not contain a first publication.",
        ),
    ):
        try:
            extract_counts(malformed_result, "missing" if name == "missing_register" else REGISTER_NAME)
        except ValueError as error:
            malformed_results.append(
                {
                    "name": name,
                    "expected_error": expected_error,
                    "actual_error": str(error),
                    "passed": str(error) == expected_error,
                }
            )
        else:
            malformed_results.append(
                {
                    "name": name,
                    "expected_error": expected_error,
                    "actual_error": None,
                    "passed": False,
                }
            )

    all_passed = all(checks.values()) and all(
        case["passed"] for case in malformed_results
    )
    report = {
        "qiskit_version": qiskit.__version__,
        "sampler": "qiskit.primitives.StatevectorSampler",
        "n_qubits": N_QUBITS,
        "classical_register": REGISTER_NAME,
        "shots_requested": SHOTS,
        "counts": counts,
        "expected_counts": expected_counts,
        "shots_observed": total_shots,
        "expected_expectations": expected_expectations.tolist(),
        "actual_expectations": actual_expectations.tolist(),
        "absolute_differences": differences.tolist(),
        "bit_order": "logical qubit 0 is the rightmost bit",
        "checks": checks,
        "malformed_result_checks": malformed_results,
        "all_passed": all_passed,
    }
    with RESULT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Sampler result saved to {RESULT_PATH}")
    print(f"Counts: {counts}")
    print(f"Decoded expectations: {actual_expectations.tolist()}")
    print(f"Validation: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise RuntimeError("Sampler result extraction validation failed.")


if __name__ == "__main__":
    main()
