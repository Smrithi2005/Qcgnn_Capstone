import json
from pathlib import Path

import numpy as np
import qiskit

from measurement_utils import counts_to_z_expectations


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "results" / "quantum_measurement_decoding_test.json"
N_QUBITS = 8
TOLERANCE = 1e-12


def as_list(values):
    return np.asarray(values, dtype=np.float64).tolist()


def main():
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite test result: {RESULT_PATH}")

    cases = [
        {
            "name": "all_zero",
            "counts": {"00000000": 128},
            "expected": [1.0] * N_QUBITS,
        },
        {
            "name": "all_one",
            "counts": {"11111111": 128},
            "expected": [-1.0] * N_QUBITS,
        },
        {
            "name": "balanced",
            "counts": {"00000000": 64, "11111111": 64},
            "expected": [0.0] * N_QUBITS,
        },
        {
            "name": "mixed_bitstrings",
            "counts": {
                "00000001": 4,
                "00000010": 2,
                "10000000": 1,
            },
            "expected": [
                -1.0 / 7.0,
                3.0 / 7.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                5.0 / 7.0,
            ],
        },
    ]

    results = []
    for case in cases:
        actual = counts_to_z_expectations(case["counts"], N_QUBITS)
        expected = np.asarray(case["expected"], dtype=np.float64)
        differences = np.abs(actual - expected)
        passed = bool(
            actual.shape == (N_QUBITS,)
            and np.isfinite(actual).all()
            and np.max(differences) <= TOLERANCE
        )
        results.append(
            {
                "name": case["name"],
                "counts": case["counts"],
                "expected": as_list(expected),
                "actual": as_list(actual),
                "absolute_differences": as_list(differences),
                "max_absolute_difference": float(np.max(differences)),
                "passed": passed,
            }
        )

    invalid_cases = [
        ("empty_counts", {}, "Measurement counts cannot be empty."),
        (
            "zero_total_shots",
            {"00000000": 0},
            "Total measurement shots must be greater than zero.",
        ),
    ]
    invalid_results = []
    for name, counts, expected_error in invalid_cases:
        try:
            counts_to_z_expectations(counts, N_QUBITS)
        except ValueError as error:
            invalid_results.append(
                {
                    "name": name,
                    "counts": counts,
                    "expected_error": expected_error,
                    "actual_error": str(error),
                    "passed": str(error) == expected_error,
                }
            )
        else:
            invalid_results.append(
                {
                    "name": name,
                    "counts": counts,
                    "expected_error": expected_error,
                    "actual_error": None,
                    "passed": False,
                }
            )

    all_passed = all(
        case["passed"] for case in results
    ) and all(case["passed"] for case in invalid_results)
    report = {
        "qiskit_version": qiskit.__version__,
        "n_qubits": N_QUBITS,
        "bit_order": "logical qubit 0 is the rightmost bit",
        "tolerance": TOLERANCE,
        "valid_cases": results,
        "invalid_cases": invalid_results,
        "all_passed": all_passed,
    }
    with RESULT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Measurement decoding result saved to {RESULT_PATH}")
    print(f"Validation: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise RuntimeError("Measurement decoding validation failed.")


if __name__ == "__main__":
    main()
