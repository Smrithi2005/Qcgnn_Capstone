from collections.abc import Mapping
from numbers import Real

import numpy as np


def counts_to_z_expectations(
    counts: Mapping[str, Real], n_qubits: int
) -> np.ndarray:
    """Convert standard Qiskit counts to [Z_0, ..., Z_(n_qubits-1)]."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if not counts:
        raise ValueError("Measurement counts cannot be empty.")

    expectations = np.zeros(n_qubits, dtype=np.float64)
    total_shots = 0.0
    for raw_bitstring, raw_count in counts.items():
        if not isinstance(raw_bitstring, str):
            raise TypeError("Measurement count keys must be bitstrings.")
        bitstring = raw_bitstring.replace(" ", "")
        if len(bitstring) != n_qubits or set(bitstring) - {"0", "1"}:
            raise ValueError(
                f"Each bitstring must contain exactly {n_qubits} binary bits."
            )
        if not isinstance(raw_count, Real) or not np.isfinite(raw_count):
            raise ValueError("Measurement counts must be finite numbers.")
        if raw_count < 0:
            raise ValueError("Measurement counts cannot be negative.")

        count = float(raw_count)
        total_shots += count
        for qubit in range(n_qubits):
            expectations[qubit] += count * (1.0 if bitstring[-1 - qubit] == "0" else -1.0)

    if total_shots <= 0:
        raise ValueError("Total measurement shots must be greater than zero.")
    expectations /= total_shots
    if not np.isfinite(expectations).all():
        raise ValueError("Decoded expectations contain non-finite values.")
    return expectations
