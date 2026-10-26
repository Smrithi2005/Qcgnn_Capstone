import argparse
import json
import sys
import time
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
except ImportError as error:
    raise RuntimeError(
        "qiskit-ibm-runtime is required; use qpu_execution_venv."
    ) from error

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from ibm_hardware_adapter import (  # noqa: E402
    N_QUBITS,
    REGISTER_NAME,
    load_normalization_stats,
    load_trained_inputs,
    prepare_hardware_circuit,
    predictions_from_measurements,
    remap_counts_to_logical,
    submit_prepared_circuit,
)
from measurement_utils import counts_to_z_expectations  # noqa: E402


ROOT = _HERE.parents[2]
RESULTS_DIR = ROOT / "results"
TARGET_ORDER = ["HOMO", "LUMO", "GAP"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preflight or explicitly submit one QCGNN circuit to IBM Quantum."
    )
    parser.add_argument("backend", help="Explicit IBM backend name, for example ibm_fez.")
    parser.add_argument(
        "--shots", type=int, default=128, help="Shots for a submitted job (default: 128)."
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Enable the separately guarded hardware submission path.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=10.0,
        help="Polling interval after submission (default: 10 seconds).",
    )
    return parser.parse_args()


def is_simulator(backend):
    name = str(getattr(backend, "name", "")).lower()
    if "simulator" in name or "sim" in name:
        return True
    return bool(getattr(backend, "simulator", False))


def require_operational_backend(backend):
    if is_simulator(backend):
        raise RuntimeError(f"Backend {backend.name} is a simulator; a real backend is required.")
    status = backend.status()
    if not status.operational:
        raise RuntimeError(f"Backend {backend.name} is not operational.")
    if int(getattr(backend, "num_qubits", 0)) < N_QUBITS:
        raise RuntimeError(
            f"Backend {backend.name} has {backend.num_qubits} qubits; "
            f"{N_QUBITS} are required."
        )
    return status


def print_preflight(backend, status, prepared):
    circuit = prepared["circuit"]
    print("IBM Quantum preflight passed.")
    print(f"Backend: {backend.name}")
    print(f"Operational: {status.operational}")
    print(f"Backend qubits: {backend.num_qubits}")
    print(f"Circuit qubits: {N_QUBITS}")
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Operation counts: {dict(circuit.count_ops())}")
    print(f"Logical-to-physical: {prepared['logical_to_physical']}")
    print(f"Physical-to-classical: {prepared['physical_to_classical']}")
    print(f"Measurement register: {REGISTER_NAME}")
    print("Submission: not requested")


def result_path(backend_name):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in backend_name
    )
    return RESULTS_DIR / f"quantum_ibm_first_run_{safe_name}_{stamp}.json"


def main():
    args = parse_args()
    if args.shots <= 0:
        raise ValueError("--shots must be positive.")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive.")

    token = getpass("IBM Quantum API key: ").strip()
    crn = input("IBM Quantum instance CRN: ").strip()
    if not token or not crn:
        raise ValueError("API key and instance CRN are required.")

    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=token,
        instance=crn,
    )
    backend = service.backend(args.backend)
    status = require_operational_backend(backend)

    angles, weights, readout = load_trained_inputs()
    mean, std = load_normalization_stats()
    prepared = prepare_hardware_circuit(angles, weights, backend=backend)
    print_preflight(backend, status, prepared)

    if not args.submit:
        return

    confirmation = input("Type SUBMIT to submit this hardware job: ").strip()
    sampler = SamplerV2(mode=backend)
    job = submit_prepared_circuit(prepared, sampler, args.shots, confirmation)
    print(f"IBM job ID: {job.job_id()}")

    while not job.done():
        print(f"Job status: {job.status()}")
        time.sleep(args.poll_seconds)
    print(f"Job status: {job.status()}")

    runtime_result = job.result()
    bit_array = getattr(runtime_result[0].data, REGISTER_NAME, None)
    if bit_array is None or not hasattr(bit_array, "get_counts"):
        raise RuntimeError(f"IBM result lacks the '{REGISTER_NAME}' register.")
    physical_counts = bit_array.get_counts()
    logical_counts = remap_counts_to_logical(
        physical_counts,
        prepared["logical_to_physical"],
        prepared["physical_to_classical"],
    )
    measurements = counts_to_z_expectations(logical_counts, N_QUBITS)
    standardized, predictions_ev = predictions_from_measurements(
        measurements, readout, mean, std
    )
    output_path = result_path(backend.name)
    report = {
        "backend": backend.name,
        "job_id": job.job_id(),
        "shots": args.shots,
        "logical_to_physical": prepared["logical_to_physical"],
        "physical_to_classical": prepared["physical_to_classical"],
        "physical_counts": physical_counts,
        "logical_counts": logical_counts,
        "quantum_measurements": measurements.tolist(),
        "target_order": TARGET_ORDER,
        "standardized_predictions": standardized.tolist(),
        "predictions_eV": predictions_ev.tolist(),
    }
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Hardware result saved to {output_path}")


if __name__ == "__main__":
    main()
