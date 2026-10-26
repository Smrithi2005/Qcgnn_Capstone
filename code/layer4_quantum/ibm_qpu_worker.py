"""One-request IBM Quantum worker for the QCGNN dashboard."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from layer4_quantum.ibm_hardware_adapter import (  # noqa: E402
    N_QUBITS,
    REGISTER_NAME,
    load_normalization_stats,
    predictions_from_measurements,
    prepare_hardware_circuit,
    remap_counts_to_logical,
    submit_prepared_circuit,
)
from layer4_quantum.measurement_utils import counts_to_z_expectations  # noqa: E402

TARGET_ORDER = ["HOMO", "LUMO", "GAP"]
JOB_STORE = ROOT / "results" / "quantum_ibm_worker_jobs.json"


def _valid_instance(instance):
    return bool(
        re.fullmatch(r"crn:[^\s,]+", instance)
        and instance.count("crn:") == 1
    )


def _finite_array(values, shape, label):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite and have shape {shape}.")
    return array


def validate_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("Worker request must be a JSON object.")
    operation = payload.get("operation")
    if operation not in {"environment", "submit", "status"}:
        raise ValueError("Worker operation must be 'environment', 'submit', or 'status'.")
    if operation == "environment":
        return {"operation": operation}
    if payload.get("confirmation") != "SUBMIT":
        raise PermissionError("Hardware submission requires confirmation='SUBMIT'.")
    if operation == "status":
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("A job_id is required.")
        return {"operation": operation, "job_id": job_id.strip()}

    backend = payload.get("backend")
    shots = payload.get("shots")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("A backend name is required.")
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer.")
    return {
        "operation": operation,
        "confirmation": "SUBMIT",
        "backend": backend.strip(),
        "shots": shots,
        "angles": _finite_array(payload.get("angles"), (N_QUBITS,), "angles"),
        "weights": _finite_array(payload.get("weights"), (2, N_QUBITS), "weights"),
    }


def _load_jobs(path=JOB_STORE):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        jobs = json.load(handle)
    if not isinstance(jobs, dict):
        raise ValueError("IBM worker job store is invalid.")
    return jobs


def _save_jobs(jobs, path=JOB_STORE):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(jobs, handle, indent=2)
    temporary.replace(path)


def _runtime_service():
    token = os.getenv("IBM_QUANTUM_API_KEY", "").strip()
    instance = os.getenv("IBM_QUANTUM_INSTANCE", "").strip()
    if not token or not _valid_instance(instance):
        token_state = "present" if token else "missing"
        instance_state = "present" if _valid_instance(instance) else "missing or invalid"
        raise RuntimeError(
            "IBM credentials are incomplete in the worker environment "
            f"(IBM_QUANTUM_API_KEY={token_state}, "
            f"IBM_QUANTUM_INSTANCE={instance_state})."
        )
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as error:
        raise RuntimeError("qiskit-ibm-runtime is required in qpu_execution_venv.") from error
    return (
        QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance),
        SamplerV2,
    )


def _environment_status():
    instance = os.getenv("IBM_QUANTUM_INSTANCE", "").strip()
    return {
        "IBM_QUANTUM_API_KEY": bool(os.getenv("IBM_QUANTUM_API_KEY", "").strip()),
        "IBM_QUANTUM_INSTANCE": _valid_instance(instance),
    }


def _require_real_backend(service, backend_name):
    backend = service.backend(backend_name)
    if bool(getattr(backend, "simulator", False)) or "sim" in str(backend.name).lower():
        raise RuntimeError(f"{backend.name} is a simulator; a real QPU is required.")
    if not backend.status().operational:
        raise RuntimeError(f"Backend {backend.name} is not operational.")
    if int(getattr(backend, "num_qubits", 0)) < N_QUBITS:
        raise RuntimeError(f"Backend {backend.name} has fewer than {N_QUBITS} qubits.")
    return backend


def _submit(payload):
    service, sampler_type = _runtime_service()
    backend = _require_real_backend(service, payload["backend"])
    prepared = prepare_hardware_circuit(
        payload["angles"], payload["weights"], backend=backend
    )
    job = submit_prepared_circuit(
        prepared, sampler_type(mode=backend), payload["shots"], payload["confirmation"]
    )
    job_id = str(job.job_id())
    jobs = _load_jobs()
    jobs[job_id] = {
        "backend": str(backend.name),
        "job_id": job_id,
        "shots": payload["shots"],
        "logical_to_physical": prepared["logical_to_physical"],
        "physical_to_classical": {
            str(key): value for key, value in prepared["physical_to_classical"].items()
        },
    }
    _save_jobs(jobs)
    return {
        "status": "submitted",
        "execution": {"state": "Queued"},
        "mode": "REAL IBM QPU",
        "backend": str(backend.name),
        "job_id": job_id,
        "job_status": str(job.status()),
        "shots": payload["shots"],
        "message": "Submitted to IBM Quantum. No simulator fallback was used.",
    }


def _decode_result(record, job):
    runtime_result = job.result()
    bit_array = getattr(runtime_result[0].data, REGISTER_NAME, None)
    if bit_array is None or not hasattr(bit_array, "get_counts"):
        raise RuntimeError(f"IBM result lacks the '{REGISTER_NAME}' register.")
    logical_counts = remap_counts_to_logical(
        bit_array.get_counts(),
        record["logical_to_physical"],
        {int(key): value for key, value in record["physical_to_classical"].items()},
    )
    measurements = counts_to_z_expectations(logical_counts, N_QUBITS)
    checkpoint = ROOT / "models" / "best_qcgnn_corrected.pt"
    import torch

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    readout = {
        key: state_dict[key].detach().cpu().numpy()
        for key in (
            "classical_readout.0.weight",
            "classical_readout.0.bias",
            "classical_readout.2.weight",
            "classical_readout.2.bias",
        )
    }
    mean, std = load_normalization_stats()
    standardized, predictions = predictions_from_measurements(
        measurements, readout, mean, std
    )
    return measurements, standardized, predictions


def _status(payload):
    jobs = _load_jobs()
    record = jobs.get(payload["job_id"])
    if record is None:
        raise KeyError("Unknown or expired hardware job.")
    service, _ = _runtime_service()
    job = service.job(record["job_id"])
    job_status = str(job.status())
    response = {
        "mode": "REAL IBM QPU",
        "backend": record["backend"],
        "job_id": record["job_id"],
        "job_status": job_status,
        "shots": record["shots"],
        "status": "pending" if not job.done() else "completed",
    }
    if not job.done():
        response["execution"] = {
            "state": "Running" if "RUN" in job_status.upper() else "Queued"
        }
        return response
    if not any(value in job_status.lower() for value in ("done", "completed", "success")):
        response.update(
            status="failed",
            execution={"state": "Failed"},
            error=f"IBM job ended with status {job_status}.",
        )
        return response
    measurements, standardized, predictions = _decode_result(record, job)
    response.update(
        execution={"state": "Completed"},
        measurements=measurements.tolist(),
        standardized_predictions=standardized.tolist(),
        prediction=dict(zip(("homo", "lumo", "gap"), predictions.tolist())),
        target_order=TARGET_ORDER,
    )
    result_path = (
        ROOT
        / "results"
        / f"quantum_ibm_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(response, handle, indent=2)
    response["result_file"] = str(result_path.relative_to(ROOT))
    return response


def handle(payload):
    request_data = validate_request(payload)
    if request_data["operation"] == "environment":
        return {"environment": _environment_status()}
    if request_data["operation"] == "submit":
        return _submit(request_data)
    return _status(request_data)


def main():
    try:
        payload = json.load(sys.stdin)
        response = {"ok": True, "result": handle(payload)}
    except Exception as error:
        response = {"ok": False, "error": str(error)}
    print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
