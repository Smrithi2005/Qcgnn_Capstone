import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from layer4_quantum import ibm_qpu_worker as worker


class FakeBitArray:
    def get_counts(self):
        return {"00000000": 4, "11111111": 4}


class FakeResult:
    class Entry:
        pass

    Entry.data = type("Data", (), {"meas": FakeBitArray()})()

    def __getitem__(self, index):
        return self.Entry()


class FakeJob:
    def __init__(self, status="DONE"):
        self._status = status

    def status(self):
        return self._status

    def done(self):
        return self._status in {"DONE", "CANCELLED"}

    def result(self):
        return FakeResult()


class FakeService:
    def __init__(self, status="DONE"):
        self._status = status

    def job(self, job_id):
        return FakeJob(self._status)


class WorkerTests(unittest.TestCase):
    def test_environment_probe_reports_presence_without_runtime(self):
        with patch.dict(
            os.environ,
            {"IBM_QUANTUM_API_KEY": "token", "IBM_QUANTUM_INSTANCE": "crn:instance"},
            clear=False,
        ), patch.object(worker, "_runtime_service", side_effect=AssertionError):
            result = worker.handle({"operation": "environment"})
        self.assertEqual(
            result["environment"],
            {"IBM_QUANTUM_API_KEY": True, "IBM_QUANTUM_INSTANCE": True},
        )

    def test_environment_probe_rejects_malformed_or_duplicate_instance_crn(self):
        for instance in ("not-a-crn", "crn:first crn:second"):
            with self.subTest(instance=instance), patch.dict(
                os.environ,
                {"IBM_QUANTUM_API_KEY": "token", "IBM_QUANTUM_INSTANCE": instance},
                clear=False,
            ):
                result = worker.handle({"operation": "environment"})
            self.assertEqual(
                result["environment"],
                {"IBM_QUANTUM_API_KEY": True, "IBM_QUANTUM_INSTANCE": False},
            )

    def test_worker_environment_diagnostic_does_not_expose_credentials(self):
        with patch.dict(
            os.environ,
            {"IBM_QUANTUM_API_KEY": "", "IBM_QUANTUM_INSTANCE": "crn:instance"},
            clear=False,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"IBM_QUANTUM_API_KEY=missing, IBM_QUANTUM_INSTANCE=present",
            ) as context:
                worker._runtime_service()
        self.assertNotIn("crn", str(context.exception))

    def test_submit_requires_exact_confirmation_and_valid_shapes(self):
        with self.assertRaises(PermissionError):
            worker.validate_request({"operation": "submit", "confirmation": "NO"})
        request = worker.validate_request({
            "operation": "submit",
            "confirmation": "SUBMIT",
            "backend": "ibm_fez",
            "shots": 128,
            "angles": [0.0] * 8,
            "weights": [[0.0] * 8, [0.0] * 8],
        })
        self.assertEqual(request["angles"].shape, (8,))
        self.assertEqual(request["weights"].shape, (2, 8))

    def test_status_requires_confirmation_and_job_id(self):
        with self.assertRaises(PermissionError):
            worker.validate_request({"operation": "status", "job_id": "job"})
        with self.assertRaises(ValueError):
            worker.validate_request({
                "operation": "status",
                "confirmation": "SUBMIT",
                "job_id": "",
            })

    def test_job_store_contains_only_non_secret_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.json"
            worker._save_jobs({"job": {"job_id": "job", "backend": "ibm_fez"}}, path)
            self.assertEqual(worker._load_jobs(path)["job"]["job_id"], "job")
            self.assertNotIn("token", json.loads(path.read_text(encoding="utf-8")))

    def test_completed_result_decodes_measurements_and_predictions(self):
        identity = list(range(worker.N_QUBITS))
        record = {
            "logical_to_physical": identity,
            "physical_to_classical": {str(index): index for index in identity},
        }
        measurements, standardized, predictions = worker._decode_result(record, FakeJob())
        np.testing.assert_allclose(measurements, np.zeros(worker.N_QUBITS))
        self.assertEqual(standardized.shape, (3,))
        self.assertEqual(predictions.shape, (3,))

    def test_status_returns_completed_result_without_runtime_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.json"
            worker._save_jobs({
                "job": {
                    "job_id": "job",
                    "backend": "ibm_fez",
                    "shots": 8,
                    "logical_to_physical": list(range(worker.N_QUBITS)),
                    "physical_to_classical": {
                        str(index): index for index in range(worker.N_QUBITS)
                    },
                }
            }, path)
            with patch.object(worker, "_load_jobs", return_value={
                "job": {
                    "job_id": "job",
                    "backend": "ibm_fez",
                    "shots": 8,
                    "logical_to_physical": list(range(worker.N_QUBITS)),
                    "physical_to_classical": {
                        str(index): index for index in range(worker.N_QUBITS)
                    },
                }
            }), patch.object(
                worker, "_runtime_service", return_value=(FakeService(), None)
            ):
                result = worker._status({
                    "operation": "status",
                    "confirmation": "SUBMIT",
                    "job_id": "job",
                })
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["mode"], "REAL IBM QPU")
            self.assertEqual(result["target_order"], ["HOMO", "LUMO", "GAP"])

    def test_status_reports_pending_and_failed_jobs(self):
        record = {
            "job_id": "job",
            "backend": "ibm_fez",
            "shots": 8,
            "logical_to_physical": list(range(worker.N_QUBITS)),
            "physical_to_classical": {
                str(index): index for index in range(worker.N_QUBITS)
            },
        }
        with patch.object(worker, "_load_jobs", return_value={"job": record}), patch.object(
            worker, "_runtime_service", return_value=(FakeService("RUNNING"), None)
        ):
            self.assertEqual(worker._status({
                "operation": "status", "confirmation": "SUBMIT", "job_id": "job"
            })["status"], "pending")
        with patch.object(worker, "_load_jobs", return_value={"job": record}), patch.object(
            worker, "_runtime_service", return_value=(FakeService("CANCELLED"), None)
        ):
            result = worker._status({
                "operation": "status", "confirmation": "SUBMIT", "job_id": "job"
            })
            self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
