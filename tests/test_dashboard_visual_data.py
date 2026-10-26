import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import qcg_nn_app


class DashboardVisualDataTests(unittest.TestCase):
    def test_worker_launch_explicitly_forwards_ibm_environment(self):
        with patch.dict(
            qcg_nn_app.os.environ,
            {
                "IBM_QUANTUM_API_KEY": "redacted-test-token",
                "IBM_QUANTUM_INSTANCE": "crn:test-instance",
            },
            clear=False,
        ), patch.object(
            qcg_nn_app, "_worker_python", return_value=Path(sys.executable)
        ), patch.object(
            qcg_nn_app.subprocess, "run",
            return_value=type("Completed", (), {
                "returncode": 0,
                "stdout": '{"ok": true, "result": {"status": "submitted"}}',
            })) as run:
            qcg_nn_app._call_hardware_worker({"operation": "status"})
            environment = run.call_args.kwargs["env"]
            self.assertEqual(environment["IBM_QUANTUM_API_KEY"], "redacted-test-token")
            self.assertEqual(environment["IBM_QUANTUM_INSTANCE"], "crn:test-instance")

    def test_hardware_status_reports_redacted_flask_and_worker_presence(self):
        with patch.dict(
            qcg_nn_app.os.environ,
            {
                "IBM_QUANTUM_API_KEY": "redacted-test-token",
                "IBM_QUANTUM_INSTANCE": "crn:test-instance",
            },
            clear=False,
        ), patch.object(
            qcg_nn_app,
            "_call_hardware_worker",
            return_value=(
                {
                    "environment": {
                        "IBM_QUANTUM_API_KEY": True,
                        "IBM_QUANTUM_INSTANCE": True,
                    },
                },
                {"python": "qpu_execution_venv\\Scripts\\python.exe", "return_code": 0},
            ),
        ) as worker_call:
            payload = qcg_nn_app.app.test_client().get(
                "/api/hardware/status"
            ).get_json()
        self.assertTrue(payload["configured"])
        worker_call.assert_called_once_with(
            {"operation": "environment"},
            diagnostics=True,
        )
        self.assertEqual(
            payload["environment"],
            {
                "flask": {
                    "IBM_QUANTUM_API_KEY": True,
                    "IBM_QUANTUM_INSTANCE": True,
                },
                "worker": {
                    "IBM_QUANTUM_API_KEY": True,
                    "IBM_QUANTUM_INSTANCE": True,
                },
            },
        )

    def test_hardware_status_reports_missing_variables(self):
        with patch.dict(
            qcg_nn_app.os.environ,
            {
                "IBM_QUANTUM_API_KEY": "",
                "IBM_QUANTUM_INSTANCE": "",
            },
            clear=False,
        ), patch.object(
            qcg_nn_app,
            "_call_hardware_worker",
            return_value=(
                {
                    "environment": {
                        "IBM_QUANTUM_API_KEY": False,
                        "IBM_QUANTUM_INSTANCE": False,
                    }
                },
                {"python": "qpu_execution_venv\\Scripts\\python.exe", "return_code": 0},
            ),
        ):
            payload = qcg_nn_app.app.test_client().get(
                "/api/hardware/status"
            ).get_json()
        self.assertFalse(payload["configured"])
        self.assertEqual(
            payload["environment"]["flask"],
            {
                "IBM_QUANTUM_API_KEY": False,
                "IBM_QUANTUM_INSTANCE": False,
            },
        )

    def test_hardware_status_rejects_malformed_or_duplicate_instance_crn(self):
        for instance in ("not-a-crn", "crn:first crn:second"):
            with self.subTest(instance=instance), patch.dict(
                qcg_nn_app.os.environ,
                {
                    "IBM_QUANTUM_API_KEY": "redacted-test-token",
                    "IBM_QUANTUM_INSTANCE": instance,
                },
                clear=False,
            ), patch.object(
                qcg_nn_app,
                "_call_hardware_worker",
                return_value=(
                    {
                        "environment": {
                            "IBM_QUANTUM_API_KEY": True,
                            "IBM_QUANTUM_INSTANCE": False,
                        }
                    },
                    {"python": "qpu_execution_venv\\Scripts\\python.exe", "return_code": 0},
                ),
            ):
                payload = qcg_nn_app.app.test_client().get(
                    "/api/hardware/status"
                ).get_json()
            self.assertFalse(payload["configured"])
            self.assertTrue(payload["environment"]["flask"]["IBM_QUANTUM_API_KEY"])
            self.assertFalse(payload["environment"]["flask"]["IBM_QUANTUM_INSTANCE"])

    def test_prediction_contains_structure_attribution_and_completed_state(self):
        result = qcg_nn_app._run_simulator("CCO")
        self.assertEqual(result["execution"]["state"], "Completed")
        self.assertEqual(result["molecule"]["canonical_smiles"], "CCO")
        self.assertGreaterEqual(len(result["structure"]["nodes"]), 3)
        self.assertEqual(len(result["structure"]["bonds"]), 2)
        self.assertTrue(result["causal_analysis"]["kept_nodes"])
        self.assertEqual(
            len(result["causal_analysis"]["attribution_scores"]),
            result["molecule"]["model_atom_count"],
        )
        self.assertTrue(all(
            "x" in node and "y" in node and "label" in node
            for node in result["structure"]["nodes"]
        ))

    def test_predict_route_returns_completed_visual_payload(self):
        response = qcg_nn_app.app.test_client().post(
            "/api/predict", json={"smiles": "C=O"}
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["execution"]["state"], "Completed")
        self.assertIn("causal_bonds", payload["structure"])


if __name__ == "__main__":
    unittest.main()
