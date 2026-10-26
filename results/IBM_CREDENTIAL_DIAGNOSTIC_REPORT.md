# IBM QPU Credential Diagnostic Report

Date: 2026-10-25

## Root cause

The process actually serving port 5000 was PID 6512:

```text
C:\Users\Dell\AppData\Local\Programs\Python\Python310\python.exe qcg_nn_app.py
```

It was not the configured project interpreter:

```text
venv\Scripts\python.exe
```

That process returned the old response shape:

```json
{"configured":false,"mode":"REAL IBM QPU"}
```

The current source returns Flask and worker presence diagnostics, so the old response proves that an older/stale Flask process was serving port 5000. The repeated worker credential message was therefore not evidence that the current worker-launch code had received empty variables.

## Current source flow

- `qcg_nn_app.py::_hardware_configuration()` reads:
  - `IBM_QUANTUM_API_KEY`
  - `IBM_QUANTUM_INSTANCE`
- The helper validates that the instance is a single `crn:` value without whitespace, commas, or a duplicated `crn:` prefix.
- `_worker_environment()` starts with `os.environ.copy()` and explicitly assigns both validated IBM variables.
- `_worker_python()` uses `QCGNN_QPU_PYTHON`, or defaults to:
  - `qpu_execution_venv\Scripts\python.exe`
- `/api/hardware/status` launches the worker only with the `environment` operation.
- The worker environment operation reports booleans only. It does not import IBM Runtime, contact IBM, inspect a backend, or submit a job.
- Submission and job polling still require exact `confirmation="SUBMIT"` where applicable.

## Safe diagnostics

The current status response includes only:

- Flask variable presence/validation booleans.
- Worker variable presence/validation booleans.
- Worker Python path.
- Worker subprocess return code.
- A non-secret worker error category when the probe cannot run.

No API key or CRN value is returned or logged.

## Tests

Command:

```powershell
venv\Scripts\python.exe -m unittest tests\test_molecule_pipeline.py tests\test_ibm_qpu_worker.py tests\test_dashboard_visual_data.py -v
```

Result:

```text
Ran 19 tests
OK
```

Also passed:

```powershell
venv\Scripts\python.exe -m py_compile qcg_nn_app.py code\layer4_quantum\ibm_qpu_worker.py tests\test_dashboard_visual_data.py tests\test_ibm_qpu_worker.py
```

Coverage includes:

- Missing API key or instance.
- Valid configuration.
- Malformed and duplicated CRNs.
- Explicit subprocess environment propagation.
- Worker executable selection.
- Status probe without IBM Runtime access or submission.
- Exact `SUBMIT` confirmation safeguards.

## Restart sequence

Run this exact sequence from the project root in one PowerShell session:

```powershell
$listener = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
if ($listener) { Stop-Process -Id $listener.OwningProcess -Force }

$env:QCGNN_QPU_PYTHON = "C:\Users\Dell\Desktop\qcgnn_project\qpu_execution_venv\Scripts\python.exe"
venv\Scripts\python.exe qcg_nn_app.py
```

Then request:

```text
http://127.0.0.1:5000/api/hardware/status
```

## Live verification limit

The fix was verified with mocked subprocesses and local tests. No real IBM QPU job was submitted. A previous read-only IBM authentication test was reported as successful, but it was not rerun during this change.
