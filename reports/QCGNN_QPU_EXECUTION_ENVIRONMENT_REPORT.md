# QCGNN QPU Execution Environment Report

**Project root:** `C:\Users\Dell\Desktop\qcgnn_project`  
**Environment created:** `qpu_execution_venv`  
**Python:** `3.10.10`  
**IBM Runtime contacted:** No  
**Hardware job submitted:** No

## Executive summary

A third isolated environment, `qpu_execution_venv`, was created without modifying `venv` or `ibm_test_venv`.

The approved minimal adapter stack was installed and verified:

| Package | Installed version |
|---|---:|
| PyTorch | `2.7.1+cu118` |
| Qiskit | `2.6.1` |
| qiskit-ibm-runtime | `0.51.0` |
| NumPy | Installed transitively |

`pip check` passed. The corrected checkpoint loaded successfully, the adapter imported and loaded its baseline/checkpoint/normalization inputs, and the complete local adapter execution path passed using a local Qiskit `StatevectorSampler`.

No IBM credentials were accessed, no IBM backend was queried, no Runtime service was created, and no hardware job was submitted.

## Pre-creation inspection

Available Python interpreters before creation:

```text
Python 3.11.9  C:\Users\Dell\AppData\Local\Programs\Python\Python311\python.exe
Python 3.10.10 C:\Users\Dell\AppData\Local\Programs\Python\Python310\python.exe
CPython 3.14.3 managed by uv
```

The existing environments both use Python `3.10.10`.

Available disk space before creation:

```text
C: free space: 41.86 GB
```

The environment was created only after the dependency scope and compatibility plan were approved.

## Existing environment comparison

| Environment | Python | Qiskit | IBM Runtime | PyTorch | PyTorch Geometric |
|---|---:|---:|---:|---:|---:|
| `venv` | 3.10.10 | 2.6.1 | Not installed | 2.7.1+cu118 | 2.7.0 |
| `ibm_test_venv` | 3.10.10 | 2.6.1 | 0.51.0 | Not installed | Not installed |
| `qpu_execution_venv` | 3.10.10 | 2.6.1 | 0.51.0 | 2.7.1+cu118 | Not installed |

The project’s existing `requirements_before_ibm.txt` was inspected. It describes the broad simulator/QCGNN environment but is not used as a blind installation source for the new environment.

## Dependency decision

The adapter imports:

```text
torch
numpy
qiskit
```

The adapter uses IBM Runtime only through an explicitly injected `SamplerV2` object; the local project environment did not contain IBM Runtime, so the new environment includes it.

PennyLane and PyTorch Geometric were intentionally not installed in the minimal environment because:

- `ibm_hardware_adapter.py` does not import PennyLane.
- The adapter loads the already-produced graph-derived angles from the baseline JSON rather than recomputing graph features.
- The adapter does not import `HybridQuantumGNN` or use PyTorch Geometric data objects.
- The corrected checkpoint’s tensors can be loaded directly with PyTorch.

The existing `venv` remains the environment for the original PennyLane/QCGNN simulator workflow.

## Installation commands used

The following commands were run from the project root:

```cmd
cd /d C:\Users\Dell\Desktop\qcgnn_project
py -3.10 -m venv qpu_execution_venv
qpu_execution_venv\Scripts\python.exe -m pip install --upgrade pip
qpu_execution_venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.7.1+cu118
qpu_execution_venv\Scripts\python.exe -m pip install qiskit==2.6.1 qiskit-ibm-runtime==0.51.0
```

The pinned PyTorch CUDA wheel matches the existing project environment. Qiskit and IBM Runtime versions match the separately validated IBM environment.

## Import and API verification

The following imports succeeded in `qpu_execution_venv`:

```python
import torch
import qiskit
import qiskit_ibm_runtime
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
```

Verified versions:

```text
torch:              2.7.1+cu118
qiskit:             2.6.1
qiskit-ibm-runtime: 0.51.0
```

The installed Runtime signatures were verified as:

```text
SamplerV2(mode=None, options=None)
SamplerV2.run(pubs, *, shots=None, dry_run=False)
```

`QiskitRuntimeService` was imported for API verification only. It was not instantiated.

## Dependency validation

Command:

```cmd
qpu_execution_venv\Scripts\python.exe -m pip check
```

Result:

```text
No broken requirements found.
```

The original `venv` and `ibm_test_venv` were not changed. Their package states were inspected only.

## Checkpoint validation

Command:

```cmd
qpu_execution_venv\Scripts\python.exe -c "import torch; s=torch.load(r'models\best_qcgnn_corrected.pt',map_location='cpu',weights_only=True); print(sorted(s)); print(tuple(s['qlayer.weights'].shape))"
```

Actual result:

```text
checkpoint keys:
[
  'classical_projection.bias',
  'classical_projection.weight',
  'classical_readout.0.bias',
  'classical_readout.0.weight',
  'classical_readout.2.bias',
  'classical_readout.2.weight',
  'qlayer.weights'
]
quantum weight shape: (2, 8)
```

The checkpoint was read only. It was not modified or rewritten.

## Adapter validation

Adapter import and input-loading command:

```cmd
qpu_execution_venv\Scripts\python.exe -c "import sys; from pathlib import Path; sys.path.insert(0,str(Path('code\layer4_quantum').resolve())); import ibm_hardware_adapter; print('adapter import PASS'); a,w,r=ibm_hardware_adapter.load_trained_inputs(); m,sd=ibm_hardware_adapter.load_normalization_stats(); print(a.shape,w.shape,m.shape,sd.shape)"
```

Actual result:

```text
adapter import PASS
(8,) (2, 8) (3,) (3,)
```

The local adapter execution path was then exercised directly without writing a replacement result JSON:

```cmd
qpu_execution_venv\Scripts\python.exe -
```

It performed:

1. Corrected baseline and checkpoint loading.
2. Local eight-qubit transpilation.
3. Local `StatevectorSampler` execution with 128 shots.
4. Layout-aware count remapping.
5. Existing counts-to-Pauli-Z decoding.
6. Classical readout.
7. Normalization reversal and Hartree-to-eV conversion once.
8. Submission-guard rejection using confirmation `"NO"`.

Actual output:

```text
mapping [0, 1, 2, 3, 4, 5, 6, 7]
shots 128
measurements [-0.21875, -0.046875, -0.109375, -0.5, 0.65625, -0.390625, 0.171875, -0.359375]
standardized [-0.20571853052111433, -0.38853198751987766, -0.27689991158658006]
predictions_ev [-6.651954164580283, -0.1997355778539184, 6.466919684620895]
submission_guard PASS
local adapter execution PASS
```

The finite-shot predictions differ from the simulator baseline because this test intentionally uses only 128 local samples. This is expected sampling variation and is not a hardware result.

## Existing adapter test result-file collision

The existing adapter test was invoked with:

```cmd
qpu_execution_venv\Scripts\python.exe code\layer4_quantum\ibm_hardware_adapter_test.py
```

It stopped before execution because its prior result already exists:

```text
FileExistsError: Refusing to overwrite dry-run result:
C:\Users\Dell\Desktop\qcgnn_project\results\quantum_ibm_adapter_dry_run.json
```

This confirms collision protection. The adapter path itself was validated directly without overwriting that established result.

## Credential and hardware safety

During this setup and validation:

- No API key was requested.
- No CRN was requested.
- No credential was written to disk.
- No credential was placed in a command line.
- No IBM service was instantiated.
- No backend was queried.
- No Runtime session was opened.
- No hardware job was submitted.

The secure connection-only command remains separate:

```cmd
ibm_test_venv\Scripts\python.exe code\layer4_quantum\ibm_qpu_connection_test.py
```

It should only be run locally when authentication testing is approved. Credentials must be entered interactively and must not be pasted into chat.

## Remaining blockers and limitations

The environment is prepared for a later approved connection, but this report does not authorize or perform one.

Still unverified:

- IBM API-key authentication in `qpu_execution_venv`
- Account/service-instance access
- Backend availability and operational status
- Backend-specific transpiler layout
- IBM Runtime SamplerV2 result extraction
- Hardware measurement noise and readout error
- Hardware prediction stability
- Frontend backend selection

The next safe step is a connection-only authentication/backend-listing test using secure local prompts. A hardware submission still requires separate approval of backend, instance, shot count, and cost.

## Final status

| Item | Status |
|---|---|
| Third environment created | Passed |
| Existing environments preserved | Passed |
| Pinned core packages installed | Passed |
| `pip check` | Passed |
| Corrected checkpoint load | Passed |
| Local adapter preparation | Passed |
| Local sampler/decoder/readout test | Passed |
| IBM authentication attempted | No |
| IBM backend queried | No |
| Hardware job submitted | No |

