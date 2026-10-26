# QCGNN IBM Hardware Adapter Dry-Run Report

**Project root:** `C:\Users\Dell\Desktop\qcgnn_project`  
**Validation type:** Local adapter preparation and dry-run only  
**Qiskit:** `2.6.1`  
**IBM Runtime inspected:** `0.51.0` in `ibm_test_venv`  
**IBM Runtime contacted:** No  
**Hardware job submitted:** No

## Executive summary

A separate IBM hardware adapter was created and validated locally for the corrected QCGNN circuit. It:

- Loads the established graph-derived angles from the simulator baseline.
- Loads the trained quantum weights and classical readout parameters from the corrected checkpoint.
- Rebuilds the exact eight-qubit QCGNN circuit with a named `meas` register.
- Supports backend-specific transpilation when an explicitly supplied backend is provided.
- Supports local transpilation through an explicit coupling map for dry-run validation.
- Inspects the actual transpiler layout and measurement mapping.
- Preserves logical measurement order `[Z_0, ..., Z_7]`.
- Provides an explicit submission function requiring `confirmation="SUBMIT"`.

The local dry-run passed. No IBM credentials, service instance, backend, Runtime session, or hardware job was used.

## Created artifacts

| File | Purpose |
|---|---|
| `code/layer4_quantum/ibm_hardware_adapter.py` | Separate circuit preparation, layout inspection, and guarded submission adapter |
| `code/layer4_quantum/ibm_hardware_adapter_test.py` | Local dry-run validation |
| `results/quantum_ibm_adapter_dry_run.json` | Actual machine-readable dry-run results |
| `reports/QCGNN_IBM_ADAPTER_DRY_RUN_REPORT.md` | This report |

No existing model, circuit, training, evaluation, checkpoint, dataset, frontend, or environment file was modified.

## Files inspected

| File | Relevance |
|---|---|
| `code/layer4_quantum/quantum_circuit.py` | Corrected PennyLane gate order |
| `code/layer4_quantum/quantum_model.py` | Quantum parameter shape and classical readout |
| `code/layer4_quantum/local_qiskit_validation.py` | Existing local Qiskit circuit mapping |
| `code/layer4_quantum/measurement_utils.py` | Ordered counts-to-Z decoder |
| `code/layer4_quantum/transpiled_layout_test.py` | Existing local layout and measurement validation |
| `results/quantum_simulator_baseline_corrected.json` | Exact input angles and baseline context |
| `models/best_qcgnn_corrected.pt` | Trained quantum weights and readout parameters |
| `reports/QCGNN_QISKIT_CIRCUIT_EQUIVALENCE_REPORT.md` | Prior circuit equivalence evidence |
| `reports/QCGNN_MEASUREMENT_DECODING_REPORT.md` | Bitstring ordering evidence |
| `reports/QCGNN_SAMPLER_RESULT_VALIDATION_REPORT.md` | Local SamplerV2 extraction evidence |
| `reports/QCGNN_TRANSPILED_LAYOUT_VALIDATION_REPORT.md` | Local layout evidence |

## Installed API facts

The project environment used for the local test contains:

```text
Qiskit: 2.6.1
qiskit-ibm-runtime: not installed in venv
```

The separate `ibm_test_venv` was inspected without contacting IBM:

```text
Qiskit: 2.6.1
qiskit-ibm-runtime: 0.51.0
```

The installed Runtime API in `ibm_test_venv` exposes:

```text
QiskitRuntimeService(*args, **kwargs)
SamplerV2(mode=None, options=None)
SamplerV2.run(pubs, *, shots=None, dry_run=False)
```

The adapter therefore does not import Runtime at module load time and does not guess or invoke a service API during dry-run. A future hardware caller can inject an actual `SamplerV2` instance created with the installed Runtime API.

## Adapter behavior

### Input loading

The adapter reads:

```text
Baseline:   results/quantum_simulator_baseline_corrected.json
Checkpoint: models/best_qcgnn_corrected.pt
```

It verifies:

```text
Input angles:       shape (8,)
Quantum weights:    shape (2, 8)
```

It also loads the four classical readout tensors:

```text
classical_readout.0.weight
classical_readout.0.bias
classical_readout.2.weight
classical_readout.2.bias
```

No normalization or eV conversion is performed by this hardware-preparation stage.

### Circuit sequence

The adapter reproduces:

1. Eight input `RY` gates.
2. Eight trainable `RY` gates from weight layer `0`.
3. Seven nearest-neighbor CNOT gates `(0,1)` through `(6,7)`.
4. Eight trainable `RY` gates from weight layer `1`.
5. Eight measurements into the named `meas` classical register.

The logical measurement order remains `[Z_0, ..., Z_7]`.

### Backend-specific transpilation

The adapter supports:

```python
prepare_hardware_circuit(..., backend=backend)
```

When a backend is supplied, it calls Qiskit’s installed `transpile` API with that backend and optimization level `1`. It then reads `TranspileLayout.final_index_layout(filter_ancillas=True)` and inspects each measurement instruction to build:

```text
logical qubit -> physical qubit
physical qubit -> classical bit
```

It does not assume an identity mapping.

For local validation only, the test supplies a bidirectional eight-qubit line coupling map and local basis gates:

```text
rz, sx, x, cx
```

This avoids any backend query.

### Guarded submission

The adapter’s submission function is separate from preparation:

```python
submit_prepared_circuit(prepared, sampler, shots, confirmation)
```

It refuses to run unless:

```text
confirmation == "SUBMIT"
```

It also requires an explicitly supplied sampler and a positive shot count. The default dry-run test never calls this function. No credentials are read or stored by the adapter.

## Local dry-run command

The exact command was:

```cmd
venv\Scripts\python.exe code\layer4_quantum\ibm_hardware_adapter_test.py
```

Actual output:

```text
Adapter dry-run result saved to C:\Users\Dell\Desktop\qcgnn_project\results\quantum_ibm_adapter_dry_run.json
Logical-to-physical mapping: [0, 1, 2, 3, 4, 5, 6, 7]
Validation: PASS
```

## Actual local results

| Check | Actual result |
|---|---|
| Baseline loaded | Passed |
| Corrected checkpoint loaded | Passed |
| Exact eight angles reused | Passed |
| Exact trained weights reused | Passed |
| Named `meas` register present | Passed |
| Logical-to-physical layout complete | Passed |
| Local physical-to-classical map complete | Passed |
| Local SamplerV2 execution | Passed |
| Shot total equals `8192` | Passed |
| Decoder output shape is `(8,)` | Passed |
| Decoder output finite | Passed |
| Finite-shot sample within `0.08` tolerance | Passed |
| Classical readout applied to local reference measurements | Passed |
| Hardware submission requested | No |

The observed local mapping was:

```text
logical_to_physical: [0, 1, 2, 3, 4, 5, 6, 7]
physical_to_classical: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
```

This identity mapping is only a property of the selected local coupling map and transpiler run. It is not evidence about an IBM backend.

## Collision protection

The adapter test refuses to overwrite:

```text
results/quantum_ibm_adapter_dry_run.json
```

A second execution was attempted and stopped safely with:

```text
FileExistsError: Refusing to overwrite dry-run result:
C:\Users\Dell\Desktop\qcgnn_project\results\quantum_ibm_adapter_dry_run.json
```

## What was validated locally

This stage validates adapter preparation, not hardware connectivity:

1. Corrected baseline and checkpoint loading.
2. Exact trained circuit reconstruction.
3. Explicit named-register measurements.
4. Local backend-like transpilation through a supplied coupling map.
5. Layout and measurement-map inspection.
6. Local SamplerV2 execution and counts extraction.
7. Logical-order decoding through the existing measurement utility.
8. Classical readout input preparation.
9. Separate guarded submission API.
10. Safe result-file collision behavior.

## What remains unverified

The following require a later approved hardware stage:

- IBM credential authentication
- Service-instance selection
- Backend retrieval
- Backend-specific transpilation and layout
- IBM Runtime `SamplerV2` execution
- Runtime result extraction on IBM’s service
- Hardware noise and readout error
- Real hardware counts
- Frontend backend selection
- End-to-end hardware inference

The adapter must be run with the separate `ibm_test_venv` only after its secure authentication and backend-access prerequisites are approved. The current local test deliberately did not import Runtime in `venv`, read credentials, query IBM, or submit a job.

## Next steps

The preparation gate passed. Before any hardware execution:

1. Review the adapter and confirm the intended IBM backend and shot budget.
2. Use secure runtime authentication without hardcoding credentials.
3. Retrieve the approved backend only after explicit approval.
4. Transpile with that backend and record its actual layout.
5. Require a separate visible `SUBMIT` confirmation.
6. Decode the returned counts into logical `[Z_0, ..., Z_7]`.
7. Apply the existing classical readout and only then apply the established target-unit conversion.

No claim of IBM hardware execution or accuracy is made by this report.

