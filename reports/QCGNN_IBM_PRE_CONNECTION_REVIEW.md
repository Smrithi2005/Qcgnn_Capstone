# QCGNN IBM Pre-Connection Review

**Project root:** `C:\Users\Dell\Desktop\qcgnn_project`  
**Review scope:** Final local adapter audit before any IBM connection  
**IBM Runtime contacted:** No  
**Hardware job submitted:** No

## Executive conclusion

The QCGNN hardware adapter’s local preparation path is validated, and two narrow readiness gaps were fixed and tested:

1. Backend count strings can now be remapped from physical/classical positions into logical order `[Z_0, ..., Z_7]`.
2. The adapter can now apply the trained classical readout, reverse the established target normalization once, and convert Hartree to eV once.

The adapter is **not yet ready for an approved hardware connection in the current environments**. The required packages are split:

| Environment | Qiskit | IBM Runtime | PyTorch | Result |
|---|---:|---:|---:|---|
| `venv` | 2.6.1 | Not installed | Available | Can load checkpoint and run local validation, but cannot create IBM Runtime service |
| `ibm_test_venv` | 2.6.1 | 0.51.0 | Not installed | Can use IBM Runtime API, but cannot load the PyTorch checkpoint through the adapter |

No package or environment change was made. A hardware run must wait for an explicitly approved compatible environment strategy.

## Files audited

| File | Audit focus |
|---|---|
| `code/layer4_quantum/ibm_hardware_adapter.py` | Circuit preparation, layout inspection, count remapping, readout, conversion, submission guard |
| `code/layer4_quantum/ibm_hardware_adapter_test.py` | Local dry-run behavior and classical readout input |
| `code/layer4_quantum/measurement_utils.py` | Counts-to-logical-Z conversion |
| `code/layer4_quantum/transpiled_layout_test.py` | Local transpilation and layout evidence |
| `reports/QCGNN_IBM_ADAPTER_DRY_RUN_REPORT.md` | Prior adapter validation |
| `reports/QCGNN_TRANSPILED_LAYOUT_VALIDATION_REPORT.md` | Prior layout and measurement validation |
| `reports/QCGNN_SIMULATOR_BASELINE_REPORT.md` | Baseline predictions, target order, and conversion convention |
| `models/best_qcgnn_corrected.pt` | Trained weights and classical readout tensors |
| `results/quantum_simulator_baseline_corrected.json` | Established graph-derived angles and baseline |
| `code/layer4_quantum/ibm_qpu_connection_test.py` | Existing secure account/backend-listing connection flow |

## IBM Runtime API audit

The installed IBM Runtime package was inspected in `ibm_test_venv` without constructing a service or contacting IBM:

```text
qiskit-ibm-runtime: 0.51.0
Qiskit:             2.6.1
```

The installed signatures are:

```text
QiskitRuntimeService(*args, **kwargs)
SamplerV2(mode=None, options=None)
SamplerV2.run(pubs, *, shots=None, dry_run=False)
```

The constructor source confirms support for:

```python
QiskitRuntimeService(
    channel="ibm_cloud",
    token=token,
    instance=crn,
)
```

The adapter does not guess or invoke this API. It accepts an explicitly created sampler through `submit_prepared_circuit`, which is appropriate for keeping authentication outside the model adapter.

## Authentication and credential handling

The adapter:

- Does not import or request credentials.
- Does not contain an API key, CRN, token, or service URL.
- Does not print, serialize, or write credentials.
- Does not create a Runtime service.
- Does not query IBM backends.

The existing connection-only script uses:

```python
token = getpass("IBM Quantum API key: ").strip()
crn = input("IBM Quantum instance CRN: ").strip()
QiskitRuntimeService(channel="ibm_cloud", token=token, instance=crn)
```

Credential values must be entered only in the local terminal. They must not be pasted into chat, source files, JSON results, Markdown reports, or command-line arguments.

## Submission guard audit

The adapter exposes submission separately:

```python
submit_prepared_circuit(prepared, sampler, shots, confirmation)
```

The function requires all of the following:

1. Exact `confirmation == "SUBMIT"`.
2. Positive shot count.
3. An explicitly supplied sampler object.

The default dry-run test does not call this function. A local guard test supplied a fake sampler and confirmation `"NO"`; the sampler’s `run` method was not called and the adapter raised:

```text
Hardware submission requires confirmation='SUBMIT'.
```

There is no default hardware path and no implicit service construction.

## Circuit and transpilation audit

The adapter rebuilds the corrected circuit using:

1. Eight input `RY` gates from the established baseline angles.
2. Eight trainable `RY` gates from checkpoint weights layer `0`.
3. Seven nearest-neighbor CNOT gates `(0,1)` through `(6,7)`.
4. Eight trainable `RY` gates from checkpoint weights layer `1`.
5. Eight measurements into the named `meas` classical register.

When a backend is supplied, the adapter calls Qiskit `transpile(..., backend=backend, optimization_level=1, ...)`. When running locally, it requires an explicit coupling map. It then inspects:

```text
TranspileLayout.final_index_layout(filter_ancillas=True)
```

and every measurement instruction to build:

```text
logical qubit → physical qubit
physical qubit → classical bit
```

It does not assume that logical qubit `i` remains physical qubit `i`.

## Counts and logical ordering

The adapter now includes a layout-aware `remap_counts_to_logical` helper. For each physical count bitstring, it:

1. Looks up the physical qubit assigned to each logical qubit.
2. Looks up the classical bit receiving that physical measurement.
3. Reads the corresponding Qiskit bit from the right-hand side.
4. Rebuilds a logical-order bitstring.
5. Passes the remapped counts to the existing `counts_to_z_expectations` decoder.

The local non-identity synthetic mapping check produced:

```text
input physical count:  {"10000000": 2}
remapped logical count: {"00000001": 2}
```

This confirms that count remapping is not tied to the local identity layout.

## Classical readout and unit-conversion audit

The corrected checkpoint readout remains:

```text
Linear(8 → 16) → ReLU → Linear(16 → 3)
```

The adapter now provides `predictions_from_measurements`, which:

1. Accepts exactly eight finite logical quantum measurements.
2. Applies the checkpoint classical readout.
3. Produces standardized predictions in target order HOMO, LUMO, GAP.
4. Loads `data/qm9/layer2/norm_stats.pkl`.
5. Verifies normalization target order `["homo", "lumo", "gap"]`.
6. Applies:
   ```text
   raw Hartree = standardized × std + mean
   eV = raw Hartree × 27.211386245988
   ```
   exactly once.

The helper was tested against the established simulator baseline. Its outputs were:

| Target | Adapter eV | Baseline eV |
|---|---:|---:|
| HOMO | -6.6767518174 | -6.6767519622 |
| LUMO | -0.1039163041 | -0.1039163751 |
| GAP | 6.5828416279 | 6.5828417309 |

The small differences are caused by comparing the independently reconstructed local Qiskit statevector with the serialized PennyLane baseline. No second normalization or eV conversion was applied.

## Local validation performed

Existing adapter dry-run:

```cmd
venv\Scripts\python.exe code\layer4_quantum\ibm_hardware_adapter_test.py
```

Result:

```text
Logical-to-physical mapping: [0, 1, 2, 3, 4, 5, 6, 7]
Validation: PASS
```

Additional local checks after the narrow adapter fixes:

- `py_compile` passed for the adapter.
- `git diff --check` passed.
- Non-identity count remapping passed.
- Adapter readout plus normalization/eV conversion matched the simulator baseline within `1.5e-7 eV`.
- Submission guard rejected non-`SUBMIT` without calling sampler `run`.
- Existing dry-run JSON was not overwritten.

No IBM connection or hardware execution was attempted.

## Exact local authentication/connection steps

These steps are for the user to perform locally only after approving the environment decision. Do not paste the token or CRN into chat.

### Step 1: Use the environment with IBM Runtime

The connection-only script requires `ibm_test_venv`:

```cmd
cd /d C:\Users\Dell\Desktop\qcgnn_project
ibm_test_venv\Scripts\python.exe code\layer4_quantum\ibm_qpu_connection_test.py
```

It prompts locally for:

```text
IBM Quantum API key:
IBM Quantum instance CRN:
```

The API key is hidden by `getpass`. The CRN is entered with normal `input()`. The script only lists accessible backends and does not submit jobs.

### Step 2: Do not run the adapter yet

Do not run the hardware adapter with `ibm_test_venv` currently, because that environment lacks PyTorch and cannot load `best_qcgnn_corrected.pt`.

Do not run a hardware submission command from `venv`, because `qiskit-ibm-runtime` is not installed there.

### Step 3: Resolve the environment blocker

Before a real execution, choose and explicitly approve one of these strategies:

1. Create a separate compatible execution environment containing the adapter’s required PyTorch stack and IBM Runtime `0.51.0`; or
2. Split checkpoint/readout preparation from Runtime submission into separately validated processes.

No package installation or environment change was made during this review.

## Risks and unresolved issues

| Risk | Status |
|---|---|
| IBM backend layout differs from local identity layout | Adapter inspects layout; still unverified until backend is supplied |
| Runtime result structure differs from local `BitArray` | IBM Runtime API is known, but no Runtime result was obtained |
| Runtime package unavailable with checkpoint environment | Current blocker |
| Hardware noise and readout error | Unverified |
| Finite-shot prediction variation | Unverified on hardware |
| Credential/account/CRN validity | Not tested in this review |
| Backend availability and operational state | Not queried |
| Frontend integration | Not implemented |
| Prediction scaling | Locally implemented and validated once |

## Readiness decision

**Adapter preparation:** Ready for review and local integration.  
**IBM connection:** Not yet ready in the current environment split.  
**Hardware submission:** Not authorized and not performed.

The next action requires explicit approval of a compatible environment strategy, backend, service instance, shot budget, and cost. Only after that should the user run the secure connection-only command, and later a separately guarded hardware submission path.

