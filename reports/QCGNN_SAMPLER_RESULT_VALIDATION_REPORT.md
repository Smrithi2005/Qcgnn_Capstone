# QCGNN Local SamplerV2 Result Validation Report

**Project root:** `C:\Users\Dell\Desktop\qcgnn_project`  
**Validation type:** Local Qiskit SamplerV2 result extraction  
**Qiskit version:** `2.6.1`  
**IBM Runtime contacted:** No  
**Hardware job submitted:** No

## Executive summary

The installed Qiskit API supports the local `StatevectorSampler` implementation of SamplerV2. An eight-qubit circuit with an explicitly named classical register was executed locally. Counts were extracted from the actual result object and passed through the existing measurement decoder.

The validation passed:

```text
Counts:              {'10000011': 128}
Shots observed:      128
Decoded output:      [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
Output values:       8 finite values
Result:              PASS
```

The test used no credentials, no IBM Runtime service, no backend lookup, and no hardware submission.

## Files inspected

| File | Relevance |
|---|---|
| `code/layer4_quantum/measurement_utils.py` | Existing counts-to-`[Z_0, ..., Z_7]` decoder |
| `code/layer4_quantum/measurement_utils_test.py` | Existing synthetic decoder tests |
| `code/layer4_quantum/local_qiskit_validation.py` | Existing Qiskit circuit and logical-qubit convention |
| `reports/QCGNN_MEASUREMENT_DECODING_REPORT.md` | Prior decoder validation and stated SamplerV2 next step |
| `reports/QCGNN_QISKIT_CIRCUIT_EQUIVALENCE_REPORT.md` | Prior local Qiskit equivalence result |
| `reports/QCGNN_IBM_HARDWARE_INTEGRATION_REPORT.md` | Overall IBM integration roadmap |

No production model, circuit, checkpoint, training, evaluation, dataset, frontend, or environment file was changed.

## New artifacts

| File | Purpose |
|---|---|
| `code/layer4_quantum/sampler_result_test.py` | Executes and validates the local SamplerV2 result extraction path |
| `results/quantum_sampler_result_test.json` | Machine-readable actual result and assertions |
| `reports/QCGNN_SAMPLER_RESULT_VALIDATION_REPORT.md` | This report |

## Installed API verification

The installed Qiskit version was checked in both relevant environments:

```text
venv:          2.6.1
ibm_test_venv: 2.6.1
```

The local SamplerV2 implementation is available:

```python
from qiskit.primitives import StatevectorSampler
```

The inspected constructor and run API are:

```text
StatevectorSampler(*, default_shots=1024, seed=None)
StatevectorSampler.run(pubs, *, shots=None)
```

No package installation or upgrade was required.

## Circuit and register

The test constructs an eight-qubit circuit with a named eight-bit classical register:

```text
Classical register: meas
Qubits:             8
Requested shots:    128
```

The circuit applies `X` to qubits `0`, `1`, and `7`, leaves the other qubits in `|0>`, and measures every qubit into the corresponding `meas` register bit.

The expected standard Qiskit display bitstring is:

```text
10000011
```

This is an intentional ordering test:

- Rightmost bit `1` represents logical qubit `0`.
- Next bit `1` represents logical qubit `1`.
- Leftmost bit `1` represents logical qubit `7`.

## Actual SamplerV2 result structure

The test uses the actual local result returned by:

```python
result = StatevectorSampler(seed=1234).run(
    [circuit],
    shots=128,
).result()
```

The observed object path was:

```text
PrimitiveResult
  └── result[0]                  SamplerPubResult
      └── result[0].data         DataBin
          └── data.meas          BitArray
              └── get_counts()    {'10000011': 128}
```

The script extracts the named register with:

```python
bit_array = result[0].data.meas
counts = bit_array.get_counts()
```

It does not construct or imitate a Runtime result object.

## Actual counts and decoding

Actual extracted counts:

```json
{"10000011": 128}
```

Observed total:

```text
128 shots
```

Expected ordered Pauli-Z values:

```text
[-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
```

Actual decoder output:

```text
[-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
```

Per-qubit absolute differences:

```text
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

The output shape is exactly `(8,)`, and every value is finite.

## Assertions and results

| Assertion | Result |
|---|---|
| Qiskit version recorded | Passed (`2.6.1`) |
| Local `StatevectorSampler` available | Passed |
| Explicit `meas` register present | Passed |
| Actual `BitArray.get_counts()` used | Passed |
| Extracted counts match deterministic circuit | Passed |
| Observed shots equal requested `128` | Passed |
| Logical qubit 0 uses rightmost bit | Passed |
| Output shape is exactly eight | Passed |
| Output values are finite | Passed |
| Decoded expectations match analytically expected values | Passed |
| Missing-register error handling | Passed |
| Malformed-result error handling | Passed |
| IBM Runtime contacted | No |
| Hardware job submitted | No |

## Error handling tests

The test also exercises extraction failures supported by the actual result access pattern.

### Missing register

Attempting to extract a register named `missing` raises:

```text
Sampler result does not contain a measurable register named 'missing'.
```

The expected and actual errors matched.

### Malformed result

Attempting to extract from a non-result object raises:

```text
Sampler result does not contain a first publication.
```

The expected and actual errors matched.

## Validation command

Run the test from the project root with:

```cmd
venv\Scripts\python.exe code\layer4_quantum\sampler_result_test.py
```

The generated result is:

```text
results/quantum_sampler_result_test.json
```

The script refuses to overwrite that result. A second execution was intentionally attempted and stopped safely with:

```text
FileExistsError: Refusing to overwrite test result:
C:\Users\Dell\Desktop\qcgnn_project\results\quantum_sampler_result_test.json
```

The `ibm_test_venv` copy was syntax-validated but was not used to overwrite or rerun the result. The complete successful run used `venv`, which already had the required Qiskit and NumPy dependencies.

## What this validates

This stage validates locally:

1. Availability of Qiskit SamplerV2’s `StatevectorSampler`.
2. Construction of an eight-qubit circuit with a named classical register.
3. Execution through the real local SamplerV2 API.
4. Extraction from `SamplerPubResult.data.meas` and `BitArray.get_counts()`.
5. Classical-register selection.
6. Qiskit bitstring ordering.
7. Shot-total preservation.
8. Conversion to eight ordered finite Pauli-Z expectations.
9. Clear handling of missing-register and malformed-result inputs.

## What remains unverified

This local test does not establish:

- IBM Runtime `SamplerV2` result-container behavior
- IBM backend availability
- Runtime authentication or service-instance configuration
- Hardware transpilation layout handling
- Physical-to-logical qubit remapping after hardware transpilation
- Finite-shot noise or readout error
- Hardware counts for the QCGNN circuit
- Frontend backend selection or end-to-end hardware inference

The next hardware-related implementation must preserve the selected classical register and explicitly account for any transpiler layout before decoding counts.

## Next steps

The local result extraction gate passed. The next controlled stage may:

1. Validate the exact QCGNN circuit’s measurement-register construction locally.
2. Validate layout-aware logical-to-physical mapping after local transpilation.
3. Connect the decoder to a separate Qiskit hardware adapter.
4. Compare decoded hardware measurements with the established simulator baseline.
5. Obtain explicit approval before submitting any IBM Quantum job.

No IBM hardware submission was performed in this stage.

