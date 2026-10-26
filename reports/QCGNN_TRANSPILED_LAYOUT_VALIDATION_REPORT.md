# QCGNN Transpiled Layout Validation Report

**Project root:** `C:\Users\Dell\Desktop\qcgnn_project`  
**Validation type:** Local Qiskit transpilation, layout, measurement, and decoding test  
**Qiskit version:** `2.6.1`  
**IBM Runtime contacted:** No  
**Hardware job submitted:** No

## Executive summary

The exact corrected QCGNN quantum circuit was rebuilt from the established simulator baseline and the trained checkpoint, then transpiled locally with Qiskit. The transpiled circuit included explicit measurements into the named `meas` classical register.

The validation passed:

```text
Logical-to-physical mapping: [0, 1, 2, 3, 4, 5, 6, 7]
Maximum exact statevector difference: 6.66133814775e-16
Maximum sampled decoded difference: 0.019578539649
Result: PASS
```

The exact statevector comparison used a tolerance of `1e-8`. The finite-shot SamplerV2 comparison used a tolerance of `0.08` with `8192` shots. No IBM service, credentials, backend query, or hardware job was used.

## Files inspected

| File | Relevance |
|---|---|
| `code/layer4_quantum/quantum_circuit.py` | Corrected PennyLane gate order |
| `code/layer4_quantum/quantum_model.py` | Eight-qubit model and trained quantum-weight shape |
| `code/layer4_quantum/local_qiskit_validation.py` | Prior local Qiskit circuit construction |
| `code/layer4_quantum/measurement_utils.py` | Existing counts-to-ordered-Z decoder |
| `code/layer4_quantum/sampler_result_test.py` | Existing local SamplerV2 result extraction |
| `results/quantum_simulator_baseline_corrected.json` | Exact graph-derived input angles |
| `reports/QCGNN_QISKIT_CIRCUIT_EQUIVALENCE_REPORT.md` | Prior untranspiled equivalence result |
| `reports/QCGNN_MEASUREMENT_DECODING_REPORT.md` | Bitstring ordering validation |
| `reports/QCGNN_SAMPLER_RESULT_VALIDATION_REPORT.md` | Local SamplerV2 structure validation |

## New artifacts

| File | Purpose |
|---|---|
| `code/layer4_quantum/transpiled_layout_test.py` | Builds, transpiles, maps, samples, and validates the exact circuit |
| `results/quantum_transpiled_layout_test.json` | Actual layout, counts, expectations, tolerances, and assertions |
| `reports/QCGNN_TRANSPILED_LAYOUT_VALIDATION_REPORT.md` | This report |

No production model, circuit, checkpoint, training, evaluation, dataset, frontend, or environment file was modified.

## Environment and transpilation

The validation ran with:

```cmd
venv\Scripts\python.exe code\layer4_quantum\transpiled_layout_test.py
```

Installed Qiskit:

```text
2.6.1
```

The local transpilation configuration was:

```text
Qubits:           8
Optimization:     level 1
Basis gates:      rz, sx, x, cx
Coupling topology: bidirectional linear chain 0 ↔ 1 ↔ ... ↔ 7
Transpiler seed:   11
```

The coupling map was supplied locally and was not obtained from an IBM backend.

## Exact circuit mapping

The script reused the eight input angles from:

```text
results/quantum_simulator_baseline_corrected.json
```

It loaded `qlayer.weights` from:

```text
models/best_qcgnn_corrected.pt
```

The reconstructed gate order was:

1. Eight input `RY(input_angles[qubit])` gates.
2. Eight trainable `RY(weights[0, qubit])` gates.
3. Seven nearest-neighbor `CX(qubit, qubit + 1)` gates.
4. Eight trainable `RY(weights[1, qubit])` gates.
5. Eight measurements into the named `meas` register.

The checkpoint quantum-weight shape was verified as `[2, 8]`.

## Transpiled layout and measurement mapping

The transpiled circuit returned a complete `TranspileLayout`. Its verified logical-to-physical mapping was:

| Logical qubit | Physical qubit |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |

The mapping was read from `layout.final_index_layout(filter_ancillas=True)` rather than assumed.

The measurement instructions were inspected directly from the transpiled circuit. The physical-to-classical-bit mapping was:

| Physical qubit | Classical bit in `meas` |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |

Because this local transpilation produced an identity layout and one-to-one measurement mapping, the physical count strings already corresponded to logical count strings. The script nevertheless performs an explicit remapping using the inspected layout and measurement instructions; it does not rely on the identity result.

For a non-identity layout, logical qubit `i` would be read from the classical bit assigned to its mapped physical qubit. The decoder then interprets the resulting logical bitstring using the established Qiskit convention that logical qubit `0` is the rightmost bit.

## Exact statevector verification

The original circuit and transpiled circuit with final measurements removed were evaluated with local Qiskit statevectors. The transpiled physical expectations were mapped back to logical order before comparison.

| Check | Result |
|---|---|
| Statevector tolerance | `1.0e-08` |
| Maximum absolute difference | `6.661338147750939e-16` |
| Statevector mapping | **PASS** |

The transpiled statevector expectations were:

```text
[
  -0.21838443429100943,
   0.03509984126735682,
  -0.06220856972604823,
  -0.46101929876432160,
   0.69861571480953810,
  -0.48258635677990025,
   0.27346288300479950,
  -0.42421718362748530
]
```

These match the local Qiskit reference statevector within numerical precision.

## Finite-shot counts and decoding

The transpiled measured circuit was executed with local:

```python
StatevectorSampler(seed=1234)
```

Requested and observed shots:

```text
Requested: 8192
Observed:  8192
```

Counts were extracted from the actual named-register path:

```text
result[0].data.meas.get_counts()
```

The complete physical and remapped logical count dictionaries are stored in the JSON result. Since the verified mapping was identity, they are equal for this run.

The existing decoder returned the following ordered logical expectations:

```text
[
  -0.2275390625,
   0.0205078125,
  -0.081787109375,
  -0.462158203125,
   0.70751953125,
  -0.489501953125,
   0.29052734375,
  -0.434814453125
]
```

Absolute differences from the exact reference were:

```text
[
  0.009154628208990517,
  0.014592028767356498,
  0.019578539648951740,
  0.001138904360677928,
  0.008903816440461232,
  0.006915596345099417,
  0.017064460745200516,
  0.010597269497514883
]
```

The maximum finite-shot difference was `0.019578539649`, below the sampling tolerance `0.08`. These differences are sampling variation, not a circuit-mapping failure.

## Validation results

| Assertion | Result |
|---|---|
| Corrected baseline loaded | Passed |
| Corrected checkpoint loaded | Passed |
| Exact input angles reused | Passed |
| Exact trained quantum weights reused | Passed |
| Eight input rotations reproduced | Passed |
| Two trainable rotation layers reproduced | Passed |
| Seven nearest-neighbor CNOTs reproduced | Passed |
| Eight logical measurements present | Passed |
| Complete transpiler layout obtained | Passed |
| Logical-to-physical mapping inspected | Passed |
| All logical qubits measured | Passed |
| Exact statevector mapping within `1e-8` | Passed |
| Count total equals `8192` | Passed |
| Decoded output has eight values | Passed |
| Decoded output is finite | Passed |
| Sampled output within `0.08` | Passed |
| IBM Runtime contacted | No |
| Hardware job submitted | No |

## Collision protection

The script refuses to overwrite:

```text
results/quantum_transpiled_layout_test.json
```

A second execution was attempted after the successful run and stopped with:

```text
FileExistsError: Refusing to overwrite validation result:
C:\Users\Dell\Desktop\qcgnn_project\results\quantum_transpiled_layout_test.json
```

## What was verified locally

This stage verified:

1. The corrected trained circuit can be transpiled locally with Qiskit `2.6.1`.
2. The transpiler exposes a complete layout.
3. Measurement instructions expose an explicit physical-to-classical mapping.
4. The mapped transpiled statevector preserves all eight logical Z expectations.
5. Actual local SamplerV2 counts can be remapped and decoded in logical order.
6. The finite-shot decoded values remain within the documented sampling tolerance.

## What remains unverified

This test does not establish:

- IBM backend topology or backend-specific layout
- IBM Runtime authentication or result-container behavior
- Hardware transpiler choices
- Physical-qubit calibration or noise
- Hardware readout error
- Real-QPU counts
- Frontend backend selection
- End-to-end hardware inference

The identity mapping observed here is specific to this local coupling map and transpiler configuration. It must not be assumed for an IBM backend. A future hardware adapter must inspect and preserve the backend-specific layout before decoding counts.

## Next steps

The local transpilation and layout gate passed. The next controlled stage may:

1. Add backend-independent layout-aware circuit preparation to a separate hardware adapter.
2. Validate backend-specific measurement-register extraction without submitting a job.
3. Preserve logical-to-physical mappings through IBM transpilation.
4. Feed decoded logical expectations into the existing classical prediction head.
5. Obtain explicit approval before submitting one controlled IBM Quantum job.

No hardware integration or accuracy claim is made by this local validation.

