# QCGNN Qiskit Circuit Equivalence Report

**Project root:** `C:\Users\Dell\Desktop\qcgnn_project`  
**Validation stage:** Local, noiseless Qiskit statevector comparison  
**IBM Runtime contacted:** No  
**Hardware job submitted:** No

## Executive summary

The corrected Layer 4 quantum circuit was reconstructed locally with Qiskit and compared with the recorded PennyLane `default.qubit` baseline for the same Layer 3 graph.

The validation passed:

```text
Maximum absolute Pauli-Z difference: 3.72627137257e-07
Tolerance:                         1.0e-06
Result:                            PASS
```

The comparison used the same eight graph-derived input angles and the exact trained quantum weights loaded from:

```text
models/best_qcgnn_corrected.pt
```

The Qiskit measurements were mapped in logical qubit order `0` through `7`. The existing classical readout was also applied numerically to the Qiskit measurements; its standardized predictions differed from the recorded baseline by at most `2.41940888657e-07`.

This validates the local logical circuit mapping only. It does not validate IBM transpilation, hardware noise, Runtime submission, or frontend hardware inference.

## Files inspected

| File | Use in validation |
|---|---|
| `code/layer4_quantum/quantum_circuit.py` | Verified the corrected PennyLane gate sequence and measurement order |
| `code/layer4_quantum/quantum_model.py` | Verified the eight-qubit model and classical readout structure |
| `code/layer4_quantum/simulator_baseline.py` | Verified how graph angles and baseline measurements are produced |
| `results/quantum_simulator_baseline_corrected.json` | Source of the exact graph-derived angles and recorded PennyLane measurements |
| `models/best_qcgnn_corrected.pt` | Source of trained quantum weights and classical readout parameters |

New validation artifacts:

| File | Purpose |
|---|---|
| `code/layer4_quantum/local_qiskit_validation.py` | Reconstructs and compares the local Qiskit circuit |
| `results/quantum_qiskit_equivalence_corrected.json` | Machine-readable comparison result |
| `reports/QCGNN_QISKIT_CIRCUIT_EQUIVALENCE_REPORT.md` | This report |

No existing project source, checkpoint, dataset, training, evaluation, or frontend file was modified.

## Reference data

The validation used:

```text
Baseline:   results/quantum_simulator_baseline_corrected.json
Checkpoint: models/best_qcgnn_corrected.pt
Graph:      graph_index 0 from data/qm9/layer3/qm9_causal_homo_test.pkl
Target order: HOMO, LUMO, GAP
```

The baseline contains eight finite input angles and eight finite PennyLane expectation values. The selected graph has no molecule identifier, so the validation uses `graph_index: 0` without inventing an ID.

## Environment verification

The existing `ibm_test_venv` was checked without installing or changing packages:

```text
Qiskit: 2.6.1
Statevector: available
PyTorch: not installed
```

Because the corrected checkpoint is a PyTorch checkpoint, `ibm_test_venv` cannot load its trained weights. Running the validation script there stops with:

```text
ModuleNotFoundError: No module named 'torch'
```

The complete validation therefore ran in the existing `venv`, which already contains both Qiskit 2.6.1 and PyTorch. No dependency changes were made.

The local simulator used was:

```python
Statevector.from_instruction(circuit)
```

No `qiskit_aer` dependency and no IBM Runtime dependency were required.

## Circuit mapping

The Qiskit circuit has eight logical qubits and reproduces the corrected PennyLane sequence exactly:

1. `RY(input_angles[qubit])` for qubits `0..7`.
2. `RY(quantum_weights[0, qubit])` for qubits `0..7`.
3. `CX(qubit, qubit + 1)` for qubits `0..6`.
4. `RY(quantum_weights[1, qubit])` for qubits `0..7`.
5. One Pauli-Z expectation for each logical qubit `0..7`.

The trained quantum weight tensor was loaded from the checkpoint and verified to have shape:

```text
[2, 8]
```

The script does not retrain, modify, or save the checkpoint.

## Qubit and measurement ordering

Qiskit Pauli labels use the conventional string order in which the rightmost character corresponds to Qiskit qubit `0`. To measure logical qubit `i`, the script constructs:

```text
"I" * (7 - i) + "Z" + "I" * i
```

The labels used were:

| Logical qubit | Qiskit Pauli label |
|---:|---|
| 0 | `IIIIIIIZ` |
| 1 | `IIIIIIZI` |
| 2 | `IIIIIZII` |
| 3 | `IIIIZIII` |
| 4 | `IIIZIIII` |
| 5 | `IIZIIIII` |
| 6 | `IZIIIIII` |
| 7 | `ZIIIIIII` |

This explicitly preserves the model’s measurement order: `[Z_0, Z_1, ..., Z_7]`.

## Measurement comparison

Tolerance:

```text
1.0e-06 absolute difference per logical qubit
```

| Qubit | PennyLane expectation | Qiskit expectation | Absolute difference |
|---:|---:|---:|---:|
| 0 | -0.21838462352752686 | -0.21838443429100948 | 1.89236517373e-07 |
| 1 | 0.03509983047842979 | 0.03509984126735650 | 1.07889267034e-08 |
| 2 | -0.06220864877104759 | -0.06220856972604826 | 7.90449993307e-08 |
| 3 | -0.46101954579353330 | -0.46101929876432207 | 2.47029211253e-07 |
| 4 | 0.69861608743667600 | 0.69861571480953880 | 3.72627137257e-07 |
| 5 | -0.48258659243583680 | -0.48258635677990060 | 2.35655936209e-07 |
| 6 | 0.27346298098564150 | 0.27346288300479950 | 9.79808419954e-08 |
| 7 | -0.42421737313270570 | -0.42421718362748510 | 1.89505220571e-07 |

Maximum absolute difference:

```text
3.7262713725727536e-07
```

Since the maximum is below `1.0e-06`, the local circuit equivalence result is **PASS**.

## Classical readout comparison

The Qiskit measurement vector was passed through the checkpoint’s existing classical readout parameters:

```text
Linear(8 → 16) → ReLU → Linear(16 → 3)
```

No normalization reversal or Hartree-to-eV conversion was applied in this comparison.

| Target | Recorded baseline standardized prediction | Qiskit standardized prediction | Absolute difference |
|---|---:|---:|---:|
| HOMO | -0.24715465307235718 | -0.24715441113146852 | 2.41940888657e-07 |
| LUMO | -0.31362456083297730 | -0.31362450531149555 | 5.55214817455e-08 |
| GAP | -0.18753270804882050 | -0.18753278749159905 | 7.94427785500e-08 |

The largest standardized prediction difference is:

```text
2.419408886566998e-07
```

The small differences are consistent with the recorded PennyLane baseline having been serialized at finite precision before this independent statevector calculation.

## Validation command

Run the completed validation in the existing environment with:

```cmd
venv\Scripts\python.exe code\layer4_quantum\local_qiskit_validation.py
```

The script refuses to overwrite the result file:

```text
results/quantum_qiskit_equivalence_corrected.json
```

It exits successfully only when all measurement differences are within the configured tolerance.

## Validation outcome

| Check | Result |
|---|---|
| Existing baseline loaded | Passed |
| Corrected checkpoint found | Passed |
| Exact trained quantum weights loaded | Passed |
| Quantum weight shape `[2, 8]` verified | Passed |
| Eight input RY gates mapped | Passed |
| Eight trainable layer-0 RY gates mapped | Passed |
| Seven linear CNOT gates mapped | Passed |
| Eight trainable layer-1 RY gates mapped | Passed |
| Logical qubit ordering verified | Passed |
| Eight Pauli-Z values compared | Passed |
| Maximum difference within `1e-6` | Passed |
| Classical standardized readout compared | Passed |
| IBM Runtime contacted | No |
| Hardware job submitted | No |

## Errors and limitations

The `ibm_test_venv` environment cannot run the complete checkpoint-based validation because PyTorch is absent. It was not modified. The existing `venv` was used instead because it already provides the required Qiskit and PyTorch packages.

This stage does not establish:

- Hardware backend availability at execution time
- IBM Runtime or SamplerV2 compatibility
- Hardware transpilation equivalence
- Hardware measurement counts
- Finite-shot statistical uncertainty
- Noise-model behavior
- Frontend routing or backend selection
- End-to-end hardware inference

## Next steps

The equivalence check passed, so the project may proceed to the next controlled stage:

1. Implement a local counts-to-Pauli-Z conversion utility.
2. Validate that utility against analytically known two-qubit bitstring distributions.
3. Build a Qiskit Runtime adapter without changing the existing PennyLane simulator path.
4. Validate transpiled-circuit metadata locally before any submission.
5. Obtain explicit approval before submitting a single controlled IBM hardware job.
6. Compare hardware predictions against this simulator baseline using the same graph, angles, trained weights, target order, and conversion convention.

Until those stages are complete, no claim of real-QPU inference or hardware accuracy should be made.

