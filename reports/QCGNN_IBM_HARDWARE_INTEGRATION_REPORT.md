# QCGNN IBM Quantum Hardware Integration Report

**Project root:** `C:\Users\Dell\Desktop\qcgnn_project`  
**Scope:** Layer 3 causal graphs → Layer 4 quantum model → IBM Quantum Runtime → frontend inference  
**Status:** Inspection and architecture report only. No integration implementation or IBM hardware job was performed.

## Executive summary

The project currently has a working local PennyLane `default.qubit` path for a hybrid QCGNN model. Layer 3 stores causal molecular subgraphs as PyTorch Geometric `Data` objects. Layer 4 pools each causal graph into 18 classical features, projects them to eight quantum input angles, executes an eight-qubit circuit, and sends eight Pauli-Z expectation values through a classical readout head to produce HOMO, LUMO, and GAP predictions.

The corrected Layer 4 circuit is implemented in `code/layer4_quantum/quantum_circuit.py`, and the corrected checkpoint is `models/best_qcgnn_corrected.pt`. The checkpoint contains a quantum weight tensor of shape `(2, 8)`.

IBM access has been tested at the account/backend-listing level in the separate `ibm_test_venv` environment. The project has not verified a successful real-QPU circuit execution. A previous `qpu_venv` submission failed because `qiskit-ibm-runtime==0.29.0` serialized circuits as QPY version 11 while the IBM service required version 13 or newer. The current `ibm_test_venv` contains Qiskit 2.6.1 and `qiskit-ibm-runtime==0.51.0`, but it does not contain PennyLane or the PyTorch/PyG model stack.

The lowest-risk integration is a separate Qiskit-native hardware adapter that preserves the existing PennyLane simulator. It should reuse the trained classical projection, quantum weights, and classical readout, while replacing only quantum execution for an explicitly selected IBM backend.

## 1. Project overview

### Existing architecture

```text
QM9 / Layer 2 molecular graphs
        ↓
Layer 3 causal / important-atom subgraphs
        ↓
Layer 4 HybridQuantumGNN
        ├── global mean + max pooling
        ├── classical projection to 8 values
        ├── 8-qubit PennyLane circuit
        └── classical readout to HOMO/LUMO/GAP
        ↓
Denormalized eV predictions
        ↓
Flask frontend and JSON API
```

### Current implementation status

| Area | Verified status |
|---|---|
| Layer 3 causal graph files | Present under `data/qm9/layer3/` |
| Layer 4 model | Implemented in `code/layer4_quantum/quantum_model.py` |
| Corrected circuit | Implemented in `code/layer4_quantum/quantum_circuit.py` |
| Corrected checkpoint | Present at `models/best_qcgnn_corrected.pt` |
| Local simulator | PennyLane `default.qubit` |
| Evaluation | Implemented in `code/layer4_quantum/quantum_evaluate.py` |
| Frontend | Flask application in `qcg_nn_app.py` |
| IBM account/backend listing | Previously successful in `ibm_test_venv` |
| Real IBM circuit execution | **Unverified** |
| Frontend backend selection | Not implemented; frontend currently uses simulator |

## 2. Layer 3 data flow

### Source and persistence

Layer 3 processing is implemented in:

- `layer3_causal.py`
- `code/layer3_qcgnn_model/qcgnn_causal_extractor.py`

The current Layer 4 evaluator loads:

```text
data/qm9/layer3/qm9_causal_homo_test.pkl
```

and falls back to:

```text
data/qm9/layer3/qm9_causal_test.pkl
```

The loader in `code/layer4_quantum/quantum_evaluate.py` is:

```python
def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
```

### Verified serialized contents

The inspected `qm9_causal_homo_test.pkl` is:

```text
container: list
length: 2500
element type: torch_geometric.data.Data
```

An inspected graph contains:

| Field | Verified format |
|---|---|
| `x` | Float tensor `[num_causal_nodes, 9]` |
| `edge_index` | Long tensor `[2, num_edges]` |
| `edge_attr` | Float tensor `[num_edges, 4]` |
| `y` | Float tensor `[1, 3]` |
| `pos` | Float tensor `[num_causal_nodes, 3]` |
| `kept_index` | Tensor identifying retained causal nodes |

The first inspected `qm9_causal_homo_test.pkl` graph had:

```text
x:           [3, 9]
edge_index:  [2, 4]
edge_attr:   [4, 4]
y:           [1, 3]
pos:         [3, 3]
kept_index:  [3]
```

The inspected `qm9_causal_test.pkl` contains 50 graphs; its first graph had five causal nodes.

### Node features

The nine node features are generated in `code/inference/molecule_inference.py` by `_node_features` and correspond to:

1. Atomic number
2. Formal charge
3. Hybridization encoding
4. Atom degree
5. Total hydrogen count
6. Aromaticity flag
7. Membership in a six-membered ring
8. Membership in a five-membered ring
9. Total valence

`config.py` confirms:

```python
NODE_FEATURE_DIM = 9
```

### Edge features

`code/inference/molecule_inference.py` creates four edge features:

1. Bond type
2. Aromaticity flag
3. Ring-membership flag
4. Conjugation flag

Layer 4 currently carries `edge_index` and `edge_attr` but does not use them in the forward computation. The model uses node features and the graph batch index for pooling.

### Targets and scaling

The target order is:

```text
HOMO, LUMO, GAP
```

Layer 2 standardizes targets using training-set statistics stored in:

```text
data/qm9/layer2/norm_stats.pkl
```

Layer 4 training compares model output directly with standardized `batch.y`. Evaluation reverses the standardization once and converts from Hartree to eV using:

```python
value_hartree = value_standardized * std + mean
value_ev = value_hartree * 27.211386245988
```

## 3. Layer 4 architecture

### Model file and class

The model is:

```text
code/layer4_quantum/quantum_model.py
HybridQuantumGNN
```

Its configured dimensions are:

| Component | Configuration |
|---|---|
| Node input | 9 |
| Mean/max pooled representation | 18 |
| Qubits | 8 |
| Trainable rotation layers | 2 |
| Entanglement layers | 1 |
| Entanglement mode | Linear |
| Quantum outputs | 8 |
| Readout hidden width | 16 |
| Prediction outputs | 3 |

### Classical input-angle generation

`HybridQuantumGNN.forward` performs:

```python
h_mean = global_mean_pool(x, batch)
h_max = global_max_pool(x, batch)
h_c = torch.cat([h_mean, h_max], dim=1)
z = self.classical_projection(h_c)
```

`z` has shape `[batch_size, 8]` and is passed directly to the quantum circuit. There is no additional angle normalization or clipping in Layer 4.

### Corrected quantum circuit

The circuit is built by:

```text
code/layer4_quantum/quantum_circuit.py: build_qnode
```

The corrected logical gate sequence is:

1. Input encoding:
   ```text
   RY(z[0]) on qubit 0
   ...
   RY(z[7]) on qubit 7
   ```
2. Trainable layer 0:
   ```text
   RY(weights[0, i]) for i = 0..7
   ```
3. Linear entanglement:
   ```text
   CNOT(0,1), CNOT(1,2), ..., CNOT(6,7)
   ```
4. Trainable layer 1:
   ```text
   RY(weights[1, i]) for i = 0..7
   ```
5. Measurements:
   ```text
   expval(PauliZ(i)) for i = 0..7
   ```

The intended operation counts are:

```text
Input RY gates:       8
Trainable RY gates:  16
Linear CNOT gates:    7
Pauli-Z outputs:      8
```

The earlier duplicate rotation/entanglement loop was removed. The current file contains one trainable rotation/entanglement loop.

### Trainable parameters

`quantum_model.py` declares:

```python
weight_shapes = {"weights": (n_rotation_layers, n_qubits)}
```

The verified checkpoint shape is:

```text
qlayer.weights: (2, 8)
```

### Classical prediction head

The readout is:

```python
nn.Sequential(
    nn.Linear(n_qubits, 16),
    nn.ReLU(),
    nn.Linear(16, num_targets)
)
```

It maps the eight quantum measurements to three standardized predictions in the order:

```text
HOMO, LUMO, GAP
```

## 4. Checkpoint and inference flow

### Evaluator

`code/layer4_quantum/quantum_evaluate.py`:

1. Selects the Layer 3 test dataset.
2. Requires `models/best_qcgnn_corrected.pt`.
3. Constructs `HybridQuantumGNN` using `config.py`.
4. Loads the state dictionary with:
   ```python
   model.load_state_dict(state_dict, strict=True)
   ```
5. Runs batches through the model.
6. Checks finite predictions and targets.
7. Reverses normalization once.
8. Converts Hartree to eV.
9. Calculates MAE, RMSE, and R².

The checkpoint is a state dictionary containing:

```text
classical_projection.weight
classical_projection.bias
qlayer.weights
classical_readout.0.weight
classical_readout.0.bias
classical_readout.2.weight
classical_readout.2.bias
```

### Frontend inference

`qcg_nn_app.py:init_app` currently loads:

```text
models/best_qcgnn.pt
```

This is the baseline checkpoint, not `best_qcgnn_corrected.pt`. The frontend should not be considered aligned with the corrected checkpoint until that selection is explicitly changed in a later approved implementation.

The frontend computes inference in the routes:

- `qcg_nn_app.py:analyze` for a stored QM9 molecule
- `qcg_nn_app.py:analyze_smiles` for a supplied SMILES string

Both routes manually perform:

```text
causal graph
→ global mean/max pooling
→ model.classical_projection
→ model.qlayer
→ model.classical_readout
→ denormalization and eV conversion
```

## 5. Frontend/backend architecture

### Framework

The frontend/backend is a Flask application in:

```text
qcg_nn_app.py
```

The application serves HTML and JSON API endpoints from the same process.

### API endpoints

Verified routes:

```text
POST /api/analyze
POST /api/analyze/smiles
```

The browser code calls these routes through `runPipeline`:

```text
qcg_nn_app.py:517
qcg_nn_app.py:524
```

### Current backend behavior

There is currently no request-level backend selection. Both routes execute the PennyLane model path:

```python
q_out = model.qlayer(z)
pred = model.classical_readout(q_out)
```

The API response exposes:

```json
{
  "quantum": {
    "num_qubits": 8,
    "input": [...],
    "measurements": [...]
  },
  "prediction": {
    "homo": ...,
    "lumo": ...,
    "gap": ...
  }
}
```

The existing response does not identify an IBM backend, job ID, shot count, or hardware execution status.

## 6. IBM Quantum integration status

### Environments

#### Existing PennyLane/Qiskit environment

`qpu_venv` was inspected with:

```text
PennyLane==0.39.0
PennyLane-qiskit==0.39.0
Qiskit==1.3.2
qiskit-ibm-runtime==0.29.0
```

This environment includes the PennyLane integration but is affected by the previously observed QPY compatibility issue.

#### IBM hardware test environment

`ibm_test_venv` was inspected with:

```text
Qiskit==2.6.1
qiskit-ibm-runtime==0.51.0
```

`pip check` passes in that environment. It does not contain PennyLane or PennyLane-Qiskit, so it is not currently a drop-in environment for the PyTorch/PennyLane QCGNN model.

### Account and backend access

The IBM connection test has previously listed:

```text
ibm_fez
ibm_kingston
ibm_marrakesh
```

This verifies account/service backend-listing access at the time of that test. It does not verify successful circuit submission or completion.

Credentials and instance CRNs are intentionally not recorded in this report.

### Previous QPY failure

An earlier real-QPU attempt failed with:

```text
Error code 3211; Job not valid. Circuit deserialization error.
QPY version 11 is lower than the configured minimum version 13.
```

Inspection showed that `qiskit-ibm-runtime==0.29.0` serialized `QuantumCircuit` objects with QPY version 11, while the installed Qiskit package reported QPY version 13. The service rejected the older serialized payload.

The current `ibm_test_venv` uses Qiskit 2.6.1 and Runtime 0.51.0 and successfully lists backends, but a completed real-QPU circuit job remains **unverified**.

### What remains unverified

- Successful submission of the eight-qubit mapped QCGNN circuit
- Successful execution on `ibm_fez`, `ibm_kingston`, or `ibm_marrakesh`
- Correct hardware counts decoding for all eight qubits
- Numerical agreement between Qiskit and PennyLane for the mapped circuit
- End-to-end graph-to-hardware-to-readout inference
- Frontend-controlled backend selection
- Runtime, queue, quota, and cost behavior for the intended workload

## 7. Proposed integration architecture

The simulator should remain the default and should not be replaced by hardware execution implicitly.

```text
Layer 3 PyG Data
        ↓
Classical pooling and projection
        ↓
8 input angles z
        ├── simulator adapter
        │     └── existing PennyLane qlayer
        └── IBM adapter
              └── equivalent Qiskit circuit + SamplerV2
        ↓
8 ordered Z expectations
        ↓
Existing classical_readout
        ↓
standardized HOMO/LUMO/GAP
        ↓
Hartree-to-eV conversion
```

### Hardware adapter responsibilities

The hardware adapter should:

1. Accept one graph-derived `z` vector and the trained `(2, 8)` quantum weights.
2. Build the eight-qubit logical circuit in the exact PennyLane order.
3. Transpile it for an allowlisted IBM backend.
4. Add measurements without changing logical qubit order.
5. Submit through `SamplerV2` only when explicitly authorized.
6. Decode counts into `[Z_0, ..., Z_7]`.
7. Return measurements, backend name, job ID, shots, and status.

The existing classical readout can then be applied to the measurement vector in the model environment:

```python
q_out = torch.tensor(measurements, dtype=torch.float32).reshape(1, 8)
pred = model.classical_readout(q_out)
```

The hardware path should not call `model.qlayer`, because that invokes PennyLane `default.qubit`.

### Environment boundary

The current `ibm_test_venv` does not contain the model dependencies. A later implementation should choose one of:

1. A dedicated environment containing both the required PyTorch/PyG stack and the current IBM Runtime stack.
2. A split process/service: the model environment computes `z` and applies the readout, while the IBM environment handles Qiskit circuit construction and Runtime submission.

The split design minimizes changes to the existing PennyLane environment but requires an explicit, validated data exchange boundary.

## 8. Implementation roadmap

### Stage 1 — Simulator baseline for one molecule

Use one graph from:

```text
data/qm9/layer3/qm9_causal_homo_test.pkl
```

Load `best_qcgnn_corrected.pt` strictly and record:

- Graph identifier if available
- Causal node count
- `z` input angles
- Quantum measurement vector
- Standardized readout output
- HOMO/LUMO/GAP in eV

Do not overwrite existing metrics.

### Stage 2 — Local Qiskit circuit validation

Construct the equivalent eight-qubit Qiskit circuit locally:

```text
for i in 0..7: RY(z[i], i)
for i in 0..7: RY(weights[0, i], i)
for i in 0..6: CX(i, i+1)
for i in 0..7: RY(weights[1, i], i)
measure all qubits
```

Compare local Qiskit expectation values with PennyLane `default.qubit` outputs using identical `z` and weights. This is required before hardware use.

### Stage 3 — Parameter mapping validation

Verify:

- `z` is copied without scaling or reordering.
- `weights[0, i]` and `weights[1, i]` map to the corresponding RY gates.
- Qubit indices remain logical `0..7`.
- The seven CNOT pairs are exactly `(0,1)` through `(6,7)`.
- Measurement order is explicitly documented.

### Stage 4 — Counts-to-expectations conversion

For each measured qubit `i`:

```text
Z_i = (count(bit i = 0) - count(bit i = 1)) / total_shots
```

For standard Qiskit count strings, qubit `i` is read as:

```python
bit = bitstring[-1 - i]
```

The output order must be:

```text
[Z_0, Z_1, ..., Z_7]
```

### Stage 5 — One-molecule QPU smoke test

Before submission:

- Confirm the IBM account and instance.
- Confirm the backend and operational status.
- Confirm the requested shot count and estimated usage/cost.
- Print circuit operation counts.
- Require explicit human confirmation.

Submit exactly one small circuit for one molecule. Record job ID and raw counts. A 128-shot run is an integration smoke test, not a model-quality evaluation.

### Stage 6 — Hardware inference

Apply the decoded eight-value vector to the existing `classical_readout`. Denormalize using the existing Layer 2 statistics and convert to eV exactly once.

Return a structured result containing:

```text
backend
job_id
shots
measurements
standardized_prediction
prediction_eV
```

### Stage 7 — Frontend integration

Add explicit backend selection to request payloads, for example:

```json
{"molecule_id": 51684, "backend": "simulator"}
```

or:

```json
{"smiles": "...", "backend": "ibm"}
```

The server must:

- Allowlist backend names.
- Keep credentials server-side.
- Preserve simulator behavior.
- Reject unknown backends clearly.
- Return backend and job metadata.
- Avoid allowing arbitrary browser-supplied IBM backend names.

### Stage 8 — Simulator-versus-hardware comparison

For the same graph, compare:

- Eight per-qubit expectations
- HOMO difference
- LUMO difference
- GAP difference
- Absolute and relative differences
- Shot count and backend metadata

Label hardware output as noisy experimental execution. Do not treat it as a replacement for simulator validation or as evidence of quantum advantage.

## 9. Risks and validation

| Risk | Required validation |
|---|---|
| Parameter order mismatch | Compare circuit parameters and expectation values against PennyLane |
| Qubit/bit ordering mismatch | Unit-test `bitstring[-1 - i]` conversion with synthetic counts |
| Finite-shot noise | Compare multiple shot counts and report uncertainty |
| Transpilation changes | Inspect transpiled operation counts and preserve logical equivalence |
| Backend constraints | Verify qubit count, coupling map, calibration, and operational status |
| Runtime/QPY incompatibility | Validate the exact installed Qiskit/Runtime pair before submission |
| Gradients | Do not assume hardware `SamplerV2` results are differentiable; hardware inference is forward-only unless a separate gradient strategy is implemented |
| Prediction scaling | Apply standardization reversal once and Hartree-to-eV conversion once |
| Checkpoint mismatch | Load the corrected checkpoint strictly and retain the corrected circuit |
| Baseline contamination | Keep `best_qcgnn.pt`, existing metrics, and simulator results separate |
| Queue and usage cost | Require backend, shots, quota, and cost approval before submission |
| Credential exposure | Use environment variables or secure prompts; never return credentials through the frontend |
| Frontend blocking | Run hardware calls asynchronously or behind a controlled job-status workflow if latency is significant |

### Minimum validation sequence

1. Verify checkpoint tensor shapes.
2. Verify local PennyLane operation counts.
3. Verify local Qiskit operation mapping.
4. Compare noiseless Qiskit and PennyLane expectations.
5. Validate synthetic counts decoding.
6. Run one explicitly approved IBM smoke test.
7. Compare hardware and simulator outputs for the same input.
8. Add frontend routing only after the adapter is independently validated.

## 10. Files to create or modify later

These are proposals only. They were not created or modified for this report.

| Proposed path | Responsibility |
|---|---|
| `code/layer4_quantum/qiskit_circuit_adapter.py` | Build the Qiskit equivalent of the corrected Layer 4 circuit |
| `code/layer4_quantum/measurement_utils.py` | Convert Qiskit counts to ordered Pauli-Z expectations |
| `code/layer4_quantum/hardware_inference.py` | Load/checkpoint bridge, backend selection, Runtime submission, and readout integration |
| `code/layer4_quantum/simulator_reference.py` | One-graph simulator baseline and comparison record |
| `code/layer4_quantum/local_qiskit_validation.py` | No-hardware circuit equivalence and operation-count tests |
| `code/layer4_quantum/ibm_qpu_connection_test.py` | Existing account/backend listing smoke test; keep separate from model inference |
| `code/layer4_quantum/ibm_qpu_real_test.py` | Existing minimal real-QPU test; keep separate from frontend inference |
| `qcg_nn_app.py` | Later add explicit backend selection and structured quantum metadata |
| `reports/QCGNN_IBM_HARDWARE_INTEGRATION_REPORT.md` | This inspection report |

The existing model, circuit, training, evaluation, checkpoint, and dataset files should remain unchanged unless a later implementation approval explicitly scopes a change.

## 11. Acceptance criteria

### Layer 3 and simulator baseline

- One known Layer 3 `Data` object loads successfully.
- `x` has nine features per node.
- `z` has shape `(1, 8)`.
- The corrected checkpoint loads strictly.
- Simulator measurements have eight finite values.
- Prediction output has exactly three values in HOMO/LUMO/GAP order.
- Denormalized output is finite and labeled eV.

### Local Qiskit equivalence

- Qiskit circuit has eight logical qubits.
- Logical gate order matches the corrected PennyLane circuit.
- Operation counts before transpilation match the expected input/trainable RY/CNOT structure.
- Noiseless Qiskit and PennyLane Z expectations agree within a declared numerical tolerance.
- Counts decoding returns exactly eight values in `[Z_0, ..., Z_7]` order.

### IBM Runtime smoke test

- `ibm_test_venv` imports Qiskit and Runtime successfully.
- `pip check` passes.
- The selected backend is available and operational.
- The circuit transpiles successfully for that backend.
- The user confirms backend, shot count, instance, and estimated usage.
- Exactly one approved job is submitted.
- A job ID and counts are returned.
- No credentials appear in output or files.

### Hardware inference

- The same graph and input angles are used for simulator and hardware paths.
- Hardware measurement output has shape `(1, 8)`.
- The existing classical readout accepts the hardware measurements.
- HOMO, LUMO, and GAP predictions are returned in eV.
- Backend name, job ID, shot count, and execution mode are included in the response.

### Frontend integration

- `backend="simulator"` preserves current behavior.
- `backend="ibm"` is explicit and allowlisted.
- Unknown backend values return a clear error.
- IBM credentials never reach browser code.
- Hardware failures do not silently fall back to simulator results.
- Simulator and hardware results can be compared for one identical molecule.

## Final next-steps checklist

- [ ] Approve creation of a local Qiskit circuit-equivalence test.
- [ ] Verify the corrected checkpoint and one Layer 3 graph in the model environment.
- [ ] Decide whether to use a combined inference environment or a split model/IBM adapter process.
- [ ] Validate Qiskit/PennyLane expectations locally before hardware access.
- [ ] Confirm IBM backend, instance, shot count, quota, and estimated cost.
- [ ] Run one explicitly approved hardware smoke test.
- [ ] Compare simulator and hardware measurements for the same molecule.
- [ ] Approve frontend API changes with explicit backend selection.
- [ ] Keep simulator and hardware metrics/results separately labeled.

