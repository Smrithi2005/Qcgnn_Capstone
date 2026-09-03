# QCGNN: Quantum-Causal Graph Neural Network for Molecular Property Prediction

## 1. PROJECT OVERVIEW
**QCGNN** is a hybrid machine-learning pipeline designed to predict quantum-chemical molecular properties using a combination of classical Graph Neural Networks (GNNs), gradient-based causal subgraph extraction, and hybrid quantum-classical computing. 

The project investigates whether stripping a molecular graph down to its most "causally relevant" structural components (a reduced subgraph) improves the ability of parameterized quantum circuits to model complex molecular orbital energies.

**Overall Pipeline:**
`QM9 Dataset` → `Layer 1 (Data Extraction)` → `Layer 2 (PyG Graph Construction)` → `Layer 3 (Causal Analysis)` → `Layer 4 (Hybrid Quantum-Classical QCGNN)` → `HOMO / LUMO / GAP Predictions`

*Note: The term "causal" in this repository refers to gradient-based attribution on graphs, not proven physical causality. Results reflect local quantum simulations (PennyLane) and do not currently claim physical quantum advantage.*

---

## 2. RESEARCH OBJECTIVE
To systematically investigate the performance of classical vs. hybrid quantum-classical GNNs on full molecular graphs vs. reduced causal subgraphs. 

Current prediction targets are:
- **HOMO** (Highest Occupied Molecular Orbital Energy)
- **LUMO** (Lowest Unoccupied Molecular Orbital Energy)
- **GAP** (HOMO-LUMO Energy Difference)

*Future extensions (currently audited but not actively predicted) include Dipole Moment, Isotropic Polarizability, and Heat Capacity (Cv).*

---

## 3. DATASET
**Dataset:** QM9 (Subset of up to 133,885 stable organic molecules).
- **Molecular Data:** Sourced from raw `.xyz` files containing 3D atomic coordinates.
- **Molecular Graphs:** Nodes represent atoms; Edges represent chemical bonds inferred via 3D Van der Waals radii.
- **Nodes:** Variable up to 18 atoms.
- **Edges:** Bidirectional edge pairs.
- **Node Features (9):** Atomic properties (e.g., atomic number, charge states).
- **Edge Features (4):** Bond properties (e.g., interatomic distances).
- **Targets (3):** HOMO, LUMO, GAP extracted directly from line 2 of the `.xyz` files.
- **Splits:** Pre-defined 80/10/10 Train/Validation/Test split generated via random seed.

---

## 4. PROJECT ARCHITECTURE

```text
                    QM9 DATA (.xyz)
                        |
                        v
              +------------------+
              | Layer 1          |
              | Data Extraction  |
              +--------+---------+
                       |
                       v
              +------------------+
              | Layer 2          |
              | Graph Construction|
              +--------+---------+
                       |
                   Full Graph
                       |
                       v
              +------------------+
              | Layer 3          |
              | Causal Analysis  |
              +--------+---------+
                       |
                 Causal Graph
                       |
                       v
              +------------------+
              | Layer 4          |
              | QCGNN            |
              +--------+---------+
                       |
                       v
             HOMO / LUMO / GAP
```

---

## 5. LAYER 1
**File:** `qcgnn_preprocessing.py` (Functions integrating with `qm9_preprocess_dataset.py`)
- **Input Format:** Raw `.xyz` coordinate files in `data/qm9/raw/`.
- **Information Extracted:** Atomic symbols, 3D positions, and scalar target properties.
- **Output:** Intermediate dictionaries passed directly in memory to Layer 2 converters. 

---

## 6. LAYER 2
**File:** `qcgnn_preprocessing.py`
- **Graph Construction:** Builds PyTorch Geometric (PyG) `Data` objects. Edges are inferred by calculating pairwise 3D Euclidean distances and filtering them against atomic Van der Waals (vdW) radii scaled by a `BOND_THRESHOLD_FACTOR` (1.1).
- **Target Normalization:** Applies Z-score normalization `(y - mean) / std` calculated strictly on the training split to prevent data leakage. Normalization constants are saved in `data/qm9/layer2/norm_stats.pkl`.
- **Output:** `qm9_train.pkl`, `qm9_val.pkl`, `qm9_test.pkl` in `data/qm9/layer2/`.

---

## 7. LAYER 3 — CAUSAL ANALYSIS
**File:** `code/layer3_qcgnn_model/qcgnn_causal_extractor.py`
- **Methodology:** Gradient-based causal attribution.
- **Architecture:** A classical GCN forward pass predicts the target from node features. A backward pass calculates the gradient magnitude of the loss with respect to the input node features (`d(loss)/d(node_features)`).
- **Subgraph Construction:** Nodes with the highest gradient magnitude are marked as structurally "important". The graph is pruned to keep only the top `CAUSAL_MIN_ATOMS` (3) to `CAUSAL_MAX_ATOMS` (12) nodes (approx 20% of original size). Edges interconnecting the remaining nodes are preserved.
- **Output:** Causal PyG objects saved as `.pkl` in `data/qm9/layer3/` (e.g., `qm9_causal_homo_train.pkl`).

---

## 8. LAYER 4 — QUANTUM QCGNN
**Files:** `code/layer4_quantum/quantum_model.py`, `quantum_circuit.py`
- **Graph Pooling:** Because causal graphs vary in size (3-12 nodes), `global_mean_pool` and `global_max_pool` are concatenated to create a fixed `18-dimensional` classical representation.
- **Classical Projection:** An `nn.Linear` layer projects the 18-dim vector to an `8-dimensional` vector.
- **Quantum Encoding:** The 8 values map to `RY(theta)` angle encoding across **8 qubits**.
- **Quantum Circuit:** `n_rotation_layers = 2`, `n_entangle_layers = 1`. Variational layers utilize parameterized `RY` gates followed by linear `CNOT` entanglement (adjacent qubits).
- **Measurement:** Pauli-Z expectation values (`<Z_i>`) yield `8 measurements`.
- **Classical Readout:** A classical MLP (`8 -> 16 -> 3`) translates measurements into predictions.

---

## 9. QUANTUM MATHEMATICS
The circuit initializes in the vacuum state $|0\rangle^{\otimes 8}$. 
1. **Encoding:** Data is embedded via angle encoding $R_Y(\theta) = e^{-i \theta Y / 2}$.
2. **Ansatz (Parameterized Circuit):** Trainable rotations $U(\theta)$ explore the Hilbert space.
3. **Entanglement:** $CNOT$ gates correlate adjacent qubits, allowing the circuit to represent non-linear interactions across the atomic features.
4. **Measurement:** Expectation values of the Pauli-Z operator $\langle Z_i \rangle$ collapse the quantum state into classical numeric bounds $[-1, 1]$.
*(All simulations are performed locally via PennyLane's default simulator; no physical quantum hardware is utilized).*

---

## 10. HYBRID DIFFERENTIATION
The PennyLane quantum node (`qnode`) is wrapped using `qml.qnn.TorchLayer`. This seamless interface allows PennyLane's internal differentiation (e.g., parameter-shift rules or backprop) to automatically attach to the PyTorch `autograd` computational graph. The classical optimizer backpropagates the loss continuously through the classical readout, quantum circuit, and classical projection.

---

## 11. TRAINING
**Optimizer:** Adam (`lr = 1e-3`)
**Loss Function:** Mean Squared Error (MSE) on Z-scored targets.
**Batch Size:** 32 (Layer 4), 16 (Layer 3)
**Epochs:** 100
**Early Stopping:** Patience of 15 epochs based on validation loss.
*(One epoch represents one complete forward and backward pass over the entire training subset).*

---

## 12. CHECKPOINTS
Located in `models/`:
- `best_classical_gnn.pt`: Experiment A (Full + Classical)
- `best_causal_classical_gnn.pt`: Experiment B (Causal + Classical)
- `best_qcgnn.pt`: Experiment C (Causal + Quantum)
- `best_full_qcgnn.pt`: Experiment D (Full + Quantum)

---

## 13. EXPERIMENTS
The project forms a 2x2 experimental matrix isolating the effects of Graph Representation vs. Model Architecture:

|             | CLASSICAL MODEL | QUANTUM MODEL |
|-------------|-----------------|---------------|
| **FULL GRAPH** | Experiment A    | Experiment D  |
| **CAUSAL GRAPH**| Experiment B    | Experiment C  |

---

## 14. EXPERIMENT A (Full Graph + Classical GENConv)
- **Setup:** The raw, 18-atom molecules passed through a state-of-the-art classical 3-layer `GENConv` network.
- **Results:** Serves as the primary classical baseline. Showed high performance (HOMO MAE: 0.212 eV), demonstrating standard deep learning capability on dense data.

## 15. EXPERIMENT B (Causal Graph + Classical GENConv)
- **Setup:** Classical `GENConv` network forced to learn from the 3-12 atom reduced causal subgraphs.
- **Results:** Performance dropped (HOMO MAE: 0.368 eV), showing that standard classical convolution heavily relies on the contextual "noise" of the full graph.

## 16. EXPERIMENT C (Causal Graph + QCGNN)
- **Setup:** The proposed novel hybrid architecture.
- **Results:** Achieved HOMO MAE: 0.408 eV. While not surpassing the full classical baseline, it successfully modeled chemical properties using significantly fewer parameters and restricted subgraph data.

## 17. EXPERIMENT D (Full Graph + QCGNN)
- **Setup:** The 8-qubit QCGNN exposed directly to the noisy 18-atom full graph.
- **Results:** Achieved HOMO MAE: 0.384 eV. 

---

## 18. RESULTS TABLE
*Metrics extracted from actual `results/*_test_metrics.json`. Errors are Mean Absolute Error (MAE) evaluated on the independent test set in physical units (eV).*

| Experiment | Graph | Model | HOMO MAE (eV) | LUMO MAE (eV) | GAP MAE (eV) |
|------------|-------|-------|---------------|---------------|--------------|
| Exp A | Full | Classical GENConv | 0.2122 | 0.3068 | 0.3501 |
| Exp B | Causal | Classical GENConv | 0.3675 | 0.7018 | 0.7191 |
| Exp D | Full | Hybrid QCGNN | 0.3836 | 0.7471 | 0.7847 |
| Exp C | Causal | Hybrid QCGNN | 0.4078 | 0.7709 | 0.8143 |

---

## 19. VISUALIZATIONS
Automated plotting is handled by `code/visualization/visualize_results.py`. 
- **Stored in:** `results/visualizations/`
- **Generates:** Training loss curves, comparative bar charts (MAE/RMSE/R²), model comparison CSVs, and isolated metric tracking for all 4 experiments.

---

## 20. INSTALLATION
**Core Dependencies:**
- Python 3.x
- PyTorch
- PyTorch Geometric (PyG)
- PennyLane
- Flask (for demo frontend)
- Matplotlib / Pandas / NumPy

---

## 21. ENVIRONMENT SETUP
*(Assuming Windows PowerShell environment)*
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision torchaudio
pip install torch_geometric
pip install pennylane flask pandas matplotlib
```

---

## 22. RUNNING THE PROJECT
**Data Preprocessing (Layers 1 & 2):**
```powershell
python qcgnn_preprocessing.py
```
**Causal Extraction (Layer 3):**
```powershell
python layer3_causal.py
```
**Model Training / Evaluation:**
```powershell
# Experiment A
python code/baselines/classical_gnn/classical_gnn_train.py
python code/baselines/classical_gnn/classical_gnn_evaluate.py

# Experiment B
python code/baselines/causal_classical_gnn/causal_classical_gnn_train.py
python code/baselines/causal_classical_gnn/causal_classical_gnn_evaluate.py

# Experiment C
python code/layer4_quantum/quantum_train.py
python code/layer4_quantum/quantum_evaluate.py

# Experiment D
python code/experiments/experiment_D_full_qcgnn/full_qcgnn_train.py
python code/experiments/experiment_D_full_qcgnn/full_qcgnn_evaluate.py
```
**Run Visualizations:**
```powershell
python code/visualization/visualize_results.py
```
**Run Frontend Demo Application:**
```powershell
python qcg_nn_app.py
```

---

## 23. FRONTEND / DEMO
**File:** `qcg_nn_app.py`
A single-file Flask application that provides a professional web dashboard.
- Features a QM9 molecule dropdown (utilizing the test set).
- Renders an interactive D3.js SVG molecular graph.
- Highlights the causal subgraph dynamically.
- Renders the 8-qubit quantum circuit and animates Pauli-Z measurements.
- Outputs real-time model predictions (HOMO, LUMO, GAP in eV) using the `best_qcgnn.pt` checkpoint.

**Arbitrary SMILES Support:**
The frontend now supports inference on arbitrary SMILES strings via an integrated RDKit 3D conformer generation adapter (`code/inference/molecule_inference.py`). It embeds SMILES into exactly the same 9-dimensional node / 4-dimensional edge PyG structure expected by the model.

---

## 24. INFERENCE-TIME CAUSAL ATTRIBUTION
The original training-time Layer 3 causal extractor (`qcgnn_causal_extractor.py`) utilized a Mean Squared Error gradient computation against the **ground-truth target** (`data.y`).

Because arbitrary molecules entered at inference time (e.g. via SMILES) do not possess a ground-truth label, running the training-time extractor would require fabricating labels. 

To preserve scientific integrity without sacrificing interpretability, inference uses **Prediction-Gradient Atom Attribution** (`code/inference/causal_attribution_inference.py`):
1. **Mathematical Formulation:** The full input graph $X$ is passed through the trained QCGNN checkpoint $f_\theta(X)$. We sum the absolute gradients of the scalar prediction magnitude $P$ with respect to the input node features:
$S_i = \sum_{j=1}^9 \left| \frac{\partial P}{\partial X_{ij}} \right|$
2. **Ranking:** Atoms are ranked by $S_i$. 
3. **Subgraph Construction:** The top $N_c$ topologically important atoms (matching the 20% pruning fraction from training) are retained to form the inference-time attribution subgraph.
4. **Final Inference:** This subgraph is passed back to the QCGNN to generate final properties.

*This isolates training-time selection from inference-time topological interpretation, ensuring no fake data is generated.*

---

## 25. CURRENT LIMITATIONS
- **Simulation:** Results rely on local, noise-free state-vector simulation via PennyLane, which does not account for physical quantum hardware noise (e.g., decoherence or gate errors).
- **Causality vs. Attribution:** The word "causal" in this repository refers to gradient-based attribution heuristics, which identifies structural correlations critical to the neural network rather than proving fundamental physical causality.

---

## 25. RESEARCH INTEGRITY: SCIENTIFIC INTERPRETATION
- **Predictions vs. Experiments:** Values output by this repository are computational model predictions, not experimental laboratory measurements.
- **Hardware Limitations:** Simulator results are not equivalent to quantum hardware results.
- **Quantum Advantage:** This project explores hybrid architectures but **does not** claim physical quantum advantage over classical models, as classical GENConv currently achieves lower error margins.

---

## 26. FUTURE WORK
1. **Multi-property prediction:** Expanding targets to include Dipole Moment, Isotropic Polarizability, and Heat Capacity (Cv).
2. **Arbitrary SMILES support:** Creating a validated RDKit 3D conformer generation adapter.
3. **Hardware Execution:** Deploying the trained circuit to IBM Quantum hardware.
4. **Qubit Ablations:** Analyzing model stability across 4 vs. 8 vs. 12 qubit projections.

---

## 27. REPRODUCIBILITY
The project features a highly reproducible environment governed by `config.py`. A global random seed ensures deterministic splitting of the 133k QM9 dataset. Checkpoint saving limits variance, and test-set evaluations use denormalization math identical across all scripts. 

---

## 28. COMPLETE PROJECT TREE
```text
qcgnn_project/
├── code/
│   ├── baselines/
│   │   ├── causal_classical_gnn/
│   │   └── classical_gnn/
│   ├── experiments/
│   │   └── experiment_D_full_qcgnn/
│   ├── layer3_qcgnn_model/
│   │   └── qcgnn_causal_extractor.py
│   ├── layer4_quantum/
│   │   ├── quantum_circuit.py
│   │   ├── quantum_model.py
│   │   ├── quantum_train.py
│   │   └── quantum_evaluate.py
│   └── visualization/
│       └── visualize_results.py
├── data/
│   └── qm9/
│       ├── raw/
│       ├── layer2/
│       └── layer3/
├── models/
│   ├── best_causal_classical_gnn.pt
│   ├── best_classical_gnn.pt
│   ├── best_full_qcgnn.pt
│   └── best_qcgnn.pt
├── results/
│   ├── visualizations/
│   └── *metrics.json
├── config.py
├── layer3_causal.py
├── qcgnn_preprocessing.py
├── qm9_preprocess_dataset.py
└── qcg_nn_app.py
```

---

## 29. CITATIONS / REFERENCES
Currently, there is no formal citation information embedded in the repository. (If you use this codebase, please link back to this repository).

## 30. AUTHOR / PROJECT INFORMATION
Project contributors: See repository commit history.
