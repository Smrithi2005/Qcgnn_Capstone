// =============================================================================
// QCGNN Hybrid Quantum Circuit — OpenQASM 3.0 Export
// Faithful representation of code/layer4_quantum/quantum_circuit.py
//   with config: n_qubits=8, n_rotation_layers=2, n_entangle_layers=1
//
// IMPORTANT — DUPLICATE GATE LOGIC IN SOURCE CODE
// ------------------------------------------------
// The source file quantum_circuit.py contains TWO loop structures that both
// emit rotation + entanglement gates at runtime:
//
//   Loop A (lines 27-55): An explicit weights[0,:] RY layer, then a
//        `while layer_idx < n_rotation_layers` loop that conditionally adds
//        more RY layers and CNOT entanglement.
//
//   Loop B (lines 62-71): A `for layer in range(n_rotation_layers)` loop
//        that re-applies ALL rotation layers and entanglement from scratch.
//
// Both loops execute — PennyLane does not "reset" in between. This QASM file
// reproduces the exact gate sequence that PennyLane actually runs.
//
// PARAMETER CLASSIFICATION
// -------------------------
// - inputs[0..7]  : DYNAMIC (runtime) — computed by classical_projection from
//                   graph pooling. Cannot be known at compile time.
// - weights[L, q] : TRAINABLE — learned via backprop through PennyLane's
//                   TorchLayer. Shape = (2, 8) = 16 parameters total.
//
// Since IBM Quantum Composer does not support input parameters or symbolic
// angles, all RY angles are set to placeholder value 0.0. The circuit
// TOPOLOGY (gate order, CNOT connectivity, qubit assignment) is exact.
//
// To visualize with trained weights, replace the 0.0 angles with values from:
//   model.qlayer.weights  (shape [2, 8], 16 floats)
// and set input angles from the output of model.classical_projection(z).
// =============================================================================

OPENQASM 3.0;
include "stdgates.inc";

// --- Register declarations ---
qubit[8] q;
bit[8] c;

// =============================================================================
// STAGE 1: Angle Encoding — RY(input[i]) on each qubit
//          Source: quantum_circuit.py lines 19-20
//          Parameter type: DYNAMIC INPUT (from classical_projection)
// =============================================================================
ry(0.0) q[0];   // ry(inputs[0])
ry(0.0) q[1];   // ry(inputs[1])
ry(0.0) q[2];   // ry(inputs[2])
ry(0.0) q[3];   // ry(inputs[3])
ry(0.0) q[4];   // ry(inputs[4])
ry(0.0) q[5];   // ry(inputs[5])
ry(0.0) q[6];   // ry(inputs[6])
ry(0.0) q[7];   // ry(inputs[7])

// =============================================================================
// STAGE 2A: First explicit parameterized layer (LOOP A, part 1)
//           Source: quantum_circuit.py lines 27-28
//           Parameter type: TRAINABLE WEIGHT weights[0, i]
// =============================================================================
ry(0.0) q[0];   // ry(weights[0, 0])
ry(0.0) q[1];   // ry(weights[0, 1])
ry(0.0) q[2];   // ry(weights[0, 2])
ry(0.0) q[3];   // ry(weights[0, 3])
ry(0.0) q[4];   // ry(weights[0, 4])
ry(0.0) q[5];   // ry(weights[0, 5])
ry(0.0) q[6];   // ry(weights[0, 6])
ry(0.0) q[7];   // ry(weights[0, 7])

// =============================================================================
// STAGE 2B: While-loop iteration layer_idx=0 (LOOP A, part 2)
//           Source: quantum_circuit.py lines 40-55
//           layer_idx=0: skips rotation (layer_idx > 0 is false)
//                        applies CNOT chain (0 < n_entangle_layers=1)
// =============================================================================
// -- Entanglement: linear CNOT chain (layer_idx=0) --
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
cx q[4], q[5];
cx q[5], q[6];
cx q[6], q[7];

// =============================================================================
// STAGE 2C: While-loop iteration layer_idx=1 (LOOP A, part 3)
//           Source: quantum_circuit.py lines 43-45
//           layer_idx=1: applies rotation (layer_idx > 0 is true)
//                        skips entanglement (1 < 1 is false)
//           Parameter type: TRAINABLE WEIGHT weights[1, i]
// =============================================================================
ry(0.0) q[0];   // ry(weights[1, 0])
ry(0.0) q[1];   // ry(weights[1, 1])
ry(0.0) q[2];   // ry(weights[1, 2])
ry(0.0) q[3];   // ry(weights[1, 3])
ry(0.0) q[4];   // ry(weights[1, 4])
ry(0.0) q[5];   // ry(weights[1, 5])
ry(0.0) q[6];   // ry(weights[1, 6])
ry(0.0) q[7];   // ry(weights[1, 7])

// =============================================================================
// STAGE 3A: For-loop layer=0 (LOOP B — DUPLICATE of Loop A)
//           Source: quantum_circuit.py lines 62-71
//           layer=0: applies rotation weights[0, i], then CNOT chain
//           Parameter type: TRAINABLE WEIGHT weights[0, i] (REUSED)
//
//    *** NOTE: This is a DUPLICATE application of weights[0,:] and the
//    *** CNOT chain. These gates were already applied in Stages 2A and 2B.
//    *** This is a bug in the source code but is faithfully reproduced here.
// =============================================================================
ry(0.0) q[0];   // ry(weights[0, 0])  [DUPLICATE]
ry(0.0) q[1];   // ry(weights[0, 1])  [DUPLICATE]
ry(0.0) q[2];   // ry(weights[0, 2])  [DUPLICATE]
ry(0.0) q[3];   // ry(weights[0, 3])  [DUPLICATE]
ry(0.0) q[4];   // ry(weights[0, 4])  [DUPLICATE]
ry(0.0) q[5];   // ry(weights[0, 5])  [DUPLICATE]
ry(0.0) q[6];   // ry(weights[0, 6])  [DUPLICATE]
ry(0.0) q[7];   // ry(weights[0, 7])  [DUPLICATE]

// -- Entanglement: linear CNOT chain (layer=0) [DUPLICATE] --
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
cx q[4], q[5];
cx q[5], q[6];
cx q[6], q[7];

// =============================================================================
// STAGE 3B: For-loop layer=1 (LOOP B — DUPLICATE of Loop A)
//           Source: quantum_circuit.py lines 62-71
//           layer=1: applies rotation weights[1, i], no entanglement
//           Parameter type: TRAINABLE WEIGHT weights[1, i] (REUSED)
//
//    *** NOTE: This is a DUPLICATE application of weights[1,:].
//    *** Already applied in Stage 2C.
// =============================================================================
ry(0.0) q[0];   // ry(weights[1, 0])  [DUPLICATE]
ry(0.0) q[1];   // ry(weights[1, 1])  [DUPLICATE]
ry(0.0) q[2];   // ry(weights[1, 2])  [DUPLICATE]
ry(0.0) q[3];   // ry(weights[1, 3])  [DUPLICATE]
ry(0.0) q[4];   // ry(weights[1, 4])  [DUPLICATE]
ry(0.0) q[5];   // ry(weights[1, 5])  [DUPLICATE]
ry(0.0) q[6];   // ry(weights[1, 6])  [DUPLICATE]
ry(0.0) q[7];   // ry(weights[1, 7])  [DUPLICATE]

// =============================================================================
// STAGE 4: Measurement — Pauli-Z expectation on all qubits
//          Source: quantum_circuit.py line 74
//
// Note: IBM Quantum Composer uses computational-basis measurements (Z-basis),
// which is equivalent to PennyLane's qml.expval(qml.PauliZ(i)) in terms of
// the measurement axis. The Composer shows raw bit outcomes; expectation
// values would be computed classically from shot statistics.
// =============================================================================
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
c[4] = measure q[4];
c[5] = measure q[5];
c[6] = measure q[6];
c[7] = measure q[7];
