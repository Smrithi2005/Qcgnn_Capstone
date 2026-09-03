from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'QCGNN Project: Baseline vs Contributions', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(4)

    def section_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 8, title, 0, 1, 'L')

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        # Handle simple bullets manually
        lines = body.split('\n')
        for line in lines:
            if line.startswith('* '):
                self.cell(5)
                self.multi_cell(0, 6, chr(149) + ' ' + line[2:])
            elif line.startswith('  * '):
                self.cell(10)
                self.multi_cell(0, 6, chr(186) + ' ' + line[4:])
            else:
                self.multi_cell(0, 6, line)
        self.ln(3)

def create_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Part 1
    pdf.chapter_title('PART 1: The Pre-Existing Foundation (What was already there)')
    
    body_p1_intro = "When I first inspected your workspace, you already had a robust, highly structured, and scientifically validated pipeline. You had successfully completed the data preprocessing, the causal extraction, and two primary experiments."
    pdf.chapter_body(body_p1_intro)

    pdf.section_title('Layer 1 & Layer 2: Raw Data & Preprocessing')
    body_l1 = (
        "* What it does: Reads raw .xyz QM9 files and converts them into PyTorch Geometric (PyG) Data objects.\n"
        "* Key Files: config.py, qcgnn_preprocessing.py, qm9_preprocess_dataset.py\n"
        "* Existing State:\n"
        "  * Extracted full molecular graphs (up to 18 atoms).\n"
        "  * Generated 9-dimensional node features and 4-dimensional edge features.\n"
        "  * Extracted exactly 3 targets: HOMO, LUMO, and GAP.\n"
        "  * Applied rigorous Z-score normalization (target_mean, target_std) saved to data/qm9/layer2/norm_stats.pkl.\n"
        "  * Outputs saved as full-graph datasets in data/qm9/layer2/ (qm9_train.pkl, etc.)."
    )
    pdf.chapter_body(body_l1)
    
    pdf.section_title('Layer 3: Causal Graph Extraction')
    body_l3 = (
        "* What it does: Uses a gradient-based causal inference model to strip away 'unimportant' atoms, reducing the full graphs to smaller, causally relevant subgraphs.\n"
        "* Key Files: layer3_causal.py, code/layer3_qcgnn_model/qcgnn_causal_extractor.py\n"
        "* Existing State:\n"
        "  * Reduced molecular graphs from ~18 atoms down to a variable 3-12 atoms.\n"
        "  * Preserved the original normalized targets perfectly.\n"
        "  * Outputs saved as causal-graph datasets in data/qm9/layer3/."
    )
    pdf.chapter_body(body_l3)

    pdf.section_title('Layer 4 / Experiment C: The Hybrid QCGNN Model')
    body_l4 = (
        "* What it does: A hybrid quantum-classical GNN running on the Causal graphs (Experiment C).\n"
        "* Key Files: code/layer4_quantum/quantum_model.py, quantum_circuit.py, quantum_train.py\n"
        "* Existing State:\n"
        "  * A highly customized PennyLane + PyTorch architecture (HybridQuantumGNN).\n"
        "  * Used Mean() + Max() pooling to convert variable graphs into a fixed 18-dim vector.\n"
        "  * Projected into an 8-qubit quantum circuit using RY angle encoding, parameterized rotation layers, and CNOT entanglement.\n"
        "  * Classical readout MLP to predict the 3 targets.\n"
        "  * Evaluated and saved to models/best_qcgnn.pt."
    )
    pdf.chapter_body(body_l4)

    pdf.section_title('Experiment A: Classical Baseline')
    body_expA = (
        "* What it does: A purely classical PyG GENConv network running on the Full graphs.\n"
        "* Key Files: code/baselines/classical_gnn/\n"
        "* Existing State:\n"
        "  * Acted as the classical benchmark against which quantum models would be compared.\n"
        "  * Trained and saved to models/best_classical_gnn.pt."
    )
    pdf.chapter_body(body_expA)

    pdf.add_page()
    
    # Part 2
    pdf.chapter_title('PART 2: What We Built Upon It (Our Additions)')
    
    body_p2_intro = "Our primary goal was to complete a 2x2 Experimental Matrix (Classical vs. Quantum across Full vs. Causal graphs) without modifying or breaking any of your existing Layers 1, 2, 3, or 4. Here is exactly what we built into the workspace:"
    pdf.chapter_body(body_p2_intro)

    pdf.section_title('1. Built Experiment B (Causal Graph + Classical GENConv)')
    body_expB = (
        "* Purpose: To determine if the performance of QCGNN (Exp C) was due to the quantum circuit or just the causal graph reduction.\n"
        "* What we added (code/baselines/causal_classical_gnn/):\n"
        "  * causal_classical_gnn_model.py: Replicated the 3-layer GENConv architecture from Experiment A.\n"
        "  * causal_classical_gnn_sanity_test.py: A strict test confirming the classical model could process the variable-sized (3-12 atom) Layer 3 causal graphs.\n"
        "  * causal_classical_gnn_train.py & evaluate.py: Custom training loops pointing exclusively to Layer 3 .pkl files and saving to best_causal_classical_gnn.pt."
    )
    pdf.chapter_body(body_expB)

    pdf.section_title('2. Built Experiment D (Full Graph + Hybrid QCGNN)')
    body_expD = (
        "* Purpose: To determine how the QCGNN behaves on complete molecules without causal reduction, providing a direct comparison to Experiment A.\n"
        "* What we added (code/experiments/experiment_D_full_qcgnn/):\n"
        "  * full_qcgnn_model.py: A smart wrapper that directly imports your validated HybridQuantumGNN.\n"
        "  * full_qcgnn_sanity_test.py: Verified that the Mean() + Max() pooling flawlessly handles the larger 18-atom graphs from Layer 2.\n"
        "  * full_qcgnn_train.py & evaluate.py: Scripts to train the quantum model on Layer 2 data, independently saving to best_full_qcgnn.pt."
    )
    pdf.chapter_body(body_expD)

    pdf.section_title('3. Rewrote the Visualization Engine')
    body_vis = (
        "* Purpose: We needed a script capable of automatically parsing all 4 experiments and generating comparative graphics for a research paper.\n"
        "* What we added (Updated code/visualization/visualize_results.py):\n"
        "  * Added dynamic loading to safely check which of the 4 experiments (A, B, C, D) have completed.\n"
        "  * Automated the creation of a clean folder structure (results/visualizations/experiment_X/).\n"
        "  * Programmed comparative bar charts (MAE, RMSE, and R2).\n"
        "  * Generated a unified model_comparison.csv aggregating all experiments."
    )
    pdf.chapter_body(body_vis)

    pdf.section_title('4. Conducted the Multi-Task Expansion Audit (3 -> 6 Targets)')
    body_audit = (
        "* Purpose: You requested an investigation into expanding the project from 3 targets to 6 targets to improve the scientific scope.\n"
        "* What we added (Feasibility Report):\n"
        "  * Built zero new code for this yet (per your instructions), but executed background scripts analyzing your raw .xyz files.\n"
        "  * Verified that 15 targets physically exist in your raw data.\n"
        "  * Identified and proposed the 3 best complementary targets: Dipole Moment (mu), Isotropic Polarizability (alpha), and Heat Capacity (cv).\n"
        "  * Laid down the exact architectural roadmap for how to implement this across Layers 2, 3, and 4 (including fixing the evaluation scripts)."
    )
    pdf.chapter_body(body_audit)

    pdf.output('QCGNN_Documentation.pdf', 'F')

if __name__ == '__main__':
    create_pdf()
