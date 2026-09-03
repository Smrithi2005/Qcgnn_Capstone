import os
import sys
from pathlib import Path
import json
import torch
import pickle
import numpy as np
from flask import Flask, jsonify, request, render_template_string
from torch_geometric.nn import global_mean_pool, global_max_pool

# ==============================================================================
# PROJECT SETUP & IMPORTS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

try:
    from layer4_quantum.quantum_model import HybridQuantumGNN
    from config import MODELS_DIR, LAYER2_DIR, LAYER3_DIR
    from config import ATOMIC_NUMBERS, VDW_RADII, BOND_THRESHOLD_FACTOR
    
    # New Inference Adapters
    from inference.molecule_inference import smiles_to_qm9_graph
    from inference.causal_attribution_inference import prediction_gradient_attribution
except ImportError as e:
    print(f"Error importing existing pipeline: {e}")
    sys.exit(1)

app = Flask(__name__)

# ==============================================================================
# GLOBAL MODEL AND DATA CACHE
# ==============================================================================
model = None
norm_stats = None
qm9_test_dict = {}
qm9_causal_dict = {}
qm9_supported_list = []

def init_app():
    global model, norm_stats, qm9_test_dict, qm9_causal_dict, qm9_supported_list
    print("Loading existing project assets...")
    
    model = HybridQuantumGNN(node_dim=9, n_qubits=8, n_rotation_layers=2, n_entangle_layers=1, num_targets=3)
    ckpt_path = MODELS_DIR / 'best_qcgnn.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
    
    norm_path = LAYER2_DIR / 'norm_stats.pkl'
    if norm_path.exists():
        with open(norm_path, 'rb') as f:
            norm_stats = pickle.load(f)
            
    with open(LAYER2_DIR / 'qm9_test.pkl', 'rb') as f:
        test_full = pickle.load(f)
        
    causal_path = LAYER3_DIR / 'qm9_causal_homo_test.pkl'
    if not causal_path.exists():
        causal_path = LAYER3_DIR / 'qm9_causal_test.pkl'
        
    with open(causal_path, 'rb') as f:
        test_causal = pickle.load(f)
        
    for i in range(len(test_full)):
        mol_id = int(test_full[i].mol_id)
        qm9_test_dict[mol_id] = test_full[i]
        qm9_causal_dict[mol_id] = test_causal[i]
        qm9_supported_list.append(mol_id)
        if len(qm9_supported_list) >= 100:
            break
            
    print("Initialization complete!")

# ==============================================================================
# HTML / CSS / JS FRONTEND
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QCGNN - Research Demonstration</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
body { font-family: -apple-system, system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; }
.header { padding: 1rem 2rem; background: #1e293b; border-bottom: 1px solid #334155; }
.header h1 { margin: 0; font-size: 1.5rem; color: #38bdf8; }
.header p { margin: 0.25rem 0 0; color: #94a3b8; font-size: 0.9rem; }
.nav { display: flex; gap: 1.5rem; margin-top: 1rem; color: #cbd5e1; font-size: 0.9rem; }
.container { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
.card { background: #1e293b; border: 1px solid #334155; border-radius: 0.5rem; padding: 1.5rem; margin-bottom: 1.5rem; }
.pipeline { display: flex; justify-content: space-between; position: relative; margin-bottom: 2rem; }
.stage { text-align: center; z-index: 1; flex: 1; opacity: 0.3; transition: opacity 0.5s; }
.stage.active { opacity: 1; }
.stage.error { opacity: 1; color: #fca5a5; }
.stage-icon { width: 40px; height: 40px; background: #334155; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.5rem; border: 2px solid #475569; }
.stage.active .stage-icon { background: #38bdf8; border-color: #0284c7; color: #fff; }
.stage.error .stage-icon { background: #fca5a5; border-color: #ef4444; color: #7f1d1d; }
.row { display: flex; gap: 1.5rem; flex-wrap: wrap; }
.col { flex: 1; min-width: 300px; }
input[type="text"], select, button { padding: 0.5rem 1rem; border-radius: 0.25rem; border: 1px solid #475569; background: #0f172a; color: #fff; }
button { background: #38bdf8; color: #0f172a; font-weight: bold; cursor: pointer; }
button:hover { background: #0ea5e9; }
.predictions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.pred-card { text-align: center; background: #0f172a; border: 1px solid #334155; border-radius: 0.5rem; padding: 1.5rem; }
.pred-value { font-size: 2rem; font-weight: bold; color: #38bdf8; margin-top: 0.5rem; }
.bar-chart { display: flex; flex-direction: column; gap: 0.5rem; }
.bar-row { display: flex; align-items: center; gap: 0.5rem; }
.bar-label { width: 30px; font-family: monospace; }
.bar-wrap { flex: 1; background: #334155; height: 16px; border-radius: 8px; position: relative; overflow: hidden; }
.bar-fill { height: 100%; background: #38bdf8; transition: width 0.5s; }
.bar-val { width: 50px; text-align: right; font-family: monospace; font-size: 0.8rem; }
.atom-node { stroke: #1e293b; stroke-width: 1.5px; }
.atom-link { stroke: #475569; stroke-opacity: 0.6; }
.highlighted { stroke: #fbbf24 !important; stroke-width: 3px !important; }
#error-box { background: #7f1d1d; color: #fca5a5; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; display: none; }
.warning-box { background: rgba(234, 179, 8, 0.1); color: #fbbf24; padding: 0.75rem; border-radius: 0.25rem; font-size: 0.85rem; border: 1px solid rgba(234, 179, 8, 0.3); margin-top: 1rem; }
.chat-msg { margin-bottom: 0.5rem; font-size: 0.9rem; }
.chat-msg.user { color: #38bdf8; }
.chat-msg.bot { color: #cbd5e1; }
</style>
</head>
<body>
<div class="header">
    <h1>QCGNN</h1>
    <p>Quantum-Causal Graph Neural Network</p>
    <div class="nav"><span>Home</span><span>Pipeline</span><span>Results</span><span>Research</span></div>
</div>
<div class="container">
    <div class="card">
        <h3>ANALYZE MOLECULE</h3>
        <p style="color: #94a3b8; font-size: 0.9rem;">Demonstration of the QCGNN pipeline. Select a trained QM9 molecule, or enter a custom SMILES string.</p>
        
        <div class="row" style="align-items: center; margin-top: 1rem;">
            <div style="flex:1;">
                <label style="display:block; font-size:0.8rem; color:#94a3b8; margin-bottom:0.25rem;">QM9 Dataset Molecule</label>
                <select id="mol-select" style="width:100%;"><option>Loading molecules...</option></select>
            </div>
            <div style="margin-top:1.25rem;">
                <button onclick="analyzeQM9()">Analyze QM9</button>
            </div>
        </div>
        
        <div class="row" style="align-items: center; margin-top: 1.5rem; border-top: 1px solid #334155; padding-top: 1.5rem;">
            <div style="flex:1;">
                <label style="display:block; font-size:0.8rem; color:#94a3b8; margin-bottom:0.25rem;">Custom Molecule (SMILES)</label>
                <input type="text" id="smiles-input" placeholder="e.g. CCO" style="width:100%;">
            </div>
            <div style="margin-top:1.25rem;">
                <button onclick="analyzeSMILES()" style="background: #10b981; color: #fff;">Analyze SMILES</button>
            </div>
        </div>
        
        <div class="warning-box">
            <strong>Scientific Disclaimer:</strong> Predictions are generated by a model trained strictly on QM9 molecular data. Results for molecules outside the training distribution may be unreliable. Atom highlighting for custom molecules represents <em>model attribution</em> (prediction gradients) and should not be interpreted as experimental or ground-truth causal evidence.
        </div>
        <div id="error-box"></div>
        <div id="status-text" style="color: #38bdf8; font-size: 0.9rem; margin-top: 1rem; font-weight: bold;"></div>
    </div>
    
    <div class="pipeline" id="pipeline-stages">
        <div class="stage" id="stage-1"><div class="stage-icon">01</div><div id="text-stage-1">MOLECULE</div></div>
        <div class="stage" id="stage-2"><div class="stage-icon">02</div><div id="text-stage-2">GRAPH</div></div>
        <div class="stage" id="stage-3"><div class="stage-icon">03</div><div id="text-stage-3">ATTRIBUTION</div></div>
        <div class="stage" id="stage-4"><div class="stage-icon">04</div><div id="text-stage-4">QUANTUM</div></div>
        <div class="stage" id="stage-5"><div class="stage-icon">05</div><div id="text-stage-5">PREDICTION</div></div>
    </div>

    <div class="row">
        <div class="col" style="flex: 1.5;">
            <div class="card" id="card-graph">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <h3>Molecular Graph (Stages 2 & 3)</h3>
                    <span id="graph-stats" style="color: #94a3b8; font-size: 0.8rem;"></span>
                </div>
                <p style="font-size: 0.85rem; color: #94a3b8;">Each atom is represented as a node and each molecular connection is represented as an edge. Numerical features are extracted from the molecular representation before being passed to the neural network.</p>
                <div id="graph-container" style="height: 350px; background: #0f172a; border-radius: 0.5rem; overflow: hidden; position: relative;"></div>
                <div id="causal-expl" style="display:none; margin-top: 1rem; font-size: 0.9rem; color: #fbbf24; background: rgba(251, 191, 36, 0.1); padding: 1rem; border-radius: 0.5rem;">
                    <strong>Inference-Time Atom Attribution:</strong> The highlighted atoms received the highest model attribution scores. This represents model-based topological importance (derived from prediction gradients or training-time causal weights), not proof of physical causality.
                </div>
            </div>
        </div>
        <div class="col">
            <div class="card" id="card-quantum">
                <h3>Quantum Circuit (Stage 4)</h3>
                <p style="font-size: 0.85rem; color: #94a3b8;">The classical graph representation is projected into the quantum input space. The values are encoded as rotation angles. A parameterized quantum circuit processes the encoded representation and measurements are converted back into classical values for prediction.<br><br><strong style="color:#38bdf8;">Quantum simulation: local PennyLane simulator</strong></p>
                <div style="background: #0f172a; padding: 1rem; border-radius: 0.5rem; overflow-x: auto;">
                    <svg viewBox="0 0 400 200" width="100%" height="200" id="circuit-svg"></svg>
                </div>
                <h4 style="margin-top: 1.5rem; font-size: 0.9rem; color: #94a3b8;">Measurements (Pauli-Z)</h4>
                <div class="bar-chart" id="quantum-bars"></div>
            </div>
        </div>
    </div>

    <div class="card" id="card-prediction">
        <h3>QCGNN Predictions (Stage 5)</h3>
        <div class="predictions">
            <div class="pred-card">
                <div style="color: #94a3b8; font-weight: bold;">HOMO</div>
                <div class="pred-value" id="val-homo">-- eV</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">Highest Occupied Molecular Orbital energy.</div>
            </div>
            <div class="pred-card">
                <div style="color: #94a3b8; font-weight: bold;">LUMO</div>
                <div class="pred-value" id="val-lumo">-- eV</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">Lowest Unoccupied Molecular Orbital energy.</div>
            </div>
            <div class="pred-card">
                <div style="color: #94a3b8; font-weight: bold;">HOMO-LUMO GAP</div>
                <div class="pred-value" id="val-gap">-- eV</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">Difference between LUMO and HOMO.</div>
            </div>
        </div>
        <div style="text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 1.5rem; font-style: italic;">
            These values are computational predictions generated by QCGNN from a QM9-trained model and should not be interpreted as experimental measurements.
        </div>
    </div>

    <div class="row">
        <div class="col" style="flex:1;">
            <div class="card" style="height: 100%;">
                <h3>Research Information</h3>
                <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; list-style: none; padding: 0;">
                    <li><strong>Dataset:</strong> QM9</li>
                    <li><strong>Model:</strong> Quantum-Causal Graph Neural Network</li>
                    <li><strong>Targets:</strong> HOMO, LUMO, GAP</li>
                    <li><strong>Quantum implementation:</strong> PennyLane local simulation (8 qubits)</li>
                    <li><strong>Graph framework:</strong> PyTorch Geometric</li>
                    <li><strong>Causal method:</strong> Prediction-Gradient Attribution</li>
                </ul>
            </div>
        </div>
        <div class="col" style="flex:1.5;">
            <div class="card" id="card-ai" style="height: 100%;">
                <h3>AI Research Assistant</h3>
                <div id="ai-chat-history" style="height: 150px; overflow-y:auto; background: #0f172a; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <div class="chat-msg bot"><strong>Assistant:</strong> Hello! I am the local rule-based QCGNN research assistant. I can explain the stages of the pipeline, HOMO, LUMO, GAP, or interpret the current results.</div>
                </div>
                <div class="row">
                    <input type="text" id="ai-input" placeholder="Ask a question..." style="flex:1;" onkeypress="if(event.key === 'Enter') askAI()">
                    <button onclick="askAI()">Send</button>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
const colorScale = { 1: '#cbd5e1', 6: '#475569', 7: '#3b82f6', 8: '#ef4444', 9: '#10b981' };
const atomName = { 1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F' };
let currentResult = null;

function init() {
    fetch('/api/qm9/molecules')
        .then(res => res.json())
        .then(data => {
            const select = document.getElementById('mol-select');
            select.innerHTML = '';
            data.molecules.forEach(id => {
                const opt = document.createElement('option');
                opt.value = id;
                opt.textContent = `QM9 Molecule ID: ${id}`;
                select.appendChild(opt);
            });
            drawCircuit();
            initEmptyBars();
        });
}

function resetUI() {
    document.getElementById('error-box').style.display = 'none';
    document.getElementById('status-text').innerText = '';
    document.getElementById('causal-expl').style.display = 'none';
    document.getElementById('val-homo').innerText = '-- eV';
    document.getElementById('val-lumo').innerText = '-- eV';
    document.getElementById('val-gap').innerText = '-- eV';
    document.getElementById('graph-stats').innerText = '';
    d3.select("#graph-container").selectAll("*").remove();
    initEmptyBars();
    
    document.querySelectorAll('.stage').forEach(s => {
        s.classList.remove('active', 'error');
        const text = s.querySelector('div:nth-child(2)');
        text.innerText = text.innerText.replace('...', '');
    });
    currentResult = null;
}

function setStageStatus(stage, status) {
    const el = document.getElementById('stage-'+stage);
    const textEl = document.getElementById('text-stage-'+stage);
    const baseText = textEl.innerText.replace('...', '');
    
    if (status === 'processing') {
        el.classList.add('active');
        el.classList.remove('error');
        textEl.innerText = baseText + '...';
        document.getElementById('status-text').innerText = `Processing: ${baseText}...`;
    } else if (status === 'completed') {
        el.classList.add('active');
        el.classList.remove('error');
        textEl.innerText = baseText;
    } else if (status === 'error') {
        el.classList.add('error');
        textEl.innerText = baseText + ' FAILED';
        document.getElementById('status-text').innerText = `Failed at: ${baseText}`;
    }
}

function analyzeQM9() {
    resetUI();
    const id = document.getElementById('mol-select').value;
    runPipeline('/api/analyze', {molecule_id: parseInt(id)});
}

function analyzeSMILES() {
    resetUI();
    const smiles = document.getElementById('smiles-input').value.trim();
    if(!smiles) return;
    runPipeline('/api/analyze/smiles', {smiles: smiles});
}

function runPipeline(endpoint, payload) {
    setStageStatus(1, 'processing');
    
    fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if(data.status === 'error') {
            setStageStatus(1, 'error');
            showError(data.error);
            return;
        }
        
        setStageStatus(1, 'completed');
        currentResult = data;
        
        setTimeout(() => { 
            setStageStatus(2, 'processing');
            drawGraph(data.graph, []); 
            document.getElementById('graph-stats').innerText = `${data.molecule.n_atoms} atoms | ${data.molecule.n_bonds} bonds | NodeDim: 9 | EdgeDim: 4`;
            
            if(data.molecule.smiles) {
                appendChat('bot', `3D conformer generated and inference-time attribution computed for SMILES: ${data.molecule.smiles}`);
            }
            
            setTimeout(() => {
                setStageStatus(2, 'completed');
                
                setStageStatus(3, 'processing');
                setTimeout(() => {
                    setStageStatus(3, 'completed');
                    drawGraph(data.graph, data.causal_analysis.kept_nodes);
                    document.getElementById('causal-expl').style.display = 'block';
                    
                    setStageStatus(4, 'processing');
                    setTimeout(() => {
                        setStageStatus(4, 'completed');
                        animateQuantum(data.quantum.measurements);
                        
                        setStageStatus(5, 'processing');
                        setTimeout(() => {
                            setStageStatus(5, 'completed');
                            document.getElementById('val-homo').innerText = data.prediction.homo.toFixed(4) + ' eV';
                            document.getElementById('val-lumo').innerText = data.prediction.lumo.toFixed(4) + ' eV';
                            document.getElementById('val-gap').innerText = data.prediction.gap.toFixed(4) + ' eV';
                            document.getElementById('status-text').innerText = 'Pipeline Complete';
                        }, 800);
                    }, 800);
                }, 800);
            }, 800);
        }, 800);
    })
    .catch(err => {
        showError('Network Error: ' + err.toString());
    });
}

function showError(msg) {
    const errBox = document.getElementById('error-box');
    errBox.innerText = msg;
    errBox.style.display = 'block';
}

function drawGraph(graph, kept_nodes) {
    const container = d3.select("#graph-container");
    container.selectAll("*").remove();
    const width = document.getElementById('graph-container').clientWidth;
    const height = 350;
    
    const svg = container.append("svg").attr("width", width).attr("height", height);
    
    const simulation = d3.forceSimulation(graph.nodes)
        .force("link", d3.forceLink(graph.edges).id(d => d.id).distance(40))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2));
        
    const link = svg.append("g")
        .selectAll("line")
        .data(graph.edges)
        .enter().append("line")
        .attr("class", d => {
            const isKept = kept_nodes.includes(d.source.id) && kept_nodes.includes(d.target.id);
            return "atom-link " + (kept_nodes.length > 0 && isKept ? "highlighted" : "");
        })
        .attr("stroke-width", 2);

    const node = svg.append("g")
        .selectAll("circle")
        .data(graph.nodes)
        .enter().append("circle")
        .attr("r", 12)
        .attr("fill", d => colorScale[d.atomic_num] || '#fff')
        .attr("class", d => {
            const isKept = kept_nodes.includes(d.id);
            return "atom-node " + (kept_nodes.length > 0 && isKept ? "highlighted" : "");
        });
        
    const labels = svg.append("g")
        .selectAll("text")
        .data(graph.nodes)
        .enter().append("text")
        .text(d => atomName[d.atomic_num] || '')
        .attr("font-size", "10px")
        .attr("text-anchor", "middle")
        .attr("dy", ".3em")
        .attr("fill", "#0f172a")
        .attr("font-weight", "bold");

    simulation.on("tick", () => {
        link.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        node.attr("cx", d => d.x).attr("cy", d => d.y);
        labels.attr("x", d => d.x).attr("y", d => d.y);
    });
}

function drawCircuit() {
    const svg = d3.select("#circuit-svg");
    svg.selectAll("*").remove();
    
    for(let i=0; i<8; i++) {
        let y = 20 + i*22;
        svg.append("line").attr("x1", 20).attr("y1", y).attr("x2", 380).attr("y2", y).attr("stroke", "#475569");
        svg.append("text").attr("x", 0).attr("y", y+4).text("q"+i).attr("fill", "#94a3b8").attr("font-size", "10px");
        
        svg.append("rect").attr("x", 40).attr("y", y-8).attr("width", 24).attr("height", 16).attr("fill", "#38bdf8").attr("rx", 2);
        svg.append("text").attr("x", 43).attr("y", y+3).text("Ry").attr("fill", "#0f172a").attr("font-size", "9px").attr("font-weight", "bold");
        
        svg.append("rect").attr("x", 85).attr("y", y-8).attr("width", 24).attr("height", 16).attr("fill", "#38bdf8").attr("rx", 2);
        svg.append("text").attr("x", 88).attr("y", y+3).text("Ry").attr("fill", "#0f172a").attr("font-size", "9px").attr("font-weight", "bold");
    }
    
    for(let i=0; i<7; i++) {
        let y1 = 20 + i*22;
        let y2 = 20 + (i+1)*22;
        let x = 140 + i*22;
        svg.append("line").attr("x1", x).attr("y1", y1).attr("x2", x).attr("y2", y2).attr("stroke", "#fbbf24");
        svg.append("circle").attr("cx", x).attr("cy", y1).attr("r", 3).attr("fill", "#fbbf24");
        svg.append("circle").attr("cx", x).attr("cy", y2).attr("r", 5).attr("fill", "none").attr("stroke", "#fbbf24").attr("stroke-width", 1.5);
        svg.append("line").attr("x1", x-5).attr("y1", y2).attr("x2", x+5).attr("y2", y2).attr("stroke", "#fbbf24").attr("stroke-width", 1.5);
        svg.append("line").attr("x1", x).attr("y1", y2-5).attr("x2", x).attr("y2", y2+5).attr("stroke", "#fbbf24").attr("stroke-width", 1.5);
    }
    
    for(let i=0; i<8; i++) {
        let y = 20 + i*22;
        svg.append("rect").attr("x", 320).attr("y", y-8).attr("width", 16).attr("height", 16).attr("fill", "#f43f5e").attr("rx", 2);
        svg.append("text").attr("x", 324).attr("y", y+3).text("M").attr("fill", "#fff").attr("font-size", "9px").attr("font-weight", "bold");
    }
}

function initEmptyBars() {
    const container = document.getElementById("quantum-bars");
    container.innerHTML = "";
    for(let i=0; i<8; i++) {
        container.innerHTML += `
            <div class="bar-row">
                <div class="bar-label">Q${i}</div>
                <div class="bar-wrap"><div class="bar-fill" id="bar-fill-${i}" style="width: 50%;"></div></div>
                <div class="bar-val" id="bar-val-${i}">0.00</div>
            </div>
        `;
    }
}

function animateQuantum(measurements) {
    for(let i=0; i<8; i++) {
        let val = measurements[i]; 
        let pct = ((val + 1) / 2) * 100;
        document.getElementById(`bar-fill-${i}`).style.width = pct + '%';
        document.getElementById(`bar-val-${i}`).innerText = val.toFixed(2);
    }
}

function appendChat(role, msg) {
    const history = document.getElementById("ai-chat-history");
    const div = document.createElement('div');
    div.className = "chat-msg " + role;
    div.innerHTML = `<strong>${role === 'user' ? 'You' : 'Assistant'}:</strong> ${msg}`;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

function askAI() {
    const input = document.getElementById('ai-input');
    const msg = input.value.trim();
    if(!msg) return;
    
    appendChat('user', msg);
    input.value = '';
    
    fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg, context: currentResult})
    })
    .then(res => res.json())
    .then(data => {
        appendChat('bot', data.response);
    })
    .catch(err => {
        appendChat('bot', 'Error communicating with assistant.');
    });
}

document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>
"""

# ==============================================================================
# API ROUTES
# ==============================================================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/qm9/molecules", methods=["GET"])
def get_molecules():
    return jsonify({"status": "success", "molecules": qm9_supported_list})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    msg = data.get("message", "").lower()
    ctx = data.get("context", None)
    
    if "homo" in msg and "gap" not in msg:
        resp = "HOMO stands for Highest Occupied Molecular Orbital. It represents the energy of the highest energy electrons in the molecule. In this model, it is predicted from the learned quantum-causal representation."
    elif "lumo" in msg and "gap" not in msg:
        resp = "LUMO stands for Lowest Unoccupied Molecular Orbital. It is the energy of the lowest empty orbital. The model predicts this based on atomic interactions."
    elif "gap" in msg:
        resp = "The HOMO-LUMO gap is the energy difference between the highest occupied and lowest unoccupied orbitals. It characterizes chemical reactivity. In this model, the predicted gap is obtained from the learned molecular representation."
    elif "attribution" in msg or "causal" in msg or "stage 3" in msg:
        resp = "The attribution stage utilizes model-based prediction-gradients. It identifies which atoms most heavily influence the trained network's prediction. The highlighted atoms received the highest model attribution scores."
    elif "quantum" in msg or "stage 4" in msg or "qubit" in msg:
        resp = "The classical graph representation is projected into an 8-dimensional quantum input space and encoded as RY rotation angles. A parameterized quantum circuit with CNOT entanglement processes it, and Pauli-Z measurements are passed to a classical network."
    elif "uncertain" in msg or "accurate" in msg or "real" in msg:
        resp = "These values are computational predictions generated by QCGNN and should not be interpreted as experimental measurements. The model is trained strictly on the QM9 dataset, meaning molecules outside this distribution may yield unreliable predictions."
    elif "result" in msg or "explain" in msg or "summary" in msg:
        if ctx and ctx.get("status") == "success":
            p = ctx["prediction"]
            resp = f"The pipeline processed the molecule, identifying {len(ctx['causal_analysis']['kept_nodes'])} structurally important atoms. The quantum circuit measured these features to predict a HOMO of {p['homo']:.3f} eV, a LUMO of {p['lumo']:.3f} eV, and a gap of {p['gap']:.3f} eV. These are computational predictions from a QM9-trained model."
        else:
            resp = "Please analyze a molecule first, and I can explain the specific results."
    else:
        resp = "I am a local rule-based research assistant. I can explain HOMO, LUMO, the GAP, the inference attribution stage, the quantum processing stage, or summarize your results."
        
    return jsonify({"response": resp})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        mol_id = int(data.get("molecule_id", -1))
        
        if mol_id not in qm9_test_dict or mol_id not in qm9_causal_dict:
            return jsonify({"status": "error", "error": f"Molecule {mol_id} not found."}), 400
            
        full_graph = qm9_test_dict[mol_id]
        causal_graph = qm9_causal_dict[mol_id]
        
        nodes = [{"id": i, "atomic_num": int(z)} for i, z in enumerate(full_graph.x[:, 0].numpy())]
        edges = []
        for s, d in zip(full_graph.edge_index[0].numpy(), full_graph.edge_index[1].numpy()):
            if s < d: edges.append({"source": int(s), "target": int(d)})
                
        kept_nodes = causal_graph.kept_index.numpy().tolist() if hasattr(causal_graph, 'kept_index') else []
        
        with torch.no_grad():
            x = causal_graph.x
            edge_index = causal_graph.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
            h_mean = global_mean_pool(x, batch)
            h_max = global_max_pool(x, batch)
            h_c = torch.cat([h_mean, h_max], dim=1)
            z = model.classical_projection(h_c)
            q_out = model.qlayer(z)
            pred = model.classical_readout(q_out)
            
            mean = torch.tensor(norm_stats['mean'])
            std = torch.tensor(norm_stats['std'])
            pred = (pred * std + mean) * 27.211386245988
            predictions = pred[0].numpy()

        return jsonify({
            "status": "success",
            "molecule": {"id": mol_id, "n_atoms": int(full_graph.x.size(0)), "n_bonds": int(full_graph.edge_index.size(1) / 2)},
            "graph": {"nodes": nodes, "edges": edges},
            "causal_analysis": {"kept_nodes": kept_nodes},
            "quantum": {"num_qubits": 8, "input": z[0].numpy().tolist(), "measurements": q_out[0].numpy().tolist()},
            "prediction": {"homo": float(predictions[0]), "lumo": float(predictions[1]), "gap": float(predictions[2])}
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/api/analyze/smiles", methods=["POST"])
def analyze_smiles():
    smiles = request.json.get("smiles", "").strip()
    if not smiles:
        return jsonify({"status": "error", "error": "Empty SMILES input"}), 400
        
    try:
        # Layer 1 & 2: Inference-time RDKit-based graph embedding
        full_data, atomic_nums = smiles_to_qm9_graph(smiles)
        
        nodes = [{"id": i, "atomic_num": int(z)} for i, z in enumerate(atomic_nums)]
        graph_edges = []
        for s, d in zip(full_data.edge_index[0].numpy(), full_data.edge_index[1].numpy()):
            if s < d:
                graph_edges.append({"source": int(s), "target": int(d)})
                
        # Layer 3: Inference-time Prediction-Gradient Attribution
        # Requires gradients to calculate topological saliency
        causal_data, kept_nodes, scores = prediction_gradient_attribution(model, full_data)
        
        # Layer 4: Quantum Model Inference (with disabled gradients now)
        with torch.no_grad():
            x = causal_data.x
            edge_index = causal_data.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
            h_mean = global_mean_pool(x, batch)
            h_max = global_max_pool(x, batch)
            h_c = torch.cat([h_mean, h_max], dim=1)
            z = model.classical_projection(h_c)
            q_out = model.qlayer(z)
            pred = model.classical_readout(q_out)
            
            # Prediction Denormalization
            mean = torch.tensor(norm_stats['mean'])
            std = torch.tensor(norm_stats['std'])
            pred = (pred * std + mean) * 27.211386245988 # Hartree to eV
            predictions = pred[0].numpy()
            
        return jsonify({
            "status": "success",
            "molecule": {"smiles": smiles, "n_atoms": len(atomic_nums), "n_bonds": len(graph_edges)},
            "graph": {"nodes": nodes, "edges": graph_edges},
            "causal_analysis": {"kept_nodes": kept_nodes, "attribution_scores": scores},
            "quantum": {"num_qubits": 8, "input": z[0].numpy().tolist(), "measurements": q_out[0].numpy().tolist()},
            "prediction": {"homo": float(predictions[0]), "lumo": float(predictions[1]), "gap": float(predictions[2])}
        })
    except Exception as e:
        return jsonify({"status": "error", "error": f"SMILES Embedding Adapter Error: {str(e)}"}), 500

if __name__ == "__main__":
    init_app()
    print("Starting QCGNN Frontend on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
