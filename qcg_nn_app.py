"""QCGNN molecular prediction dashboard.

The web layer deliberately stays thin: graph construction and attribution are
provided by the project's inference modules, while this module owns request
validation, model execution, result persistence, and the UI.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from flask import Flask, jsonify, render_template_string, request
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.nn import global_max_pool, global_mean_pool

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from config import RESULTS_DIR  # noqa: E402
from inference.causal_attribution_inference import (  # noqa: E402
    prediction_gradient_attribution,
)
from inference.molecule_inference import smiles_to_qm9_graph  # noqa: E402
from layer4_quantum.quantum_model import HybridQuantumGNN  # noqa: E402

logging.basicConfig(level=os.getenv("QCGNN_LOG_LEVEL", "INFO"))
logger = logging.getLogger("qcgnn.dashboard")
app = Flask(__name__)

HARTREE_TO_EV = 27.211386245988
TARGETS = ("homo", "lumo", "gap")
MODEL_PATH = ROOT / "models" / "best_qcgnn_corrected.pt"
NORM_PATH = ROOT / "data" / "qm9" / "layer2" / "norm_stats.pkl"


def _load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    network = HybridQuantumGNN()
    network.load_state_dict(checkpoint)
    network.eval()
    with NORM_PATH.open("rb") as handle:
        norm = pickle.load(handle)
    if norm.get("targets") != list(TARGETS):
        raise ValueError(f"Unexpected target order in normalization statistics: {norm.get('targets')}")
    mean = torch.tensor(norm["mean"], dtype=torch.float32)
    std = torch.tensor(norm["std"], dtype=torch.float32)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Normalization statistics must contain three targets.")
    return network, mean, std


model, norm_mean, norm_std = _load_model()


def _prediction_from_causal(causal_data):
    batch = torch.zeros(causal_data.x.size(0), dtype=torch.long)
    with torch.no_grad():
        pooled = torch.cat(
            (global_mean_pool(causal_data.x, batch), global_max_pool(causal_data.x, batch)),
            dim=1,
        )
        angles = model.classical_projection(pooled)
        measurements = model.qlayer(angles)
        standardized = model.classical_readout(measurements)
        predictions = (standardized * norm_std + norm_mean) * HARTREE_TO_EV
    return angles[0].numpy(), measurements[0].numpy(), predictions[0].numpy()


def _safe_result_path(prefix):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return RESULTS_DIR / f"{prefix}_{stamp}_{uuid4().hex[:8]}.json"


def _chemistry(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return {
        "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "exact_mass": Descriptors.ExactMolWt(molecule),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(molecule),
    }


def _graph_payload(data, kept_nodes, scores):
    nodes = [dict(atom) for atom in data.display_heavy_atoms]
    display_ids = {node["id"] for node in nodes}
    for node, position in zip(nodes, data.display_heavy_pos):
        node["x"], node["y"] = float(position[0]), float(position[1])
        node["attribution"] = (
            float(scores[node["id"]]) if node["id"] < len(scores) else None
        )
    bonds = [
        {
            "source": int(bond["source"]),
            "target": int(bond["target"]),
            "bond_type": float(bond["bond_type"]),
            "is_aromatic": bool(bond["is_aromatic"]),
        }
        for bond in data.display_heavy_bonds
    ]
    kept = set(kept_nodes) & display_ids
    causal_edges = [
        bond for bond in bonds if bond["source"] in kept and bond["target"] in kept
    ]
    return {
        "nodes": nodes,
        "bonds": bonds,
        "causal_nodes": sorted(kept),
        "causal_bonds": causal_edges,
        "attribution_available": len(scores) == int(data.x.size(0)),
    }


def _run_simulator(smiles):
    data, atomic_numbers = smiles_to_qm9_graph(smiles)
    causal, kept_nodes, scores = prediction_gradient_attribution(model, data)
    angles, measurements, predictions = _prediction_from_causal(causal)
    graph = _graph_payload(data, kept_nodes, scores)
    display_selected = graph["causal_nodes"]
    result = {
        "status": "success",
        "execution": {"state": "Completed"},
        "execution_id": uuid4().hex,
        "mode": "LOCAL SIMULATOR",
        "molecule": {
            "input_smiles": smiles,
            "atom_count": len(atomic_numbers),
            "model_atom_count": int(data.x.size(0)),
            **_chemistry(smiles),
        },
        "structure": graph,
        "causal_analysis": {
            "method": "prediction-gradient saliency",
            "available": True,
            "kept_nodes": kept_nodes,
            "selected_atom_indices": display_selected,
            "model_selected_atom_indices": kept_nodes,
            "attribution_scores": scores,
            "display_attribution_scores": [
                node["attribution"] for node in graph["nodes"]
            ],
        },
        "quantum": {
            "num_qubits": 8,
            "input_angles": angles.tolist(),
            "z_expectations": measurements.tolist(),
        },
        "prediction": dict(zip(TARGETS, predictions.astype(float).tolist())),
    }
    path = _safe_result_path("quantum_simulator")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    result["result_file"] = str(path.relative_to(ROOT))
    return result


def _valid_instance(instance):
    return bool(
        re.fullmatch(r"crn:[^\s,]+", instance)
        and instance.count("crn:") == 1
    )


def _hardware_configuration():
    token = os.getenv("IBM_QUANTUM_API_KEY", "").strip()
    instance = os.getenv("IBM_QUANTUM_INSTANCE", "").strip()
    return bool(token and _valid_instance(instance)), token, instance


def _legacy_hardware_submit(smiles, backend_name, shots):
    return _hardware_submit(smiles, backend_name, shots)
    service, sampler_type = _require_hardware()
    backend = service.backend(backend_name)
    if bool(getattr(backend, "simulator", False)) or "sim" in str(backend.name).lower():
        raise RuntimeError(f"{backend.name} is a simulator; a real QPU is required.")
    status = backend.status()
    if not status.operational:
        raise RuntimeError(f"Backend {backend.name} is not operational.")
    if int(getattr(backend, "num_qubits", 0)) < N_QUBITS:
        raise RuntimeError(f"Backend {backend.name} has fewer than {N_QUBITS} qubits.")

    data, _ = smiles_to_qm9_graph(smiles)
    causal, kept_nodes, scores = prediction_gradient_attribution(model, data)
    angles, _, _ = _prediction_from_causal(causal)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    weights = checkpoint["qlayer.weights"].detach().cpu().numpy()
    prepared = prepare_hardware_circuit(angles, weights, backend=backend)
    sampler = sampler_type(mode=backend)
    job = submit_prepared_circuit(prepared, sampler, shots, "SUBMIT")
    job_id = str(job.job_id())
    with hardware_lock:
        hardware_jobs[job_id] = {
            "job": job,
            "prepared": prepared,
            "smiles": smiles,
            "kept_nodes": kept_nodes,
            "scores": scores,
            "backend": str(backend.name),
            "shots": shots,
        }
    return {
        "status": "submitted",
        "mode": "REAL IBM QPU",
        "backend": str(backend.name),
        "job_id": job_id,
        "job_status": str(job.status()),
        "shots": shots,
        "message": "Submitted to IBM Quantum. Poll the job status; no simulator fallback is used.",
    }


def _legacy_hardware_status(job_id):
    return _hardware_status(job_id)
    from layer4_quantum.ibm_hardware_adapter import (
        load_normalization_stats,
        predictions_from_measurements,
        remap_counts_to_logical,
    )
    from layer4_quantum.measurement_utils import counts_to_z_expectations

    with hardware_lock:
        record = hardware_jobs.get(job_id)
    if record is None:
        return {"status": "error", "error": "Unknown or expired hardware job."}, 404
    job = record["job"]
    job_status = str(job.status())
    response = {
        "status": "completed" if job.done() else "pending",
        "mode": "REAL IBM QPU",
        "backend": record["backend"],
        "job_id": job_id,
        "job_status": job_status,
        "shots": record["shots"],
    }
    if not job.done():
        return response, 200
    if not any(value in job_status.lower() for value in ("done", "completed", "success")):
        response["status"] = "failed"
        response["error"] = f"IBM job ended with status {job_status}."
        return response, 200
    runtime_result = job.result()
    register = getattr(runtime_result[0].data, "meas", None)
    if register is None or not hasattr(register, "get_counts"):
        raise RuntimeError("IBM result does not contain the meas register.")
    physical_counts = register.get_counts()
    prepared = record["prepared"]
    logical_counts = remap_counts_to_logical(
        physical_counts,
        prepared["logical_to_physical"],
        prepared["physical_to_classical"],
    )
    measurements = counts_to_z_expectations(logical_counts, 8)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    readout = {
        key: checkpoint[key].detach().cpu().numpy()
        for key in (
            "classical_readout.0.weight",
            "classical_readout.0.bias",
            "classical_readout.2.weight",
            "classical_readout.2.bias",
        )
    }
    mean, std = load_normalization_stats()
    standardized, predictions = predictions_from_measurements(measurements, readout, mean, std)
    result = {
        **response,
        "measurements": measurements.tolist(),
        "standardized_predictions": standardized.tolist(),
        "prediction": dict(zip(TARGETS, predictions.tolist())),
    }
    path = _safe_result_path("quantum_ibm")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    result["result_file"] = str(path.relative_to(ROOT))
    return result, 200


def _worker_python():
    configured = os.getenv("QCGNN_QPU_PYTHON")
    return Path(configured) if configured else ROOT / "qpu_execution_venv" / "Scripts" / "python.exe"


def _worker_environment():
    environment = os.environ.copy()
    configured, token, instance = _hardware_configuration()
    if configured:
        environment["IBM_QUANTUM_API_KEY"] = token
        environment["IBM_QUANTUM_INSTANCE"] = instance
    return environment


def _call_hardware_worker(payload, diagnostics=False):
    worker = ROOT / "code" / "layer4_quantum" / "ibm_qpu_worker.py"
    python = _worker_python()
    if not python.exists():
        raise RuntimeError(f"QPU worker Python was not found: {python}")
    completed = subprocess.run(
        [str(python), str(worker)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=_worker_environment(),
        timeout=120,
        check=False,
        cwd=ROOT,
    )
    worker_diagnostics = {
        "python": str(python),
        "return_code": completed.returncode,
    }
    if completed.returncode != 0:
        raise RuntimeError("IBM worker failed to start.")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("IBM worker returned an invalid response.") from error
    if not response.get("ok"):
        message = response.get("error", "IBM worker failed.")
        if message == "Unknown or expired hardware job.":
            raise KeyError(message)
        raise RuntimeError(message)
    result = response["result"]
    return (result, worker_diagnostics) if diagnostics else result


def _hardware_submit(smiles, backend_name, shots):
    data, _ = smiles_to_qm9_graph(smiles)
    causal, kept_nodes, scores = prediction_gradient_attribution(model, data)
    angles, _, _ = _prediction_from_causal(causal)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    weights = checkpoint["qlayer.weights"].detach().cpu().numpy()
    result = _call_hardware_worker({
        "operation": "submit",
        "angles": angles.tolist(),
        "weights": weights.tolist(),
        "backend": backend_name,
        "shots": shots,
        "confirmation": "SUBMIT",
    })
    graph = _graph_payload(data, kept_nodes, scores)
    result.update(
        molecule={"input_smiles": smiles, "model_atom_count": int(data.x.size(0)), **_chemistry(smiles)},
        structure=graph,
        causal_analysis={
            "available": True,
            "method": "prediction-gradient saliency",
            "kept_nodes": kept_nodes,
            "selected_atom_indices": graph["causal_nodes"],
            "model_selected_atom_indices": kept_nodes,
            "attribution_scores": scores,
            "display_attribution_scores": [
                node["attribution"] for node in graph["nodes"]
            ],
        },
        quantum={"num_qubits": 8, "input_angles": angles.tolist()},
    )
    return result


def _hardware_status(job_id):
    return _call_hardware_worker({
        "operation": "status",
        "job_id": job_id,
        "confirmation": "SUBMIT",
    }), 200


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QCGNN Molecular Dashboard</title>
<style>
:root{--ink:#172033;--muted:#667085;--line:#e4e7ec;--panel:#fff;--bg:#f5f7fb;--blue:#2457d6;--teal:#0d9488;--red:#b42318}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}
header{background:#101828;color:#fff;padding:24px max(22px,calc((100% - 1240px)/2));display:flex;justify-content:space-between;gap:18px;align-items:center}
.brand{display:flex;gap:13px;align-items:center}.mark{background:#3b82f6;width:42px;height:42px;border-radius:12px;display:grid;place-items:center;font-weight:800;font-size:21px}.brand h1{font-size:19px;margin:0}.brand p{margin:2px 0 0;color:#b9c4d4;font-size:13px}
.pill{border:1px solid #344054;border-radius:999px;padding:7px 12px;font-size:12px;color:#d0d5dd}
main{max-width:1240px;margin:30px auto;padding:0 22px}.hero{margin-bottom:22px}.eyebrow{color:var(--blue);text-transform:uppercase;font-weight:800;font-size:11px;letter-spacing:.13em}.hero h2{margin:4px 0;font-size:30px;letter-spacing:-.03em}.hero p{color:var(--muted);margin:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:0 8px 24px #1018280a;margin-bottom:18px}.grid{display:grid;grid-template-columns:1.05fr 1fr;gap:18px}.field{display:flex;gap:10px;align-items:end}.field>div{flex:1}label{display:block;color:var(--muted);font-size:12px;font-weight:700;margin-bottom:7px}input,select{width:100%;border:1px solid #d0d5dd;border-radius:9px;padding:12px 13px;font:inherit;color:var(--ink);background:#fff}input:focus,select:focus{outline:3px solid #dbe6ff;border-color:var(--blue)}button{border:0;border-radius:9px;padding:12px 17px;font:inherit;font-weight:750;cursor:pointer;background:var(--blue);color:#fff}button.secondary{background:#eef2ff;color:#2447a8}button.danger{background:#b42318}button:disabled{opacity:.55;cursor:wait}.hint{font-size:12px;color:var(--muted);margin-top:10px}.error{display:none;margin-top:13px;padding:11px 13px;border-radius:9px;background:#fef3f2;color:var(--red);border:1px solid #fecdca;font-size:13px}.mode{display:flex;gap:8px;margin:14px 0}.mode button{flex:1;background:#f2f4f7;color:#475467}.mode button.active{background:#e8efff;color:#1746b0;border:1px solid #9db8ff}.hardware{display:none}.section-title{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:14px}.section-title h3{margin:0;font-size:17px}.section-title span{font-size:12px;color:var(--muted)}#results{display:none}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.metric{border:1px solid #dbe4ff;background:#f7f9ff;padding:16px;border-radius:12px}.metric small{color:var(--muted);font-weight:750}.metric strong{display:block;color:var(--blue);font-size:24px;margin-top:4px}.structure{background:#fbfcfe;border:1px solid var(--line);border-radius:12px;padding:10px;min-height:290px}.structure svg{width:100%;height:270px}.bond{stroke:#98a2b3;stroke-width:3}.atom{stroke:#fff;stroke-width:2}.atom-label{font-size:12px;font-weight:800;text-anchor:middle;dominant-baseline:central;pointer-events:none}.causal{color:var(--teal)}.causal-panel{display:flex;flex-direction:column;gap:10px}.causal-panel .structure{min-height:190px}.causal-panel .structure svg{height:180px}.chart{min-height:92px;margin-top:14px}.chart svg{width:100%;height:110px}.chart-label{font-size:11px;fill:#667085}.chart-value{font-size:11px;fill:#172033;font-weight:700}.meta{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}.meta div{background:#f8fafc;border-radius:9px;padding:9px;font-size:12px}.meta b{display:block;font-size:14px}.chips{display:flex;flex-wrap:wrap;gap:7px}.chip{font:12px ui-monospace,monospace;background:#f2f4f7;padding:6px 8px;border-radius:7px}.status{padding:11px 13px;border-radius:9px;background:#ecfdf3;color:#067647;font-size:13px}.status.pending{background:#fffaeb;color:#b54708}.status.failed{background:#fef3f2;color:var(--red)}
@media(max-width:800px){header{align-items:flex-start;flex-direction:column}.grid{grid-template-columns:1fr}.field{flex-direction:column;align-items:stretch}.metrics{grid-template-columns:1fr}.meta{grid-template-columns:repeat(2,1fr)}}
</style></head>
<body><header><div class="brand"><div class="mark">Q</div><div><h1>QCGNN Molecular Dashboard</h1><p>Quantum-causal electronic structure prediction</p></div></div><div class="pill">HOMO · LUMO · GAP / eV</div></header>
<main><div class="hero"><div class="eyebrow">Inference workspace</div><h2>Predict from a molecular structure</h2><p>Every result is computed from the submitted SMILES through RDKit, causal attribution, and the trained QCGNN checkpoint.</p></div>
<section class="card"><div class="field"><div><label for="smiles">SMILES string</label><input id="smiles" value="CCO" placeholder="e.g. CCO or c1ccccc1" autocomplete="off"></div><button id="predictButton" onclick="predictMolecule()">Predict</button></div>
<div class="mode"><button id="localMode" class="active" onclick="setMode('local')">LOCAL SIMULATOR</button><button id="hardwareMode" onclick="setMode('hardware')">REAL IBM QPU</button></div>
<div id="hardwarePanel" class="hardware"><div class="field"><div><label for="backend">IBM backend</label><input id="backend" placeholder="e.g. ibm_fez"></div><div><label for="shots">Shots</label><input id="shots" type="number" min="1" value="1024"></div></div><p class="hint">Server authentication status: <b id="auth">checking…</b>. Credentials are never entered in this page.</p><button class="danger" id="submitHardwareButton" onclick="submitHardwareJob()">Prepare confirmation</button></div>
<div id="error" class="error"></div><p class="hint">Supported elements: H, C, N, O, F, P, S, Cl, Br, I. Simulator predictions do not contact IBM Quantum.</p></section>
<section id="results"><div class="grid"><div class="card"><div class="section-title"><h3>Predicted properties</h3><span id="modeLabel"></span></div><div class="metrics"><div class="metric"><small>HOMO</small><strong id="homo">—</strong></div><div class="metric"><small>LUMO</small><strong id="lumo">—</strong></div><div class="metric"><small>GAP</small><strong id="gap">—</strong></div></div><div id="propertyChart" class="chart"></div><div class="meta" id="meta"></div></div>
<div class="card"><div class="section-title"><h3>Execution status</h3><span id="job"></span></div><div id="status" class="status">Ready</div><div class="hint" id="resultFile"></div></div></div>
<div class="grid"><div class="card"><div class="section-title"><h3>Chemical structure</h3><span>RDKit 2D coordinates · heavy atoms</span></div><div class="structure"><svg id="structure" viewBox="0 0 600 270"></svg></div></div><div class="card causal-panel"><div class="section-title"><h3>Causal attribution subgraph</h3><span class="causal">gradient-selected atoms</span></div><div class="structure"><svg id="causalStructure" viewBox="0 0 600 220"></svg></div><div id="causalInfo" class="hint"></div><div id="attributionLegend" class="chips"></div><div id="angles" class="chips"></div></div></div><div class="card"><div class="section-title"><h3>Quantum measurements</h3><span>actual Z expectation values</span></div><div id="measurementChart" class="chart"></div></div></section></main>
<section id="comparison" class="card comparison" style="display:none"><div class="section-title"><h3>Simulator versus hardware</h3><span>absolute prediction differences / eV</span></div><div id="comparisonContent"></div></section></main>
<script>
let mode='local', polling=null, currentData=null, comparisonResults={};
function setMode(next){mode=next;localMode.classList.toggle('active',next==='local');hardwareMode.classList.toggle('active',next==='hardware');hardwarePanel.style.display=next==='hardware'?'block':'none';error.style.display='none';if(next==='hardware')fetch('/api/hardware/status').then(r=>r.json()).then(x=>auth.textContent=x.configured?'configured':'not configured').catch(()=>auth.textContent='unavailable')}
function showError(message){error.textContent=message;error.style.display='block'}
function drawGraph(g,target,selected){const svg=document.getElementById(target);svg.innerHTML='';const nodes=(g&&Array.isArray(g.nodes))?g.nodes:[],bonds=(g&&Array.isArray(g.bonds))?g.bonds:[];if(!nodes.length){svg.innerHTML='<text x="300" y="110" text-anchor="middle" fill="#667085">No graph data available</text>';return}const xs=nodes.map(n=>Number(n.x)),ys=nodes.map(n=>Number(n.y));if(xs.some(Number.isNaN)||ys.some(Number.isNaN)){svg.innerHTML='<text x="300" y="110" text-anchor="middle" fill="#b42318">Invalid structure coordinates</text>';return}const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),scale=Math.min(520/(maxx-minx||1),180/(maxy-miny||1)),pos=n=>({x:40+(n.x-minx)*scale+(520-(maxx-minx)*scale)/2,y:20+(n.y-miny)*scale+(180-(maxy-miny)*scale)/2}),p=new Map(nodes.map(n=>[n.id,pos(n)]));bonds.forEach(b=>{const a=p.get(b.source),z=p.get(b.target);if(!a||!z)return;const e=document.createElementNS('http://www.w3.org/2000/svg','line');e.setAttribute('x1',a.x);e.setAttribute('y1',a.y);e.setAttribute('x2',z.x);e.setAttribute('y2',z.y);e.setAttribute('class','bond');svg.appendChild(e)});nodes.forEach(n=>{const a=p.get(n.id),c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',a.x);c.setAttribute('cy',a.y);c.setAttribute('r',16);c.setAttribute('fill',selected.has(n.id)?'#99e6d9':'#b8ccff');c.setAttribute('class','atom');svg.appendChild(c);const t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',a.x);t.setAttribute('y',a.y);t.setAttribute('class','atom-label');t.textContent=n.label||n.symbol;svg.appendChild(t)})}
function bars(target,items,min,max,zero){const host=document.getElementById(target);if(!items||!items.length){host.textContent='No measurements available for this execution.';return}const width=560,height=100,base=zero?height/2:height-22,range=max-min||1,svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox',`0 0 ${width} ${height}`);if(zero){const line=document.createElementNS('http://www.w3.org/2000/svg','line');line.setAttribute('x1',20);line.setAttribute('x2',width-10);line.setAttribute('y1',base);line.setAttribute('y2',base);line.setAttribute('stroke','#98a2b3');svg.appendChild(line)}const gap=(width-35)/items.length;items.forEach((item,i)=>{const value=Number(item.value),h=Math.max(2,Math.abs(value-min)/range*height);const y=zero?(value>=0?base-h:base):height-h-18;const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');rect.setAttribute('x',25+i*gap);rect.setAttribute('y',y);rect.setAttribute('width',Math.max(10,gap-8));rect.setAttribute('height',h);rect.setAttribute('fill',zero?(value>=0?'#2457d6':'#0d9488'):'#2457d6');svg.appendChild(rect);const label=document.createElementNS('http://www.w3.org/2000/svg','text');label.setAttribute('x',25+i*gap+(gap-8)/2);label.setAttribute('y',height-3);label.setAttribute('text-anchor','middle');label.setAttribute('class','chart-label');label.textContent=item.label;svg.appendChild(label);const text=document.createElementNS('http://www.w3.org/2000/svg','text');text.setAttribute('x',25+i*gap+(gap-8)/2);text.setAttribute('y',Math.max(12,y-3));text.setAttribute('text-anchor','middle');text.setAttribute('class','chart-value');text.textContent=value.toFixed(2);svg.appendChild(text)});host.replaceChildren(svg)}
function updateComparison(data){const key=(data.molecule&&data.molecule.input_smiles||'').trim();if(!key||!data.prediction)return;comparisonResults[key]=comparisonResults[key]||{};comparisonResults[key][data.mode==='REAL IBM QPU'?'hardware':'local']=data.prediction;const pair=comparisonResults[key];if(!pair.local||!pair.hardware){document.getElementById('comparison').style.display='none';return}document.getElementById('comparison').style.display='block';document.getElementById('comparisonContent').innerHTML=['homo','lumo','gap'].map(k=>{const a=Number(pair.local[k]),b=Number(pair.hardware[k]);return `<div class="comparison-row"><b>${k.toUpperCase()}</b><span>Local ${a.toFixed(4)} eV</span><span>Hardware ${b.toFixed(4)} eV</span><strong>|Δ| ${(Math.abs(a-b)).toFixed(4)} eV</strong></div>`}).join('')}
function render(data){currentData=data;updateComparison(data);document.getElementById('results').style.display='block';document.getElementById('modeLabel').textContent=data.mode||'LOCAL SIMULATOR';['homo','lumo','gap'].forEach(k=>{const v=data.prediction&&data.prediction[k];document.getElementById(k).textContent=v==null?'—':Number(v).toFixed(4)+' eV'});const m=data.molecule||{},g=data.structure||{},ca=data.causal_analysis||{},nodes=Array.isArray(g.nodes)?g.nodes:[],selected=new Set(Array.isArray(g.causal_nodes)?g.causal_nodes:[]);document.getElementById('meta').innerHTML=`<div><b>${m.formula||'—'}</b>Formula</div><div><b>${m.heavy_atoms??'—'}</b>Heavy atoms</div><div><b>${m.rings??'—'}</b>Rings</div>`;drawGraph(g,'structure',selected);drawGraph({nodes:nodes.filter(n=>selected.has(n.id)),bonds:Array.isArray(g.causal_bonds)?g.causal_bonds:[]},'causalStructure',selected);document.getElementById('causalInfo').textContent=ca.available===false?`Attribution unavailable: ${ca.reason||'unknown reason'}`:ca.selected_atom_indices?`${ca.selected_atom_indices.length} of ${m.model_atom_count||nodes.length} model atoms selected by ${ca.method||'prediction-gradient saliency'}. Selected atoms are highlighted in teal.`:'Attribution data unavailable for this execution.';document.getElementById('attributionLegend').innerHTML=nodes.length?nodes.map(n=>`<span class="chip">${n.label||n.symbol} #${n.id}: ${n.attribution==null?'—':Number(n.attribution).toFixed(4)}</span>`).join(''):'No atom attribution values available.';document.getElementById('angles').innerHTML=(data.quantum&&data.quantum.input_angles||[]).map((v,i)=>`<span class="chip">q${i}: ${Number(v).toFixed(4)}</span>`).join('');bars('propertyChart',[{label:'HOMO',value:data.prediction?.homo},{label:'LUMO',value:data.prediction?.lumo},{label:'GAP',value:data.prediction?.gap}],-10,10,false);bars('measurementChart',(data.quantum&&data.quantum.z_expectations||data.quantum&&data.quantum.measurements||[]).map((value,i)=>({label:'Z'+i,value})), -1,1,true);const state=(data.execution&&data.execution.state)||(data.status==='success'?'Completed':data.status==='submitted'?'Queued':data.status==='error'?'Failed':'Ready');const statusEl=document.getElementById('status');statusEl.textContent=data.message||state;statusEl.className='status '+(state==='Failed'?'failed':state==='Queued'||state==='Running'||state==='Submitting'?'pending':'');document.getElementById('job').textContent=data.job_id||'';document.getElementById('resultFile').textContent=data.result_file?`Saved: ${data.result_file}`:''}
async function predictMolecule(){const errorEl=document.getElementById('error'),input=document.getElementById('smiles'),button=document.getElementById('predictButton'),statusEl=document.getElementById('status');errorEl.style.display='none';const value=input.value.trim();if(!value)return showError('Enter a SMILES string.');statusEl.textContent='Running';statusEl.className='status pending';button.disabled=true;button.textContent='Running…';try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles:value})});const d=await r.json();if(!r.ok)throw Error(d.error||'Prediction failed.');render(d)}catch(e){statusEl.textContent='Failed';statusEl.className='status failed';showError(e.message)}finally{button.disabled=false;button.textContent='Predict'}}
async function submitHardwareJob(){const input=document.getElementById('smiles'),backendInput=document.getElementById('backend'),shotsInput=document.getElementById('shots'),button=document.getElementById('submitHardwareButton');const value=input.value.trim(),backendName=backendInput.value.trim(),shotCount=Number(shotsInput.value);if(!value||!backendName||!Number.isInteger(shotCount)||shotCount<1)return showError('Provide SMILES, a backend name, and positive integer shots.');const ok=confirm(`Submit a REAL IBM QPU job now?\n\nBackend: ${backendName}\nShots: ${shotCount}\n\nThis action contacts IBM Quantum and may incur queue time or account cost.`);if(!ok)return;button.disabled=true;try{const r=await fetch('/api/hardware/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles:value,backend:backendName,shots:shotCount,confirmation:'SUBMIT'})});const d=await r.json();if(!r.ok)throw Error(d.error||'Hardware submission failed.');render(d);if(d.job_id)polling=setInterval(()=>pollJob(d.job_id),5000)}catch(e){showError(e.message)}finally{button.disabled=false}}
async function pollJob(id){const r=await fetch('/api/hardware/jobs/'+encodeURIComponent(id));const d=await r.json();const statusEl=document.getElementById('status');statusEl.textContent=d.error||d.execution?.state||`IBM job: ${d.job_status}`;statusEl.className='status '+(d.status==='pending'?'pending':d.status==='failed'?'failed':'');document.getElementById('job').textContent=d.job_id||id;if(d.status==='completed'){clearInterval(polling);render(Object.assign({},currentData,d,{molecule:currentData.molecule,structure:currentData.structure,causal_analysis:currentData.causal_analysis,quantum:Object.assign({},currentData.quantum,d)}))}if(d.status==='failed')clearInterval(polling)}
</script></body></html>"""


@app.get("/")
def index():
    return render_template_string(HTML)


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    smiles = payload.get("smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        return jsonify({"status": "error", "error": "SMILES input must be a non-empty string."}), 400
    try:
        return jsonify(_run_simulator(smiles.strip()))
    except ValueError as error:
        return jsonify({"status": "error", "error": str(error)}), 400
    except Exception:
        logger.exception("Simulator inference failed")
        return jsonify({"status": "error", "error": "Inference failed. Check the server log for details."}), 500


@app.get("/api/hardware/status")
def hardware_status():
    configured, _, _ = _hardware_configuration()
    worker_diagnostics = {
        "python": str(_worker_python()),
        "return_code": None,
    }
    try:
        worker_result, worker_diagnostics = _call_hardware_worker(
            {"operation": "environment"},
            diagnostics=True,
        )
        worker_environment = worker_result["environment"]
        worker_error = None
    except Exception:
        worker_environment = {
            "IBM_QUANTUM_API_KEY": False,
            "IBM_QUANTUM_INSTANCE": False,
        }
        worker_error = "worker_environment_unavailable"
    worker_configured = all(worker_environment.values())
    return jsonify({
        "configured": configured and worker_configured,
        "mode": "REAL IBM QPU",
        "environment": {
            "flask": {
                "IBM_QUANTUM_API_KEY": bool(os.getenv("IBM_QUANTUM_API_KEY", "").strip()),
                "IBM_QUANTUM_INSTANCE": _valid_instance(
                    os.getenv("IBM_QUANTUM_INSTANCE", "").strip()
                ),
            },
            "worker": worker_environment,
        },
        "worker_diagnostics": worker_diagnostics,
        **({"worker_error": worker_error} if worker_error else {}),
    })


@app.post("/api/hardware/submit")
def hardware_submit():
    payload = request.get_json(silent=True) or {}
    if payload.get("confirmation") != "SUBMIT":
        return jsonify({"status": "error", "error": "Explicit confirmation='SUBMIT' is required."}), 400
    smiles = payload.get("smiles")
    backend = payload.get("backend")
    shots = payload.get("shots")
    if not isinstance(smiles, str) or not isinstance(backend, str) or not isinstance(shots, int) or shots <= 0:
        return jsonify({"status": "error", "error": "SMILES, backend, and positive integer shots are required."}), 400
    try:
        return jsonify(_hardware_submit(smiles.strip(), backend.strip(), shots))
    except (ValueError, RuntimeError, PermissionError) as error:
        return jsonify({"status": "error", "error": str(error)}), 400
    except Exception:
        logger.exception("IBM submission failed")
        return jsonify({"status": "error", "error": "IBM submission failed; no simulator fallback was used."}), 502


@app.get("/api/hardware/jobs/<job_id>")
def hardware_job(job_id):
    try:
        result, code = _hardware_status(job_id)
        return jsonify(result), code
    except KeyError:
        return jsonify({"status": "error", "error": "Unknown or expired hardware job."}), 404
    except Exception:
        logger.exception("IBM job status failed")
        return jsonify({"status": "failed", "error": "Unable to read IBM job status."}), 502


if __name__ == "__main__":
    print("Starting QCGNN dashboard on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
