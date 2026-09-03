
"""
LAYER 3 (FINAL) — Causal / Important-Atom Extraction (classical, no quantum)
===========================================================================
METHOD (matches the Part 3 report):
    PRIMARY    : Integrated Gradients (IG) node-level attribution.
    VALIDATION : Removal-based subgraph fidelity (sufficiency / fidelity+ and
                 comprehensiveness / fidelity-), with a random-selection
                 control and an IG completeness (faithfulness) check.

DESIGN DECISION — FEATURE-ONLY MODEL (why this file dropped geometry)
--------------------------------------------------------------------
Earlier iterations used a *distance-aware* GCN that read 3D coordinates
(pos) through an RBF edge-weight channel. It was more accurate (~0.24 eV
HOMO), but it created a METHOD/EVALUATION MISMATCH:

  * IG attributes through NODE FEATURES x (holding geometry fixed), so it
    ranks atoms by their feature contribution.
  * The distance-GCN's prediction was dominated by GEOMETRY (edge weights
    from pos), not features.
  * Removal-based fidelity deletes an atom AND its geometry.

So IG was scoring one channel while fidelity tested another — and IG did not
beat a random atom ranking (verified with clean completeness ~1e-2). That is
a real finding, not a bug: node-feature importance and geometric importance
diverge for a geometry-dominated GNN.

Resolution (Option A1): make attribution and model consistent by training a
FEATURE-ONLY GCN — no pos, prediction depends only on x and connectivity.
Now "which atoms matter" (IG on x) and "which atoms the model uses" are the
same object, so removal fidelity is a fair test of IG. This also keeps IG as
the single, axiomatically-justified primary method from the Part 3 report
(completeness + sensitivity), keeps everything deterministic/reproducible,
and needs no torch_geometric.explain. The cost is a small accuracy drop,
which is the right trade for an attribution study.

IG BASELINE
-----------
IG integrates gradients along a straight path from a baseline x' to x. We use
the DATASET MEAN ATOM as x' (on-manifold: feature magnitudes match real
atoms, so f(x') is sane and the path integral is tractable). A zero baseline
is off-manifold for a trained model and breaks completeness — see --baseline.

UPSTREAM DATA CONTRACT (verified against this project's Layer 2)
---------------------------------------------------------------
    data.x          : [num_nodes, 9]  chemical features (x[:,0] = atomic num)
    data.edge_index : [2, num_edges]
    data.edge_attr  : [num_edges, 4]
    data.y          : [1, 3]  ALREADY-NORMALISED [homo, lumo, gap]
    (data.pos exists but is intentionally NOT used by the model here)
norm_stats.pkl : {'mean','std','targets'} in raw Hartree, for real-unit MAE.

USAGE
-----
    # 1. train feature-only GNN, no IG (cheap, ~20 min):
    python layer3_causal.py --fresh --train-only --epochs 100
    # 2. fast validation gate on 100 test graphs (per target):
    python layer3_causal.py --eval-only 100 --target 0     # homo
    # 3. full pipeline for ALL three targets, writes subgraphs + CSVs:
    python layer3_causal.py --all-targets
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve()
for _cand in (_HERE.parent, _HERE.parent.parent):
    if (_cand / "config.py").exists():
        sys.path.insert(0, str(_cand))
        break
try:
    import config  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Could not import config.py; run from project root.") from e

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.utils import subgraph
    _HAS_PYG = True
except Exception:  # pragma: no cover
    _HAS_PYG = False

HARTREE_TO_EV = 27.211386245988

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("layer3")


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 1. FEATURE-ONLY GNN
# ----------------------------------------------------------------------------
# Prediction depends only on node features x and connectivity (edge_index).
# NO pos, NO distance edge-weights. This is deliberate: it makes IG (which
# attributes x) consistent with the model, so removal fidelity fairly tests
# the attribution. See the module docstring.
# ============================================================================
class FeatureGCN(nn.Module):
    def __init__(self, in_dim: int = config.NODE_FEATURE_DIM,
                 hidden: int = config.GCN_HIDDEN_DIM,
                 num_targets: int = len(config.QM9_TARGET_NAMES),
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, add_self_loops=True)
        self.conv2 = GCNConv(hidden, hidden, add_self_loops=True)
        self.conv3 = GCNConv(hidden, hidden, add_self_loops=True)
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, num_targets)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        h = F.silu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.silu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.silu(self.conv3(h, edge_index))
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        hg = global_mean_pool(h, batch)
        hg = F.silu(self.lin1(hg))
        return self.lin2(hg)


# ============================================================================
# 2. Data loading (normalisation-aware)
# ============================================================================
def _load_split(path: Path) -> List["Data"]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "data" in obj:
        obj = obj["data"]
    if not isinstance(obj, (list, tuple)):
        raise TypeError(f"{path} did not contain a list of Data objects.")
    logger.info("Loaded %d graphs from %s", len(obj), path.name)
    return list(obj)


def _load_norm_stats() -> Optional[Dict[str, list]]:
    for p in (config.LAYER2_DIR / "norm_stats.pkl",
              config.LAYER2_OUTPUT_PKL.parent / "norm_stats.pkl"):
        if p.exists():
            with open(p, "rb") as f:
                ns = pickle.load(f)
            logger.info("Loaded norm_stats: mean=%s std=%s (Hartree)",
                        [round(m, 5) for m in ns["mean"]],
                        [round(s, 5) for s in ns["std"]])
            return ns
    logger.warning("norm_stats.pkl not found — MAE reported in NORMALISED units.")
    return None


def _feature_mean(graphs: Sequence["Data"], device: str) -> Tensor:
    """Dataset mean atom-feature vector — the on-manifold IG baseline."""
    total, count = None, 0
    for g in graphs:
        s = g.x.float().sum(0)
        total = s if total is None else total + s
        count += g.x.size(0)
    return (total / max(count, 1)).to(device)


# ============================================================================
# 3. Train / load feature-only GNN
# ============================================================================
def _unnormalise(y_norm, norm, device):
    if norm is None:
        return y_norm
    mean = torch.tensor(norm["mean"], device=device, dtype=y_norm.dtype)
    std = torch.tensor(norm["std"], device=device, dtype=y_norm.dtype)
    return y_norm * std + mean


def train_gnn(train_graphs, val_graphs, norm, device=config.DEVICE,
              epochs=100, lr=config.LEARNING_RATE,
              batch_size=config.LAYER3_BATCH_SIZE, save_path=None):
    set_seed()
    model = FeatureGCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    names = config.QM9_TARGET_NAMES
    best_val, best_state, bad = float("inf"), None, 0
    patience = config.EARLY_STOPPING_PATIENCE

    for ep in range(1, epochs + 1):
        model.train()
        tr = 0.0
        for b in train_loader:
            b = b.to(device)
            opt.zero_grad()
            pred = model(b.x, b.edge_index, b.batch)
            tgt = b.y.view(pred.size(0), -1).float().to(device)
            loss = F.smooth_l1_loss(pred, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),
                                     config.GRADIENT_CLIP_MAX_NORM)
            opt.step()
            tr += loss.item() * pred.size(0)

        model.eval()
        abs_err = torch.zeros(len(names), device=device)
        n = 0
        with torch.no_grad():
            for b in val_loader:
                b = b.to(device)
                pred = model(b.x, b.edge_index, b.batch)
                pred_r = _unnormalise(pred, norm, device)
                tgt_r = _unnormalise(
                    b.y.view(pred.size(0), -1).float().to(device), norm, device)
                abs_err += (pred_r - tgt_r).abs().sum(0)
                n += pred.size(0)
        mae_ev = (abs_err / max(n, 1)) * HARTREE_TO_EV
        logger.info("epoch %3d | train %.4f | val MAE eV %s | mean eV %.4f",
                    ep, tr / len(train_graphs),
                    {k: round(v, 4) for k, v in zip(names, mae_ev.tolist())},
                    mae_ev.mean().item())
        score = mae_ev.mean().item()
        sched.step(score)
        if score < best_val - 1e-5:
            best_val = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.info("Early stop @ %d (best mean eV %.4f)", ep, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, save_path)
        logger.info("Saved GNN -> %s (best val mean MAE %.4f eV)",
                    save_path, best_val)
    return model


# ============================================================================
# 4. Integrated Gradients (PRIMARY) — mean-atom baseline, feature-only model
# ============================================================================
@dataclass
class IGConfig:
    steps: int = 100
    target_index: int = 0
    aggregate: str = "abs_sum"       # 'abs_sum' | 'sum' | 'l2'
    baseline: str = "mean"           # 'mean' | 'zero'


def _make_baseline(x, mode, feature_mean):
    if mode == "zero":
        return torch.zeros_like(x)
    if mode == "mean":
        if feature_mean is None:
            return x.mean(0, keepdim=True).expand_as(x).clone()
        return feature_mean.to(x.device).unsqueeze(0).expand_as(x).clone()
    raise ValueError(mode)


def integrated_gradients(model, data, ig_cfg, device=config.DEVICE,
                         feature_mean=None):
    model.eval()
    x = data.x.to(device).float()
    edge_index = data.edge_index.to(device)
    baseline = _make_baseline(x, ig_cfg.baseline, feature_mean)

    diff = x - baseline
    grad_accum = torch.zeros_like(x)
    for k in range(1, ig_cfg.steps + 1):
        alpha = k / ig_cfg.steps
        x_i = (baseline + alpha * diff).clone().detach().requires_grad_(True)
        out = model(x_i, edge_index)
        grad = torch.autograd.grad(out[0, ig_cfg.target_index], x_i)[0]
        grad_accum += grad
    attributions = diff * (grad_accum / ig_cfg.steps)

    if ig_cfg.aggregate == "abs_sum":
        node_scores = attributions.abs().sum(dim=1)
    elif ig_cfg.aggregate == "sum":
        node_scores = attributions.sum(dim=1)
    elif ig_cfg.aggregate == "l2":
        node_scores = attributions.norm(dim=1)
    else:
        raise ValueError(ig_cfg.aggregate)

    with torch.no_grad():
        f_x = model(x, edge_index)[0, ig_cfg.target_index].item()
        f_b = model(baseline, edge_index)[0, ig_cfg.target_index].item()
    completeness = abs(attributions.sum().item() - (f_x - f_b))
    return {"node_scores": node_scores.detach().cpu(),
            "completeness": completeness, "f_x": f_x, "f_baseline": f_b}


# ============================================================================
# 5. Subgraph selection + induction
# ============================================================================
def select_top_k(node_scores, num_nodes):
    k = int(round(config.CAUSAL_AUTO_FRACTION * num_nodes))
    k = max(config.CAUSAL_MIN_ATOMS, min(k, config.CAUSAL_MAX_ATOMS, num_nodes))
    return torch.topk(node_scores, k=k, largest=True).indices.sort().values


def induce_subgraph(data, keep_idx):
    dev = data.edge_index.device
    keep_idx = keep_idx.to(dev)
    ei, ea = subgraph(keep_idx, data.edge_index,
                      edge_attr=getattr(data, "edge_attr", None),
                      relabel_nodes=True, num_nodes=data.x.size(0))
    sub = Data(x=data.x[keep_idx].clone(), edge_index=ei)
    if ea is not None:
        sub.edge_attr = ea
    if getattr(data, "y", None) is not None:
        sub.y = data.y.clone()
    if getattr(data, "pos", None) is not None:
        sub.pos = data.pos[keep_idx].clone()   # carried for downstream, unused by model
    sub.kept_index = keep_idx.detach().cpu()
    return sub


# ============================================================================
# 6. VALIDATION — removal-based fidelity (feature-only model)
# ============================================================================
@torch.no_grad()
def _predict(model, data, target_index, device):
    x = data.x.to(device).float()
    ei = data.edge_index.to(device)
    if x.size(0) == 0:
        return 0.0
    return model(x, ei)[0, target_index].item()


@torch.no_grad()
def fidelity_by_removal(model, data, node_scores, target_index,
                        k_fracs, device=config.DEVICE):
    n = data.x.size(0)
    f_full = _predict(model, data, target_index, device)
    order = torch.argsort(node_scores, descending=True)
    all_idx = set(range(n))
    suff, comp = [], []
    for frac in k_fracs:
        k = max(1, min(n, int(round(frac * n))))
        top = order[:k].tolist()
        keep = torch.tensor(sorted(top), dtype=torch.long)
        f_keep = _predict(model, induce_subgraph(data, keep), target_index, device)
        rest = torch.tensor(sorted(all_idx - set(top)), dtype=torch.long)
        f_rest = f_full if rest.numel() == 0 else _predict(
            model, induce_subgraph(data, rest), target_index, device)
        suff.append(abs(f_full - f_keep))
        comp.append(abs(f_full - f_rest))
    return {"fidelity_plus": suff, "fidelity_minus": comp, "f_full": f_full}


def random_fidelity(model, data, target_index, k_fracs, device, seed):
    g = torch.Generator().manual_seed(seed)
    rand = torch.rand(data.x.size(0), generator=g)
    return fidelity_by_removal(model, data, rand, target_index, k_fracs, device)


# ============================================================================
# 7. Orchestration over a split (one target)
# ============================================================================
@dataclass
class Layer3Result:
    graphs: List["Data"] = field(default_factory=list)
    per_graph: List[dict] = field(default_factory=list)
    fidelity_curves: dict = field(default_factory=dict)


def process_split(model, graphs, ig_cfg, device=config.DEVICE,
                  feature_mean=None,
                  fidelity_fracs=(0.1, 0.2, 0.3, 0.4, 0.5)):
    res = Layer3Result()
    agg = {f"plus_{f}": [] for f in fidelity_fracs}
    agg.update({f"minus_{f}": [] for f in fidelity_fracs})
    agg.update({f"rand_{f}": [] for f in fidelity_fracs})
    comp_all = []
    for i, data in enumerate(graphs):
        ig = integrated_gradients(model, data, ig_cfg, device, feature_mean)
        scores = ig["node_scores"]
        keep = select_top_k(scores, data.x.size(0))
        sub = induce_subgraph(data, keep)
        fid = fidelity_by_removal(model, data, scores, ig_cfg.target_index,
                                  fidelity_fracs, device)
        rnd = random_fidelity(model, data, ig_cfg.target_index,
                              fidelity_fracs, device, seed=i)
        for j, f in enumerate(fidelity_fracs):
            agg[f"plus_{f}"].append(fid["fidelity_plus"][j])
            agg[f"minus_{f}"].append(fid["fidelity_minus"][j])
            agg[f"rand_{f}"].append(rnd["fidelity_plus"][j])
        comp_all.append(ig["completeness"])
        res.graphs.append(sub)
        res.per_graph.append({
            "index": i, "num_nodes": int(data.x.size(0)),
            "num_kept": int(keep.numel()), "kept_atoms": keep.tolist(),
            "completeness": ig["completeness"],
            "f_x": ig["f_x"], "f_baseline": ig["f_baseline"]})
        if (i + 1) % 500 == 0:
            logger.info("  processed %d / %d", i + 1, len(graphs))

    res.fidelity_curves = {
        "k_frac": list(fidelity_fracs),
        "fidelity_plus_mean": [float(np.mean(agg[f"plus_{f}"])) for f in fidelity_fracs],
        "fidelity_minus_mean": [float(np.mean(agg[f"minus_{f}"])) for f in fidelity_fracs],
        "random_fidelity_plus_mean": [float(np.mean(agg[f"rand_{f}"])) for f in fidelity_fracs],
        "completeness_mean": float(np.mean(comp_all)),
        "completeness_max": float(np.max(comp_all))}
    return res


# ============================================================================
# 8. Persistence
# ============================================================================
def _write_csv(path, rows):
    import csv
    if not rows:
        path.write_text(""); return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _out_paths(target_name: str):
    """Per-target output paths so 3 targets don't overwrite each other."""
    base = config.LAYER3_DIR
    return {
        "train": base / f"qm9_causal_{target_name}_train.pkl",
        "val":   base / f"qm9_causal_{target_name}_val.pkl",
        "test":  base / f"qm9_causal_{target_name}_test.pkl",
    }


def save_results(name, target_name, res, out_pkl):
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(res.graphs, f)
    _write_csv(out_pkl.with_suffix(".diagnostics.csv"), res.per_graph)
    fc = res.fidelity_curves
    rows = [{
        "target": target_name,
        "k_frac": fc["k_frac"][i],
        "fidelity_plus_mean": fc["fidelity_plus_mean"][i],
        "fidelity_minus_mean": fc["fidelity_minus_mean"][i],
        "random_fidelity_plus_mean": fc["random_fidelity_plus_mean"][i],
        "ig_beats_random": fc["fidelity_plus_mean"][i] < fc["random_fidelity_plus_mean"][i],
    } for i in range(len(fc["k_frac"]))]
    _write_csv(out_pkl.with_suffix(".fidelity.csv"), rows)
    wins = sum(r["ig_beats_random"] for r in rows)
    logger.info("[%s/%s] saved %d subgraphs | completeness mean=%.2e max=%.2e",
                target_name, name, len(res.graphs),
                fc["completeness_mean"], fc["completeness_max"])
    logger.info("[%s/%s] IG beats random at %d/%d levels",
                target_name, name, wins, len(rows))


# ============================================================================
# 9. Eval gate (prints a verdict table, writes nothing)
# ============================================================================
def eval_gate(model, test_g, ig_cfg, feature_mean, device, n):
    sample = test_g[:n]
    tname = config.QM9_TARGET_NAMES[ig_cfg.target_index]
    logger.info("=== EVAL-ONLY [%s]: IG+fidelity on %d test graphs (no files) ===",
                tname, len(sample))
    res = process_split(model, sample, ig_cfg, device, feature_mean)
    fc = res.fidelity_curves
    logger.info("completeness mean=%.2e max=%.2e",
                fc["completeness_mean"], fc["completeness_max"])
    if fc["completeness_mean"] > 0.1:
        logger.warning("COMPLETENESS HIGH (>0.1) — IG unfaithful; fidelity "
                       "below is unreliable. Try --ig-steps 200.")
    logger.info("%-8s %-12s %-14s %-8s", "k_frac", "IG_fid+", "random_fid+", "IG_wins")
    wins = 0
    for i, kf in enumerate(fc["k_frac"]):
        ig_p = fc["fidelity_plus_mean"][i]
        rd_p = fc["random_fidelity_plus_mean"][i]
        win = ig_p < rd_p
        wins += win
        logger.info("%-8.2f %-12.4f %-14.4f %-8s", kf, ig_p, rd_p,
                    "YES" if win else "no")
    logger.info("VERDICT [%s]: IG beats random at %d/%d levels "
                "(k=%.2f is the extraction fraction)",
                tname, wins, len(fc["k_frac"]), config.CAUSAL_AUTO_FRACTION)


# ============================================================================
# 10. Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Layer 3 FINAL — feature-only IG + removal fidelity")
    ap.add_argument("--target", type=int, default=0, help="0=homo 1=lumo 2=gap")
    ap.add_argument("--all-targets", action="store_true",
                    help="run the full pipeline for homo, lumo AND gap")
    ap.add_argument("--ig-steps", type=int, default=100)
    ap.add_argument("--baseline", choices=["mean", "zero"], default="mean")
    ap.add_argument("--aggregate", choices=["abs_sum", "l2", "sum"],
                    default="abs_sum",
                    help="per-atom aggregation of feature attributions. "
                         "'abs_sum' (default), 'l2' (sharper localization, "
                         "often better for diffuse targets like HOMO), 'sum' "
                         "(signed).")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoint, retrain")
    ap.add_argument("--train-only", action="store_true",
                    help="train + save the GNN and STOP (cheap tuning).")
    ap.add_argument("--eval-only", type=int, default=0, metavar="N",
                    help="IG+fidelity on N test graphs, print verdict, write nothing.")
    args = ap.parse_args()

    if not _HAS_PYG:
        raise RuntimeError("PyTorch Geometric required.")
    set_seed()
    device = config.DEVICE
    norm = _load_norm_stats()

    train_g = _load_split(config.LAYER2_OUTPUT_TRAIN)
    val_g = _load_split(config.LAYER2_OUTPUT_VAL)
    test_g = _load_split(config.LAYER2_OUTPUT_TEST)
    if args.limit:
        train_g = train_g[:max(args.limit, 2000)]
        val_g, test_g = val_g[:args.limit], test_g[:args.limit]

    # NOTE: new checkpoint name — feature-only model, distinct from the old
    # distance_gnn.pt so there is no accidental reuse.
    ckpt = config.LAYER3_DIR / "feature_gnn.pt"
    if ckpt.exists() and not args.fresh:
        logger.info("Loading GNN from %s (use --fresh to retrain)", ckpt)
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        model = FeatureGCN().to(device)
        model.load_state_dict(blob["state_dict"])
    else:
        logger.info("Training FEATURE-ONLY GNN%s ...",
                    " (fresh)" if args.fresh else "")
        model = train_gnn(train_g, val_g, norm, device=device,
                          epochs=args.epochs, save_path=ckpt)

    if args.train_only:
        logger.info("--train-only done -> %s", ckpt)
        return

    feature_mean = _feature_mean(train_g, device)
    logger.info("IG baseline='%s' steps=%d aggregate='%s' | mean_atom[0:3]=%s",
                args.baseline, args.ig_steps, args.aggregate,
                [round(v, 3) for v in feature_mean[:3].tolist()])

    if args.eval_only:
        ig_cfg = IGConfig(steps=args.ig_steps, target_index=args.target,
                          baseline=args.baseline, aggregate=args.aggregate)
        eval_gate(model, test_g, ig_cfg, feature_mean, device, args.eval_only)
        return

    # full pipeline
    targets = range(len(config.QM9_TARGET_NAMES)) if args.all_targets else [args.target]
    for ti in targets:
        tname = config.QM9_TARGET_NAMES[ti]
        ig_cfg = IGConfig(steps=args.ig_steps, target_index=ti,
                          baseline=args.baseline, aggregate=args.aggregate)
        paths = _out_paths(tname)
        for name, graphs in (("train", train_g), ("val", val_g), ("test", test_g)):
            logger.info("=== Layer 3 [%s] on %s (%d graphs) ===",
                        tname, name, len(graphs))
            res = process_split(model, graphs, ig_cfg, device, feature_mean)
            save_results(name, tname, res, paths[name])
    logger.info("Layer 3 complete -> %s", config.LAYER3_DIR)


if __name__ == "__main__":
    main()