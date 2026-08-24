"""
QM9 QCGNN Visualization Script
================================
Place this file in: qcgnn_project/   (root folder)
Run with         : python qm9_visualize.py

Generates all plots into: qcgnn_project/results/visualizations/
    0. summary_dashboard.png
    1. training_loss_curves.png
    2. test_metrics_bar.png
    3. property_distributions.png
    4. correlation_plots.png
    5. coulomb_matrices.png
    6. feature_heatmaps.png

Requirements:
    pip install matplotlib seaborn
"""

import os, sys, json, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
warnings.filterwarnings('ignore')

# ==============================================================================
# PATHS  — all relative to this script (place in qcgnn_project/ root)
# ==============================================================================
ROOT      = Path(__file__).resolve().parent
L2_DIR    = ROOT / "data" / "qm9" / "layer2"
RESULTS   = ROOT / "results"
VIZ_DIR   = RESULTS / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PKL = L2_DIR / "qm9_train.pkl"
NORM_PKL  = L2_DIR / "norm_stats.pkl"
HISTORY   = RESULTS / "training_history.json"
METRICS   = RESULTS / "test_metrics.json"

# ==============================================================================
# DARK THEME
# ==============================================================================
BG      = '#05070f'
PANEL   = '#0c1022'
BORDER  = '#1a2040'
ACCENT  = '#4af0c4'
ACCENT2 = '#7b5cf0'
ACCENT3 = '#f06c4a'
MUTED   = '#5a6080'
HOMO_C  = '#4af0c4'
LUMO_C  = '#f06c4a'
GAP_C   = '#7b5cf0'

plt.rcParams.update({
    'figure.facecolor' : BG,
    'axes.facecolor'   : PANEL,
    'axes.edgecolor'   : BORDER,
    'axes.labelcolor'  : '#e8eaf6',
    'xtick.color'      : MUTED,
    'ytick.color'      : MUTED,
    'text.color'       : '#e8eaf6',
    'grid.color'       : BORDER,
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.5,
    'font.family'      : 'monospace',
    'axes.titleweight' : 'bold',
    'figure.dpi'       : 130,
})

print("=" * 60)
print("  QM9 QCGNN — VISUALIZATION SUITE")
print("=" * 60)

# ==============================================================================
# HELPERS
# ==============================================================================
def savefig(name):
    path = VIZ_DIR / name
    plt.savefig(path, bbox_inches='tight', facecolor=BG, dpi=130)
    plt.close()
    print(f"  [OK] {name}")

def spine_style(ax):
    for s in ax.spines.values():
        s.set_edgecolor(BORDER)

def get_norm(norm_stats, key):
    """Safely get mean/std lists from norm_stats."""
    m = norm_stats['mean']
    s = norm_stats['std']
    idx = {'homo':0,'lumo':1,'gap':2}
    if isinstance(m, (list, np.ndarray)):
        return float(m[idx[key]]), float(s[idx[key]])
    return float(m), float(s)

# ==============================================================================
# 1. TRAINING LOSS CURVES
# ==============================================================================
def plot_loss_curves():
    print("\n[1/6] Training loss curves...")
    if not HISTORY.exists():
        print("      [SKIP] training_history.json not found"); return

    with open(HISTORY) as f:
        h = json.load(f)

    train_loss = h['train_loss']
    val_loss   = h['val_loss']
    lr_vals    = h.get('lr', [])
    epochs     = list(range(1, len(train_loss)+1))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('QCGNN — Training History', fontsize=13, color=ACCENT, y=1.02)

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, color=ACCENT,  lw=2, label='Train Loss')
    ax.plot(epochs, val_loss,   color=ACCENT3, lw=2, ls='--', label='Val Loss')
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.08, color=ACCENT2)
    best = int(np.argmin(val_loss)) + 1
    ax.axvline(best, color='white', ls=':', lw=1.2, label=f'Best epoch: {best}')
    ax.set_title('Loss (MAE)', fontsize=10, color=ACCENT)
    ax.set_xlabel('Epoch', fontsize=8); ax.set_ylabel('Loss', fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    spine_style(ax); ax.tick_params(labelsize=7)
    # Annotate final values
    ax.annotate(f'Final train: {train_loss[-1]:.4f}',
                xy=(epochs[-1], train_loss[-1]),
                xytext=(-80, 10), textcoords='offset points',
                color=ACCENT, fontsize=7,
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=0.8))

    # LR schedule
    ax2 = axes[1]
    if lr_vals:
        ax2.plot(epochs, lr_vals, color=GAP_C, lw=2)
        ax2.set_title('Learning Rate Schedule (Cosine Annealing)',
                      fontsize=10, color=ACCENT)
        ax2.set_xlabel('Epoch', fontsize=8); ax2.set_ylabel('LR', fontsize=8)
        ax2.set_yscale('log'); ax2.grid(True, alpha=0.3)
        spine_style(ax2); ax2.tick_params(labelsize=7)
    else:
        ax2.text(0.5, 0.5, 'LR data not available',
                 ha='center', va='center', color=MUTED, fontsize=10,
                 transform=ax2.transAxes)
        ax2.axis('off')

    plt.tight_layout()
    savefig("1_training_loss_curves.png")

# ==============================================================================
# 2. TEST METRICS BAR CHART
# ==============================================================================
def plot_test_metrics():
    print("\n[2/6] Test metrics...")
    if not METRICS.exists():
        print("      [SKIP] test_metrics.json not found"); return

    with open(METRICS) as f:
        m = json.load(f)

    targets   = ['homo', 'lumo', 'gap']
    colors    = [HOMO_C, LUMO_C, GAP_C]
    mae_vals  = [m[f'{t}_mae']  for t in targets]
    rmse_vals = [m[f'{t}_rmse'] for t in targets]
    labels    = ['HOMO', 'LUMO', 'GAP']
    x         = np.arange(3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('QCGNN — Test Set Results', fontsize=13, color=ACCENT, y=1.02)

    for ax, vals, metric in zip(axes,
                                [mae_vals, rmse_vals],
                                ['MAE (eV)', 'RMSE (eV)']):
        bars = ax.bar(x, vals, color=colors, alpha=0.85,
                      edgecolor=BG, linewidth=1.5, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=10, color='white', fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(metric, fontsize=10, color=ACCENT)
        ax.set_ylabel(metric, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        spine_style(ax); ax.tick_params(labelsize=8)
        mean_val = np.mean(vals)
        ax.axhline(mean_val, color='white', ls='--', lw=1.2,
                   label=f'Mean: {mean_val:.4f}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    savefig("2_test_metrics_bar.png")

# ==============================================================================
# 3. PROPERTY DISTRIBUTIONS
# ==============================================================================
def plot_distributions(train_data, norm_stats):
    print("\n[3/6] Property distributions...")

    sample  = train_data[:5000]
    all_y   = np.array([d.y.numpy().flatten() for d in sample])
    targets = norm_stats.get('targets', ['homo','lumo','gap'])
    colors  = [HOMO_C, LUMO_C, GAP_C]
    labels  = ['HOMO (eV)', 'LUMO (eV)', 'HOMO-LUMO Gap (eV)']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('QM9 — Property Distributions (5k sample)',
                 fontsize=13, color=ACCENT, y=1.02)

    for i, (ax, col, lbl, tgt) in enumerate(zip(axes, colors, labels, targets)):
        mean_v, std_v = get_norm(norm_stats, tgt)
        vals = all_y[:, i] * std_v + mean_v

        ax.hist(vals, bins=60, color=col, alpha=0.75, edgecolor='none')
        ax.axvline(vals.mean(), color='white', ls='--', lw=1.5,
                   label=f'Mean: {vals.mean():.3f}')
        ax.axvline(vals.mean()+vals.std(), color=col, ls=':', lw=1,
                   alpha=0.7, label=f'sigma={vals.std():.3f}')
        ax.axvline(vals.mean()-vals.std(), color=col, ls=':', lw=1, alpha=0.7)
        ax.set_title(lbl, fontsize=10, color=ACCENT)
        ax.set_xlabel('Energy (eV)', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        spine_style(ax); ax.tick_params(labelsize=7)

    plt.tight_layout()
    savefig("3_property_distributions.png")

# ==============================================================================
# 4. CORRELATION PLOTS
# ==============================================================================
def plot_correlations(train_data, norm_stats):
    print("\n[4/6] Correlation plots...")

    sample  = train_data[:3000]
    all_y   = np.array([d.y.numpy().flatten() for d in sample])
    targets = norm_stats.get('targets', ['homo','lumo','gap'])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('QM9 — Pairwise Property Correlations',
                 fontsize=13, color=ACCENT, y=1.02)

    pairs  = [(0,1),(0,2),(1,2)]
    cmaps  = ['viridis','plasma','inferno']

    for ax, (i,j), cm in zip(axes, pairs, cmaps):
        xs, ys = all_y[:,i], all_y[:,j]
        xl, yl = targets[i].upper(), targets[j].upper()

        ax.hexbin(xs, ys, gridsize=40, cmap=cm, linewidths=0.2, mincnt=1)
        m, b = np.polyfit(xs, ys, 1)
        xr   = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xr, m*xr+b, color='white', lw=1.5, ls='--', alpha=0.9)

        corr = np.corrcoef(xs, ys)[0,1]
        ax.set_title(f'{xl} vs {yl}  |  r = {corr:.3f}',
                     fontsize=10, color=ACCENT)
        ax.set_xlabel(f'{xl} (normalized)', fontsize=8)
        ax.set_ylabel(f'{yl} (normalized)', fontsize=8)
        ax.grid(True, alpha=0.3); spine_style(ax); ax.tick_params(labelsize=7)

    plt.tight_layout()
    savefig("4_correlation_plots.png")

# ==============================================================================
# 5. COULOMB MATRIX HEATMAPS
# ==============================================================================
def build_coulomb(pos, znums):
    n = len(znums)
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            zi,zj = znums[i],znums[j]
            if i==j:
                M[i,j] = 0.5*zi**2.4
            else:
                d = np.linalg.norm(np.array(pos[i])-np.array(pos[j]))
                M[i,j] = zi*zj/d if d>1e-6 else 0
    return M

def plot_coulomb(train_data):
    print("\n[5/6] Coulomb matrix heatmaps...")

    n_total = min(500, len(train_data))
    sizes   = sorted([(i, train_data[i].x.shape[0]) for i in range(n_total)],
                     key=lambda x: x[1])
    step    = max(1, len(sizes)//6)
    indices = [sizes[min(i*step, len(sizes)-1)][0] for i in range(6)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('QM9 — Coulomb Matrix Representations',
                 fontsize=13, color=ACCENT, y=1.01)

    for ax, idx in zip(axes.flat, indices):
        d = train_data[idx]
        n = d.x.shape[0]

        pos = d.pos.numpy() if (hasattr(d,'pos') and d.pos is not None) \
              else np.random.RandomState(idx).randn(n,3)

        znums = []
        for k in range(n):
            v = float(d.x[k,0])
            if   v < 0.15: znums.append(1)
            elif v < 0.35: znums.append(6)
            elif v < 0.55: znums.append(7)
            elif v < 0.75: znums.append(8)
            else:          znums.append(9)

        M     = build_coulomb(pos, znums)
        M_vis = np.log1p(np.abs(M))

        im = ax.imshow(M_vis, cmap='viridis', aspect='auto')
        ax.set_title(f'Molecule #{idx}  |  N={n} atoms',
                     fontsize=8, color=ACCENT)
        ax.set_xlabel('Atom index', fontsize=7)
        ax.set_ylabel('Atom index', fontsize=7)
        ax.tick_params(labelsize=6)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)
        cb.set_label('log(1+|C|)', fontsize=6)

    plt.tight_layout()
    savefig("5_coulomb_matrices.png")

# ==============================================================================
# 6. FEATURE HEATMAPS
# ==============================================================================
def plot_feature_heatmaps(train_data):
    print("\n[6/6] Feature heatmaps...")

    sample     = train_data[:10]
    node_feats = np.vstack([d.x.numpy()[:8]        for d in sample])[:40]
    edge_feats = np.vstack([d.edge_attr.numpy()[:8] for d in sample])[:40]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('QM9 — Node & Edge Feature Matrices',
                 fontsize=13, color=ACCENT, y=1.02)

    for ax, feats, title, cm in zip(
        axes,
        [node_feats, edge_feats],
        ['Node Features (9D per atom)', 'Edge Features (4D per bond)'],
        ['viridis', 'plasma']
    ):
        im = ax.imshow(feats, aspect='auto', cmap=cm)
        ax.set_title(title, fontsize=10, color=ACCENT)
        ax.set_xlabel('Feature Dimension', fontsize=8)
        ax.set_ylabel('Atom / Bond Index', fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        spine_style(ax)

    plt.tight_layout()
    savefig("6_feature_heatmaps.png")

# ==============================================================================
# BONUS: SUMMARY DASHBOARD
# ==============================================================================
def plot_summary_dashboard(train_data, norm_stats):
    print("\n[*] Summary dashboard...")

    sample  = train_data[:2000]
    all_y   = np.array([d.y.numpy().flatten() for d in sample])
    targets = norm_stats.get('targets', ['homo','lumo','gap'])
    colors  = [HOMO_C, LUMO_C, GAP_C]

    fig = plt.figure(figsize=(20, 11), facecolor=BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Row 1: 3 distributions
    for i, (col, tgt) in enumerate(zip(colors, targets)):
        ax     = fig.add_subplot(gs[0, i])
        mean_v, std_v = get_norm(norm_stats, tgt)
        vals   = all_y[:, i] * std_v + mean_v
        ax.hist(vals, bins=50, color=col, alpha=0.75, edgecolor='none')
        ax.axvline(vals.mean(), color='white', ls='--', lw=1.2)
        ax.set_title(f'{tgt.upper()}  mean={vals.mean():.3f} eV',
                     fontsize=9, color=ACCENT)
        ax.set_xlabel('eV', fontsize=7); ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2); spine_style(ax)

    # Row 1 col 4: scatter
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(all_y[:,0], all_y[:,2], c=all_y[:,1],
                cmap='plasma', s=4, alpha=0.5)
    ax4.set_title('HOMO vs GAP\n(color=LUMO)', fontsize=9, color=ACCENT)
    ax4.set_xlabel('HOMO (norm)', fontsize=7)
    ax4.set_ylabel('GAP (norm)',  fontsize=7)
    ax4.tick_params(labelsize=6); ax4.grid(True, alpha=0.2); spine_style(ax4)

    # Row 2: atom count, bond count, loss, metrics
    n_atoms = [d.x.shape[0]             for d in sample]
    n_bonds = [d.edge_index.shape[1]//2  for d in sample]

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(n_atoms, bins=range(1, max(n_atoms)+2),
             color=ACCENT, alpha=0.8, edgecolor=BG)
    ax5.set_title(f'Atoms/mol  mean={np.mean(n_atoms):.1f}',
                  fontsize=9, color=ACCENT)
    ax5.tick_params(labelsize=6); ax5.grid(True, alpha=0.2); spine_style(ax5)

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(n_bonds, bins=40, color=ACCENT2, alpha=0.8, edgecolor=BG)
    ax6.set_title(f'Bonds/mol  mean={np.mean(n_bonds):.1f}',
                  fontsize=9, color=ACCENT)
    ax6.tick_params(labelsize=6); ax6.grid(True, alpha=0.2); spine_style(ax6)

    ax7 = fig.add_subplot(gs[1, 2])
    if HISTORY.exists():
        with open(HISTORY) as f:
            h = json.load(f)
        ep = list(range(1, len(h['train_loss'])+1))
        ax7.plot(ep, h['train_loss'], color=ACCENT,  lw=1.5, label='Train')
        ax7.plot(ep, h['val_loss'],   color=ACCENT3, lw=1.5, ls='--', label='Val')
        ax7.set_title('Loss Curves', fontsize=9, color=ACCENT)
        ax7.set_xlabel('Epoch', fontsize=7)
        ax7.legend(fontsize=7); ax7.grid(True, alpha=0.2)
        spine_style(ax7); ax7.tick_params(labelsize=6)
    else:
        ax7.text(0.5,0.5,'No history',ha='center',va='center',
                 color=MUTED,transform=ax7.transAxes); ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    if METRICS.exists():
        with open(METRICS) as f:
            m = json.load(f)
        mae_vals = [m['homo_mae'], m['lumo_mae'], m['gap_mae']]
        bars = ax8.bar(['HOMO','LUMO','GAP'], mae_vals,
                       color=colors, alpha=0.85, edgecolor=BG, width=0.5)
        for bar, val in zip(bars, mae_vals):
            ax8.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+max(mae_vals)*0.02,
                     f'{val:.3f}', ha='center', fontsize=9,
                     color='white', fontweight='bold')
        ax8.set_title('Test MAE (eV)', fontsize=9, color=ACCENT)
        ax8.grid(True, alpha=0.2, axis='y')
        spine_style(ax8); ax8.tick_params(labelsize=7)
    else:
        ax8.text(0.5,0.5,'No metrics',ha='center',va='center',
                 color=MUTED,transform=ax8.transAxes); ax8.axis('off')

    fig.suptitle('QCGNN — Complete Project Dashboard',
                 fontsize=15, color=ACCENT, y=1.01, fontweight='bold')
    savefig("0_summary_dashboard.png")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f"\nLooking for data at:\n  {L2_DIR}\n")

    if not TRAIN_PKL.exists():
        print(f"[ERROR] Not found: {TRAIN_PKL}")
        print("Make sure this script is in the qcgnn_project/ root folder.")
        sys.exit(1)

    print("Loading data...")
    with open(TRAIN_PKL, 'rb') as f:
        train_data = pickle.load(f)
    with open(NORM_PKL,  'rb') as f:
        norm_stats = pickle.load(f)
    print(f"  Loaded {len(train_data):,} training molecules\n")

    plot_loss_curves()
    plot_test_metrics()
    plot_distributions(train_data, norm_stats)
    plot_correlations(train_data, norm_stats)
    plot_coulomb(train_data)
    plot_feature_heatmaps(train_data)
    plot_summary_dashboard(train_data, norm_stats)

    print("\n" + "=" * 60)
    print("  DONE! Plots saved to:")
    print(f"  {VIZ_DIR}")
    print("=" * 60)
    for f in sorted(VIZ_DIR.glob("*.png")):
        print(f"  + {f.name}")
    print("=" * 60)

if __name__ == '__main__':
    main()