import json
import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_json(filepath):
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def plot_loss_curve(history, output_path, title):
    if not history: return
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss', color='tab:blue', linewidth=2)
    
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs, history['val_loss'], label='Validation Loss', color='tab:orange', linewidth=2)
        
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Created: {output_path}")

def plot_metric_bar(metrics, metric_name, output_path, ylabel, title, color='tab:blue'):
    if not metrics: return
    targets = ['HOMO', 'LUMO', 'GAP']
    vals = [metrics[t][metric_name] for t in targets]
    
    x = np.arange(len(targets))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(6, 5))
    rects = ax.bar(x, vals, width, color=color)
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
                    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Created: {output_path}")

def plot_metric_comparison(metrics1, label1, metrics2, label2, metric_name, output_path, ylabel, title):
    if not metrics1 or not metrics2: return
    targets = ['HOMO', 'LUMO', 'GAP']
    vals1 = [metrics1[t][metric_name] for t in targets]
    vals2 = [metrics2[t][metric_name] for t in targets]
    
    x = np.arange(len(targets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, vals1, width, label=label1, color='tab:blue')
    rects2 = ax.bar(x + width/2, vals2, width, label=label2, color='tab:orange')
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Created: {output_path}")

def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    results_dir = root_dir / "results"
    vis_dir = ensure_dir(results_dir / "visualizations")
    
    expA_dir = ensure_dir(vis_dir / "experiment_A_full_classical_gnn")
    expB_dir = ensure_dir(vis_dir / "experiment_B_causal_classical_gnn")
    expC_dir = ensure_dir(vis_dir / "experiment_C_causal_qcgnn")
    expD_dir = ensure_dir(vis_dir / "experiment_D_full_qcgnn")
    
    # Load all available histories
    hist_A = load_json(results_dir / "classical_gnn_training_history.json")
    hist_B = load_json(results_dir / "causal_classical_gnn_training_history.json")
    hist_C = load_json(results_dir / "quantum_training_history.json")
    hist_D = load_json(results_dir / "full_qcgnn_training_history.json")
    
    # Load all available metrics
    met_A = load_json(results_dir / "classical_gnn_test_metrics.json")
    met_B = load_json(results_dir / "causal_classical_gnn_test_metrics.json")
    met_C = load_json(results_dir / "quantum_test_metrics.json")
    met_D = load_json(results_dir / "full_qcgnn_test_metrics.json")
    
    # ---------------- EXPERIMENT A (Previously requested) ----------------
    if hist_A:
        plot_loss_curve(hist_A, ensure_dir(expA_dir / "training_curves") / "classical_gnn_loss_curve.png", "Experiment A: Full-Graph Classical GENConv Loss")
    
    # ---------------- EXPERIMENT B ----------------
    if hist_B:
        train_dir = ensure_dir(expB_dir / "training_curves")
        plot_loss_curve(hist_B, train_dir / "causal_classical_gnn_loss_curve.png", "Experiment B: Causal-Graph Classical GENConv Loss")
        
    if met_B:
        met_dir = ensure_dir(expB_dir / "metrics")
        plot_metric_bar(met_B, "MAE", met_dir / "causal_classical_gnn_mae.png", "MAE (eV) [Lower is better]", "Causal-Graph Classical GENConv: MAE")
        plot_metric_bar(met_B, "RMSE", met_dir / "causal_classical_gnn_rmse.png", "RMSE (eV) [Lower is better]", "Causal-Graph Classical GENConv: RMSE")
        plot_metric_bar(met_B, "R2", met_dir / "causal_classical_gnn_r2.png", "R2 [Higher is better]", "Causal-Graph Classical GENConv: R2")
        
    if met_B and met_C:
        comp_dir = ensure_dir(expB_dir / "comparison")
        title = "Causal-Graph Classical GENConv vs Causal-Graph QCGNN"
        plot_metric_comparison(met_B, "Causal-Graph Classical GENConv", met_C, "Causal-Graph QCGNN", "MAE", comp_dir / "causal_classical_gnn_vs_qcgnn_mae.png", "MAE (eV) [Lower is better]", title)
        plot_metric_comparison(met_B, "Causal-Graph Classical GENConv", met_C, "Causal-Graph QCGNN", "RMSE", comp_dir / "causal_classical_gnn_vs_qcgnn_rmse.png", "RMSE (eV) [Lower is better]", title)
        plot_metric_comparison(met_B, "Causal-Graph Classical GENConv", met_C, "Causal-Graph QCGNN", "R2", comp_dir / "causal_classical_gnn_vs_qcgnn_r2.png", "R2 [Higher is better]", title)
        
    # ---------------- EXPERIMENT C (Previously requested) ----------------
    if hist_C:
        plot_loss_curve(hist_C, ensure_dir(expC_dir / "training_curves") / "qcgnn_loss_curve.png", "Experiment C: Causal-Graph QCGNN Loss")
        
    # ---------------- EXPERIMENT D ----------------
    if hist_D:
        train_dir = ensure_dir(expD_dir / "training_curves")
        plot_loss_curve(hist_D, train_dir / "full_qcgnn_loss_curve.png", "Experiment D: Full-Graph QCGNN Loss")
        
    if met_D:
        met_dir = ensure_dir(expD_dir / "metrics")
        plot_metric_bar(met_D, "MAE", met_dir / "full_qcgnn_mae.png", "MAE (eV) [Lower is better]", "Full-Graph QCGNN: MAE")
        plot_metric_bar(met_D, "RMSE", met_dir / "full_qcgnn_rmse.png", "RMSE (eV) [Lower is better]", "Full-Graph QCGNN: RMSE")
        plot_metric_bar(met_D, "R2", met_dir / "full_qcgnn_r2.png", "R2 [Higher is better]", "Full-Graph QCGNN: R2")
        
    if met_D and met_A:
        comp_dir = ensure_dir(expD_dir / "comparison")
        title = "Full-Graph Classical GENConv vs Full-Graph QCGNN"
        plot_metric_comparison(met_A, "Full-Graph Classical GENConv", met_D, "Full-Graph QCGNN", "MAE", comp_dir / "full_qcgnn_vs_classical_full_graph_mae.png", "MAE (eV) [Lower is better]", title)
        plot_metric_comparison(met_A, "Full-Graph Classical GENConv", met_D, "Full-Graph QCGNN", "RMSE", comp_dir / "full_qcgnn_vs_classical_full_graph_rmse.png", "RMSE (eV) [Lower is better]", title)
        plot_metric_comparison(met_A, "Full-Graph Classical GENConv", met_D, "Full-Graph QCGNN", "R2", comp_dir / "full_qcgnn_vs_classical_full_graph_r2.png", "R2 [Higher is better]", title)
        
    # Note: When generating CSV, we will just include whatever is available
    csv_rows = []
    targets = ['HOMO', 'LUMO', 'GAP']
    
    if met_A:
        for t in targets:
            csv_rows.append({'Model': 'Classical GENConv', 'Graph': 'Full Graph', 'Experiment': 'A', 'Target': t, 'MAE': met_A[t]['MAE'], 'RMSE': met_A[t]['RMSE'], 'R2': met_A[t]['R2']})
    if met_B:
        for t in targets:
            csv_rows.append({'Model': 'Classical GENConv', 'Graph': 'Causal Graph', 'Experiment': 'B', 'Target': t, 'MAE': met_B[t]['MAE'], 'RMSE': met_B[t]['RMSE'], 'R2': met_B[t]['R2']})
    if met_C:
        for t in targets:
            csv_rows.append({'Model': 'QCGNN', 'Graph': 'Causal Graph', 'Experiment': 'C', 'Target': t, 'MAE': met_C[t]['MAE'], 'RMSE': met_C[t]['RMSE'], 'R2': met_C[t]['R2']})
    if met_D:
        for t in targets:
            csv_rows.append({'Model': 'QCGNN', 'Graph': 'Full Graph', 'Experiment': 'D', 'Target': t, 'MAE': met_D[t]['MAE'], 'RMSE': met_D[t]['RMSE'], 'R2': met_D[t]['R2']})

            
    if csv_rows:
        csv_path = vis_dir / "model_comparison.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Experiment', 'Model', 'Graph', 'Target', 'MAE', 'RMSE', 'R2'])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Created: {csv_path}")

    print("\nVisualization generation complete!")

if __name__ == "__main__":
    main()
