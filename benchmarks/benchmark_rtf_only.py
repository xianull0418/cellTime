
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import fire
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.rtf_only import RTFOnlySystem
from dataset import TemporalCellDataset, collate_fn_temporal

# Set English font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def compute_correlation_rowwise(x, y):
    """
    Compute Pearson correlation row-wise between two tensors x and y.
    x, y: (batch_size, n_features)
    """
    # Centering
    x_mean = x - x.mean(dim=1, keepdim=True)
    y_mean = y - y.mean(dim=1, keepdim=True)
    
    # Normalization
    x_norm = x_mean.norm(dim=1, keepdim=True)
    y_norm = y_mean.norm(dim=1, keepdim=True)
    
    # Avoid division by zero
    x_norm = torch.clamp(x_norm, min=1e-8)
    y_norm = torch.clamp(y_norm, min=1e-8)
    
    # Correlation
    correlation = (x_mean * y_mean).sum(dim=1, keepdim=True) / (x_norm * y_norm)
    return correlation.squeeze()

def load_model(ckpt_path, device):
    print(f"Loading model from {ckpt_path}...")
    model = RTFOnlySystem.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def sample_with_model(model, x_cur, t_cur, t_next, sample_steps, device):
    """
    根据模型模式进行采样（支持 direct 和 inversion 两种模式）

    Args:
        model: RTFOnlySystem 模型
        x_cur: 当前细胞表达 [B, n_genes]
        t_cur: 当前时间 [B]
        t_next: 目标时间 [B]
        sample_steps: 采样步数
        device: 设备

    Returns:
        x_pred: 预测的下一时刻表达 [B, n_genes]
    """
    mode = getattr(model, 'mode', 'direct')

    if mode == "direct":
        cond = None
        null_cond = None
        if model.cfg.model.use_cond:
            cond = torch.stack([t_cur, t_next], dim=-1)
            null_cond = torch.zeros_like(cond)

        trajectory = model.model.sample(
            x_cur,
            sample_steps=sample_steps,
            cond=cond,
            null_cond=null_cond,
            cfg_scale=getattr(model.cfg.training, 'cfg_scale', 2.0),
            normalize=model.cfg.model.normalize
        )
    else:  # inversion
        cond_start = torch.stack([t_cur], dim=-1) if model.cfg.model.use_cond else None
        cond_target = torch.stack([t_next], dim=-1) if model.cfg.model.use_cond else None
        null_cond = torch.zeros(x_cur.shape[0], 1, device=device) if model.cfg.model.use_cond else None

        trajectory = model.model.sample(
            x_cur,
            sample_steps=sample_steps,
            cond_start=cond_start,
            cond_target=cond_target,
            null_cond=null_cond,
            cfg_scale=getattr(model.cfg.training, 'cfg_scale', 2.0),
            normalize=model.cfg.model.normalize
        )

    x_pred = trajectory[-1].to(device)
    return x_pred

def get_dataset(model, data_path=None):
    # Use data path from config if not provided
    if data_path is None:
        data_path = model.cfg.data.data_path

    print(f"Loading dataset from {data_path}...")

    target_genes = None
    if hasattr(model.cfg.data, 'target_genes_path') and model.cfg.data.target_genes_path:
        path = Path(model.cfg.data.target_genes_path)
        if path.exists():
            with open(path, 'r') as f:
                target_genes = [line.strip() for line in f if line.strip()]

    # Handle n_genes=0 case
    max_genes = model.cfg.model.n_genes if model.cfg.model.n_genes > 0 else None

    dataset = TemporalCellDataset(
        data=data_path,
        max_genes=max_genes,
        target_genes=target_genes,
        valid_pairs_only=True,
        time_col=model.cfg.data.time_col,
        next_cell_col=model.cfg.data.next_cell_col,
        verbose=True,
    )
    return dataset

def run_benchmark(
    ckpt_path: str = "output/rtf_only_experiment/checkpoints/rtf-epoch=146-train_loss=0.0955.ckpt",
    data_path: str = None,
    output_dir: str = "benchmarks/results/rtf_only",
    batch_size: int = 100,
    sample_steps: int = 20,
    vis_cells_per_time: int = 50,
    device: str = "cuda",
    seed: int = 42
):
    """
    Run benchmark for RTF Only model.
    
    Args:
        ckpt_path: Path to model checkpoint
        data_path: Path to dataset (optional, defaults to config in checkpoint)
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        sample_steps: Number of steps for ODE solver
        vis_cells_per_time: Number of cells to visualize per timepoint
        device: 'cuda' or 'cpu'
        seed: Random seed
    """
    pl.seed_everything(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = load_model(ckpt_path, device)
    
    # 2. Load Dataset
    dataset = get_dataset(model, data_path)

    # Print model mode
    mode = getattr(model, 'mode', 'direct')
    print(f"\nModel mode: {mode}")

    # 3. Compute Correlations (Global)
    print("\nComputing correlations on dataset...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_temporal
    )

    all_corrs = []
    all_mse = []

    # To save time on huge datasets, we can limit the number of batches for metric calculation
    # But for benchmark, we usually want a good representation. Let's do 100 batches (~10k cells)
    MAX_BATCHES = 100

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=min(len(dataloader), MAX_BATCHES))):
            if i >= MAX_BATCHES:
                break

            x_cur = batch["x_cur"].to(device)
            x_next = batch["x_next"].to(device)
            t_cur = batch["t_cur"].to(device)
            t_next = batch["t_next"].to(device)

            # Sample prediction using helper function (supports both modes)
            x_pred = sample_with_model(model, x_cur, t_cur, t_next, sample_steps, device)
            
            # Compute metrics
            corr = compute_correlation_rowwise(x_next, x_pred)
            mse = F.mse_loss(x_pred, x_next, reduction='none').mean(dim=1)
            
            all_corrs.append(corr.cpu())
            all_mse.append(mse.cpu())
            
    all_corrs = torch.cat(all_corrs).numpy()
    all_mse = torch.cat(all_mse).numpy()
    
    print(f"Mean Correlation: {np.mean(all_corrs):.4f} +/- {np.std(all_corrs):.4f}")
    print(f"Mean MSE: {np.mean(all_mse):.4f} +/- {np.std(all_mse):.4f}")
    
    # Save metrics
    df_metrics = pd.DataFrame({"Correlation": all_corrs, "MSE": all_mse})
    df_metrics.to_csv(output_path / "metrics.csv", index=False)
    
    # Plot Correlation Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(all_corrs, bins=50, kde=True)
    plt.title("Distribution of Predicted vs True Correlation")
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Count")
    plt.savefig(output_path / "correlation_dist.png")
    plt.close()
    
    # 4. Visualization (Trajectory)
    print("\nGenerating trajectory visualization...")

    # Get uns information
    adata = dataset.adata
    timepoints = adata.uns.get('timepoints', [])
    time_unit = adata.uns.get('time_unit', 'Time')

    if len(timepoints) == 0:
        # Fallback if not in uns, try to infer from obs
        timepoints = sorted(adata.obs[model.cfg.data.time_col].unique())

    # Filter timepoints that are numbers (handle potential strings/NaNs)
    timepoints = [t for t in timepoints if isinstance(t, (int, float, np.number))]
    timepoints.sort()

    # Collect cells for visualization - organized by timepoint
    vis_data_by_time = {}  # {t_start: [list of vis_data dicts]}
    pca_training_data = []

    valid_indices = dataset.valid_indices
    t_col = model.cfg.data.time_col

    # Build a mapping: dataset_idx -> adata_idx -> time
    # dataset[i] uses valid_indices[i] to get adata data
    dataset_times = adata.obs[t_col].iloc[valid_indices].values

    # Iterate through timepoints (except last)
    for t in timepoints[:-1]:
        # Find dataset indices where time == t
        # dataset_idx is the index into dataset (0 to len(dataset)-1)
        matching_dataset_indices = np.where(dataset_times == t)[0]

        if len(matching_dataset_indices) == 0:
            continue

        # Sample cells (use vis_cells_per_time parameter)
        n_sample = min(vis_cells_per_time, len(matching_dataset_indices))
        selected_dataset_indices = np.random.choice(matching_dataset_indices, n_sample, replace=False)

        # Get data for these cells using dataset __getitem__
        batch_data = [dataset[int(idx)] for idx in selected_dataset_indices]

        if len(batch_data) == 0:
            continue

        batch = collate_fn_temporal(batch_data)

        x_cur = batch["x_cur"].to(device)
        x_next = batch["x_next"].to(device)
        t_cur = batch["t_cur"].to(device)
        t_next = batch["t_next"].to(device)

        # Predict using helper function (supports both modes)
        with torch.no_grad():
            x_pred = sample_with_model(model, x_cur, t_cur, t_next, sample_steps, device)

        # Move to CPU
        x_cur_np = x_cur.cpu().numpy()
        x_next_np = x_next.cpu().numpy()
        x_pred_np = x_pred.cpu().numpy()

        vis_data_by_time[t] = []

        for i in range(len(x_cur_np)):
            pca_training_data.extend([x_cur_np[i], x_next_np[i], x_pred_np[i]])

            # Compute per-cell correlation
            corr_pred = np.corrcoef(x_next_np[i], x_pred_np[i])[0, 1]

            vis_data_by_time[t].append({
                "x_start": x_cur_np[i],
                "x_true": x_next_np[i],
                "x_pred": x_pred_np[i],
                "corr": corr_pred
            })

    if not pca_training_data:
        print("No data found for visualization!")
        return

    # Compute PCA on all collected data
    print("Computing PCA...")
    pca_data_mat = np.array(pca_training_data)
    pca = PCA(n_components=2)
    pca.fit(pca_data_mat)

    # ========== Figure 1: 2D PCA Space Trajectory ==========
    print("Generating 2D PCA trajectory plot...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color palette for timepoints
    time_colors = plt.cm.viridis(np.linspace(0, 1, len(vis_data_by_time)))

    from matplotlib.lines import Line2D

    for t_idx, (t_start, cells) in enumerate(vis_data_by_time.items()):
        color = time_colors[t_idx]

        for cell in cells:
            # Project to PCA space
            pc_start = pca.transform(cell["x_start"].reshape(1, -1))[0]
            pc_true = pca.transform(cell["x_true"].reshape(1, -1))[0]
            pc_pred = pca.transform(cell["x_pred"].reshape(1, -1))[0]

            # Plot start cell
            ax.scatter(pc_start[0], pc_start[1], c=[color], s=40, alpha=0.7,
                      marker='o', edgecolors='black', linewidths=0.5, zorder=3)

            # Plot true next cell (smaller, lighter)
            ax.scatter(pc_true[0], pc_true[1], c=[color], s=25, alpha=0.4,
                      marker='s', zorder=2)

            # Plot predicted next cell
            ax.scatter(pc_pred[0], pc_pred[1], c='red', s=30, alpha=0.6,
                      marker='^', edgecolors='darkred', linewidths=0.5, zorder=4)

            # Arrow from start to predicted
            ax.annotate('', xy=(pc_pred[0], pc_pred[1]), xytext=(pc_start[0], pc_start[1]),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.4, lw=1))

            # Dashed line from start to true
            ax.plot([pc_start[0], pc_true[0]], [pc_start[1], pc_true[1]],
                   '--', color=color, alpha=0.3, lw=0.8)

    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=10, label='Start Cell (t)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=8, alpha=0.5, label='True Next (t+1)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=10, label='Predicted (t+1)'),
        Line2D([0], [0], color='red', lw=2, alpha=0.5, label='Prediction Arrow'),
        Line2D([0], [0], linestyle='--', color='gray', lw=1, alpha=0.5, label='True Direction')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Colorbar for timepoints
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=min(vis_data_by_time.keys()),
                                                   vmax=max(vis_data_by_time.keys())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label(f'{time_unit}', fontsize=12)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Cell Trajectory Prediction in PCA Space', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "trajectory_pca_2d.png", dpi=150)
    plt.close()

    # ========== Figure 2: Per-Timepoint Comparison ==========
    print("Generating per-timepoint comparison plots...")
    n_times = len(vis_data_by_time)
    n_cols = min(4, n_times)
    n_rows = (n_times + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_times == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for t_idx, (t_start, cells) in enumerate(vis_data_by_time.items()):
        row, col = t_idx // n_cols, t_idx % n_cols
        ax = axes[row, col]

        # Project all cells for this timepoint
        starts = np.array([pca.transform(c["x_start"].reshape(1, -1))[0] for c in cells])
        trues = np.array([pca.transform(c["x_true"].reshape(1, -1))[0] for c in cells])
        preds = np.array([pca.transform(c["x_pred"].reshape(1, -1))[0] for c in cells])

        # Plot
        ax.scatter(starts[:, 0], starts[:, 1], c='blue', s=30, alpha=0.7,
                  marker='o', label='Start', edgecolors='darkblue', linewidths=0.5)
        ax.scatter(trues[:, 0], trues[:, 1], c='green', s=25, alpha=0.5,
                  marker='s', label='True Next')
        ax.scatter(preds[:, 0], preds[:, 1], c='red', s=30, alpha=0.7,
                  marker='^', label='Predicted', edgecolors='darkred', linewidths=0.5)

        # Draw arrows
        for i in range(len(cells)):
            ax.annotate('', xy=(preds[i, 0], preds[i, 1]), xytext=(starts[i, 0], starts[i, 1]),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.3, lw=0.8))

        # Compute mean correlation for this timepoint
        mean_corr = np.mean([c["corr"] for c in cells])

        ax.set_title(f't={t_start} (n={len(cells)}, r={mean_corr:.3f})', fontsize=11)
        ax.set_xlabel('PC1', fontsize=9)
        ax.set_ylabel('PC2', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

        if t_idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    # Hide unused subplots
    for t_idx in range(n_times, n_rows * n_cols):
        row, col = t_idx // n_cols, t_idx % n_cols
        axes[row, col].axis('off')

    plt.suptitle('Per-Timepoint Trajectory Prediction', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "trajectory_per_timepoint.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========== Figure 3: Correlation Summary ==========
    print("Generating correlation summary plot...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 3a: Correlation by timepoint (boxplot)
    ax = axes[0]
    time_labels = []
    corr_data = []
    for t_start, cells in vis_data_by_time.items():
        time_labels.append(str(t_start))
        corr_data.append([c["corr"] for c in cells])

    bp = ax.boxplot(corr_data, labels=time_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], time_colors[:len(time_labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel(f'{time_unit}', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Prediction Accuracy by Timepoint', fontsize=12)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.2, 1.1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 3b: Overall correlation distribution
    ax = axes[1]
    all_corrs_vis = [c["corr"] for cells in vis_data_by_time.values() for c in cells]
    sns.histplot(all_corrs_vis, bins=30, kde=True, ax=ax, color='steelblue')
    ax.axvline(x=np.mean(all_corrs_vis), color='red', linestyle='--',
              label=f'Mean: {np.mean(all_corrs_vis):.3f}')
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Overall Prediction Correlation', fontsize=12)
    ax.legend()

    # 3c: Mean correlation trend over time
    ax = axes[2]
    times = list(vis_data_by_time.keys())
    mean_corrs = [np.mean([c["corr"] for c in cells]) for cells in vis_data_by_time.values()]
    std_corrs = [np.std([c["corr"] for c in cells]) for cells in vis_data_by_time.values()]

    ax.errorbar(times, mean_corrs, yerr=std_corrs, fmt='o-', capsize=5,
               color='steelblue', markeredgecolor='darkblue', markersize=8)
    ax.fill_between(times,
                    np.array(mean_corrs) - np.array(std_corrs),
                    np.array(mean_corrs) + np.array(std_corrs),
                    alpha=0.2, color='steelblue')
    ax.set_xlabel(f'{time_unit}', fontsize=12)
    ax.set_ylabel('Mean Correlation', fontsize=12)
    ax.set_title('Prediction Accuracy Over Time', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path / "trajectory_correlation_summary.png", dpi=150)
    plt.close()

    # ========== Figure 4: Vector Field Style Visualization ==========
    print("Generating vector field visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot all start cells as background
    all_starts = []
    all_preds = []
    all_trues = []
    all_times = []

    for t_start, cells in vis_data_by_time.items():
        for c in cells:
            all_starts.append(pca.transform(c["x_start"].reshape(1, -1))[0])
            all_preds.append(pca.transform(c["x_pred"].reshape(1, -1))[0])
            all_trues.append(pca.transform(c["x_true"].reshape(1, -1))[0])
            all_times.append(t_start)

    all_starts = np.array(all_starts)
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    all_times = np.array(all_times)

    # Compute displacement vectors
    pred_displacement = all_preds - all_starts
    true_displacement = all_trues - all_starts

    # Plot quiver (vector field)
    # Predicted direction (red)
    ax.quiver(all_starts[:, 0], all_starts[:, 1],
             pred_displacement[:, 0], pred_displacement[:, 1],
             color='red', alpha=0.5, scale=1, scale_units='xy', angles='xy',
             width=0.003, headwidth=4, label='Predicted Direction')

    # True direction (blue, smaller)
    ax.quiver(all_starts[:, 0], all_starts[:, 1],
             true_displacement[:, 0], true_displacement[:, 1],
             color='blue', alpha=0.3, scale=1, scale_units='xy', angles='xy',
             width=0.002, headwidth=3, label='True Direction')

    # Color start points by time
    scatter = ax.scatter(all_starts[:, 0], all_starts[:, 1], c=all_times,
                        cmap='viridis', s=50, alpha=0.8, edgecolors='black',
                        linewidths=0.5, zorder=5)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label(f'{time_unit}', fontsize=12)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Cell Trajectory Vector Field (Predicted vs True)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "trajectory_vector_field.png", dpi=150)
    plt.close()

    print(f"\nAll trajectory plots saved to {output_path}:")
    print(f"  - trajectory_pca_2d.png: 2D PCA space trajectories")
    print(f"  - trajectory_per_timepoint.png: Per-timepoint comparison")
    print(f"  - trajectory_correlation_summary.png: Correlation statistics")
    print(f"  - trajectory_vector_field.png: Vector field visualization")

if __name__ == "__main__":
    fire.Fire(run_benchmark)

