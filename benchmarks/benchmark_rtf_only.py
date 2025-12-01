
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
    
    dataset = TemporalCellDataset(
        data=data_path,
        max_genes=model.cfg.model.n_genes,
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
    vis_cells_per_time: int = 10,
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
            
            cond = None
            if model.cfg.model.use_cond:
                cond = torch.stack([t_cur, t_next], dim=-1)
            
            # Sample prediction
            trajectory = model.model.sample(
                x_cur,
                sample_steps=sample_steps,
                cond=cond,
                normalize=model.cfg.model.normalize
            )
            x_pred = trajectory[-1].to(device)
            
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
    
    # Collect cells for visualization
    vis_data = [] # List of (time, x_vector, type)
    
    # PCA Training Data (subset)
    pca_training_data = []
    
    # Iterate through timepoints (except last)
    for t in timepoints[:-1]:
        # Find cells at this timepoint that have a valid next pair
        # We iterate through dataset indices to find matching times
        # (Inefficient for large datasets, but okay for viz subset)
        
        # Better: Use adata to find indices, then map to dataset valid indices
        # But dataset might filter valid_pairs_only.
        
        # Let's just scan the dataloader again or pick random indices from dataset
        # Since we need specific timepoints, we'll scan a bit or use adata index logic
        
        # Find indices in dataset where t_cur == t
        # This is slow. Let's assume we can just grab from adata directly and assume the model works on raw expression
        # But model needs normalization/preprocessing used in dataset __getitem__
        
        # Fast way: Use the valid_indices in dataset
        valid_indices = dataset.valid_indices
        
        # Filter valid_indices for current timepoint
        # Accessing adata.obs directly is fast
        t_col = model.cfg.data.time_col
        current_t_indices = np.where(adata.obs[t_col].iloc[valid_indices] == t)[0]
        
        if len(current_t_indices) == 0:
            continue
            
        # Sample random cells
        if len(current_t_indices) > vis_cells_per_time:
            selected_sub_indices = np.random.choice(current_t_indices, vis_cells_per_time, replace=False)
        else:
            selected_sub_indices = current_t_indices
            
        dataset_indices = [valid_indices[i] for i in selected_sub_indices]
        
        # Collect data
        batch_data = [dataset[i] for i in dataset_indices]
        batch = collate_fn_temporal(batch_data)
        
        x_cur = batch["x_cur"].to(device)
        x_next = batch["x_next"].to(device) # True next
        t_cur = batch["t_cur"].to(device)
        t_next = batch["t_next"].to(device)
        
        # Predict
        cond = None
        if model.cfg.model.use_cond:
            cond = torch.stack([t_cur, t_next], dim=-1)
            
        with torch.no_grad():
            traj = model.model.sample(
                x_cur,
                sample_steps=sample_steps,
                cond=cond,
                normalize=model.cfg.model.normalize
            )
            x_pred = traj[-1]
        
        # Move to CPU
        x_cur_np = x_cur.cpu().numpy()
        x_next_np = x_next.cpu().numpy()
        x_pred_np = x_pred.cpu().numpy()
        t_cur_np = t_cur.cpu().numpy()
        t_next_np = t_next.cpu().numpy()
        
        # Convert time back to original scale if normalized in dataset
        # Dataset divides by 100 if max > 10. We should use the raw time from adata or reconstruct
        # But for plotting x-axis, we can use the t values we have or the known timepoints.
        # Let's use the scalar 't' from the loop for the start time, and t_next from data for end
        
        for i in range(len(x_cur_np)):
            # Store for PCA
            pca_training_data.append(x_cur_np[i])
            pca_training_data.append(x_next_np[i])
            pca_training_data.append(x_pred_np[i])
            
            # Store for Plotting
            # Structure: (time_start, time_end, x_start_vec, x_end_true_vec, x_end_pred_vec)
            vis_data.append({
                "t_start": t, # Use the loop variable (original scale)
                "t_end": t_next_np[i] * 100 if t > 10 else t_next_np[i], # Heuristic fix if dataset normalized
                # Better: rely on dataset.times being normalized or not. 
                # In dataset: if self.times.max() > 10: self.times = self.times / 100.0
                # We should revert this for plotting if we want original days
                
                "x_start": x_cur_np[i],
                "x_true": x_next_np[i],
                "x_pred": x_pred_np[i]
            })

    if not vis_data:
        print("No data found for visualization!")
        return

    # Fix time scaling for plotting
    # Check if we need to rescale dataset times (dataset divides by 100 if max > 10)
    # But here we recorded 't' from timepoints list which is original scale.
    # We need to ensure t_end matches.
    # Let's trust 't' and assume the next timepoint is the next in the list
    # Or just use the stored values and fix scale if < 1.
    
    # Compute PCA
    print("Computing PCA...")
    pca_data_mat = np.array(pca_training_data)
    pca = PCA(n_components=2)
    pca.fit(pca_data_mat)
    
    # Project and Plot
    plt.figure(figsize=(12, 8))
    
    # Plot background/context (optional: plot all real cells faint?)
    # For clarity, just plot the selected trajectories
    
    colors = sns.color_palette("husl", len(vis_data))
    
    # We want X-axis = Time
    # Y-axis = PC1
    
    for idx, item in enumerate(vis_data):
        t1 = item["t_start"]
        
        # Determine t2 (next time)
        # If dataset normalized time, t_end might be 0.115 instead of 11.5
        t2 = item["t_end"]
        if t1 > 10 and t2 < 10: 
            t2 = t2 * 100
            
        # Project
        pc_start = pca.transform(item["x_start"].reshape(1, -1))[0, 0] # PC1
        pc_true = pca.transform(item["x_true"].reshape(1, -1))[0, 0]
        pc_pred = pca.transform(item["x_pred"].reshape(1, -1))[0, 0]
        
        # Jitter time slightly to avoid overlap
        jitter = np.random.uniform(-0.05, 0.05)
        
        # Plot Start
        plt.scatter(t1 + jitter, pc_start, color='black', s=30, alpha=0.6, zorder=2, marker='o')
        
        # Plot True End
        plt.scatter(t2 + jitter, pc_true, color='gray', s=30, alpha=0.4, zorder=1, marker='x')
        
        # Plot Pred End
        plt.scatter(t2 + jitter, pc_pred, color='red', s=30, alpha=0.8, zorder=3, marker='^')
        
        # Draw Arrow: Start -> Pred
        plt.arrow(
            t1 + jitter, pc_start, 
            (t2 - t1), (pc_pred - pc_start),
            color='red', alpha=0.3, width=0.002, head_width=0.05, length_includes_head=True
        )
        
        # Draw Line: Start -> True (Optional, maybe too messy)
        # plt.plot([t1 + jitter, t2 + jitter], [pc_start, pc_true], 'k--', alpha=0.1)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='Start Cell (t)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', label='Predicted (t+1)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', label='True (t+1)'),
        Line2D([0], [0], color='red', lw=2, alpha=0.5, label='Prediction Path')
    ]
    plt.legend(handles=legend_elements)
    
    plt.xlabel(f"{time_unit}")
    plt.ylabel("PC1 (Gene Expression)")
    plt.title(f"Cell Trajectory Prediction (PC1 vs Time)")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(output_path / "trajectory_pc1_time.png")
    print(f"Plots saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(run_benchmark)

