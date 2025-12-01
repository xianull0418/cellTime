
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ae import AESystem
from dataset import ParquetIterableDataset, ZarrIterableDataset, collate_fn_static

# Set English font for plots
plt.rcParams['font.family'] = 'DejaVu Sans' 

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

def evaluate_dataset(model, dataset, device, max_batches=None, desc="Evaluating"):
    """
    Evaluate model on a dataset.
    Returns a DataFrame with metrics for each sample.
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=2048, 
        num_workers=8, 
        collate_fn=collate_fn_static,
        pin_memory=True
    )
    
    model.eval()
    model.to(device)
    
    mse_losses = []
    correlations = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc=desc, total=max_batches if max_batches else None):
            if max_batches and i >= max_batches:
                break
            
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            
            # Per-sample MSE (mean over genes)
            mse = F.mse_loss(reconstructed, batch, reduction='none').mean(dim=1)
            mse_losses.append(mse.cpu())
            
            # Per-sample Correlation
            corr = compute_correlation_rowwise(batch, reconstructed)
            correlations.append(corr.cpu())
            
    if not mse_losses:
        return pd.DataFrame()
        
    mse_all = torch.cat(mse_losses).numpy()
    corr_all = torch.cat(correlations).numpy()
    
    return pd.DataFrame({
        "MSE": mse_all,
        "Correlation": corr_all
    })

def main():
    # Paths
    ckpt_path = "/gpfs/flash/home/jcw/projects/research/cellTime/output/ae_large_scale/version2_8epochs/checkpoints/last.ckpt"
    processed_dir = Path("/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens/.parquet")
    output_dir = Path("benchmarks/results/ae-epoch=07-val_loss=0.0904.ckpt")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading model from {ckpt_path}...")
    try:
        # Fix: Infer n_genes from checkpoint weights if hyperparameters have n_genes=0
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # Check if n_genes is 0 in hyperparameters
        saved_n_genes = ckpt['hyper_parameters']['model']['n_genes']

        if saved_n_genes == 0:
            print(f"Warning: n_genes in checkpoint is 0, inferring from model weights...")
            # Infer n_genes from encoder's first layer weight shape
            encoder_weight_key = 'autoencoder.encoder.network.0.1.weight'
            if encoder_weight_key in ckpt['state_dict']:
                encoder_weight_shape = ckpt['state_dict'][encoder_weight_key].shape
                inferred_n_genes = encoder_weight_shape[1]  # [hidden_dim, n_genes]
                print(f"Inferred n_genes from weights: {inferred_n_genes}")

                # Update hyperparameters
                ckpt['hyper_parameters']['model']['n_genes'] = inferred_n_genes

                # Save fixed checkpoint temporarily
                fixed_ckpt_path = ckpt_path.replace('.ckpt', '_fixed.ckpt')
                torch.save(ckpt, fixed_ckpt_path)
                print(f"Saved fixed checkpoint to {fixed_ckpt_path}")

                # Load from fixed checkpoint
                model_system = AESystem.load_from_checkpoint(fixed_ckpt_path)
            else:
                raise ValueError(f"Cannot infer n_genes: key '{encoder_weight_key}' not found in checkpoint")
        else:
            model_system = AESystem.load_from_checkpoint(ckpt_path)

        print(f"Model loaded successfully. n_genes: {model_system.cfg.model.n_genes}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Datasets
    test_path = processed_dir / "test_shards"
    ood_path = processed_dir / "ood_shards"
    
    datasets = {}
    if test_path.exists():
        print(f"Loading Test Dataset from {test_path}...")
        datasets["Test"] = ParquetIterableDataset(test_path, verbose=True, shuffle_shards=False)
    else:
        print(f"Test path not found: {test_path}")
        
    if ood_path.exists():
        print(f"Loading OOD Dataset from {ood_path}...")
        datasets["OOD"] = ParquetIterableDataset(ood_path, verbose=True, shuffle_shards=False)
    else:
        print(f"OOD path not found: {ood_path}")

    # Evaluation
    results = {}
    for name, dataset in datasets.items():
        # Evaluate on a subset for speed if needed, or full. 
        # Since it's streaming, we can limit batches if we want a quick check, or remove limit for full.
        # For benchmark, we usually want full or substantial amount.
        # Let's use a reasonable limit for "benchmarking" if datasets are huge, e.g., 100 batches * 2048 = 200k cells.
        # Or just run until completion if not infinite. ParquetIterableDataset is finite if cycle=False (default).
        print(f"Evaluating {name} set...")
        df = evaluate_dataset(model_system, dataset, device, max_batches=200, desc=f"Eval {name}")
        df["Dataset"] = name
        results[name] = df
        
        print(f"{name} Results - MSE: {df['MSE'].mean():.4f} +/- {df['MSE'].std():.4f}, "
              f"Corr: {df['Correlation'].mean():.4f} +/- {df['Correlation'].std():.4f}")

    # Combine results
    full_df = pd.concat(results.values(), ignore_index=True)
    
    # Plotting
    print("Generating plots...")
    
    # 1. Histogram of Correlations
    plt.figure(figsize=(10, 6))
    sns.histplot(data=full_df, x="Correlation", hue="Dataset", element="step", stat="density", common_norm=False, bins=50)
    plt.title("Distribution of Reconstruction Correlation")
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Density")
    plt.savefig(output_dir / "reconstruction_correlation_dist.png")
    plt.close()
    
    # 2. Histogram of MSE
    plt.figure(figsize=(10, 6))
    sns.histplot(data=full_df, x="MSE", hue="Dataset", element="step", stat="density", common_norm=False, bins=50)
    plt.title("Distribution of Reconstruction MSE")
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Density")
    plt.xlim(0, 0.5) # Zoom in if needed
    plt.savefig(output_dir / "reconstruction_mse_dist.png")
    plt.close()
    
    # 3. Boxplot comparison
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=full_df, x="Dataset", y="Correlation")
    plt.title("Reconstruction Correlation by Dataset")
    plt.savefig(output_dir / "reconstruction_correlation_boxplot.png")
    plt.close()
    
    # Save raw metrics
    full_df.to_csv(output_dir / "benchmark_metrics.csv", index=False)
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()

