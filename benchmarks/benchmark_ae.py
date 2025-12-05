"""
Autoencoder Benchmark Script

Evaluates trained autoencoder models on test and OOD datasets.
Supports the new parquet-based data architecture.

Usage:
    python benchmarks/benchmark_ae.py --ckpt_path="output/ae_test/checkpoints/last.ckpt"
    python benchmarks/benchmark_ae.py --ckpt_path="..." --data_dir="data/ae_test_subset"
    python benchmarks/benchmark_ae.py --help
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fire
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ae import AESystem
from dataset import ParquetIterableDataset, collate_fn_static

# Set English font for plots
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def compute_correlation_rowwise(x, y):
    """
    Compute Pearson correlation row-wise between two tensors x and y.
    x, y: (batch_size, n_features)
    """
    x_mean = x - x.mean(dim=1, keepdim=True)
    y_mean = y - y.mean(dim=1, keepdim=True)

    x_norm = x_mean.norm(dim=1, keepdim=True)
    y_norm = y_mean.norm(dim=1, keepdim=True)

    x_norm = torch.clamp(x_norm, min=1e-8)
    y_norm = torch.clamp(y_norm, min=1e-8)

    correlation = (x_mean * y_mean).sum(dim=1, keepdim=True) / (x_norm * y_norm)
    return correlation.squeeze()


def compute_cosine_similarity_rowwise(x, y):
    """
    Compute cosine similarity row-wise between two tensors.
    """
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return (x_norm * y_norm).sum(dim=1)


def compute_r2_rowwise(y_true, y_pred):
    """
    Compute R² (coefficient of determination) row-wise.
    """
    ss_res = ((y_true - y_pred) ** 2).sum(dim=1)
    ss_tot = ((y_true - y_true.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    ss_tot = torch.clamp(ss_tot, min=1e-8)
    return 1 - ss_res / ss_tot


def load_model(ckpt_path, device):
    """
    Load AESystem model from checkpoint, handling n_genes=0 issue.
    """
    print(f"Loading model from {ckpt_path}...")

    ckpt = torch.load(ckpt_path, map_location='cpu')
    saved_n_genes = ckpt['hyper_parameters']['model']['n_genes']

    if saved_n_genes == 0:
        print(f"Warning: n_genes in checkpoint is 0, inferring from model weights...")

        # Try different possible key patterns
        possible_keys = [
            'autoencoder.encoder.hidden_layers.0.linear.weight',  # Current model structure
            'autoencoder.encoder.network.0.1.weight',             # Old model structure
        ]

        inferred_n_genes = None
        for key in possible_keys:
            if key in ckpt['state_dict']:
                weight_shape = ckpt['state_dict'][key].shape
                # For encoder first layer: [hidden_dim, n_genes]
                inferred_n_genes = weight_shape[1]
                print(f"Inferred n_genes from '{key}': {inferred_n_genes}")
                break

        if inferred_n_genes is None:
            # Try to find any encoder weight that has n_genes as input dim
            for key, tensor in ckpt['state_dict'].items():
                if 'encoder' in key and 'weight' in key and len(tensor.shape) == 2:
                    # First layer typically has largest input dimension
                    if tensor.shape[1] > 10000:  # Likely n_genes
                        inferred_n_genes = tensor.shape[1]
                        print(f"Inferred n_genes from '{key}': {inferred_n_genes}")
                        break

        if inferred_n_genes is None:
            raise ValueError("Cannot infer n_genes from checkpoint weights")

        ckpt['hyper_parameters']['model']['n_genes'] = inferred_n_genes

        # Save fixed checkpoint temporarily
        fixed_ckpt_path = ckpt_path.replace('.ckpt', '_fixed.ckpt')
        torch.save(ckpt, fixed_ckpt_path)
        print(f"Saved fixed checkpoint to {fixed_ckpt_path}")

        model_system = AESystem.load_from_checkpoint(fixed_ckpt_path, map_location=device)
    else:
        model_system = AESystem.load_from_checkpoint(ckpt_path, map_location=device)

    model_system.to(device)
    model_system.eval()
    print(f"Model loaded. n_genes: {model_system.cfg.model.n_genes}, latent_dim: {model_system.cfg.model.latent_dim}")
    return model_system


def evaluate_dataset(model, dataset, device, batch_size=2048, max_batches=None, desc="Evaluating"):
    """
    Evaluate model on a dataset.
    Returns a DataFrame with per-sample metrics.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_fn_static,
        pin_memory=True
    )

    mse_losses = []
    correlations = []
    cosine_sims = []
    r2_scores = []

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), desc=desc, total=max_batches if max_batches else None)
        for i, batch in pbar:
            if max_batches and i >= max_batches:
                break

            batch = batch.to(device)
            reconstructed, _ = model(batch)

            # Per-sample MSE
            mse = F.mse_loss(reconstructed, batch, reduction='none').mean(dim=1)
            mse_losses.append(mse.cpu())

            # Per-sample Correlation
            corr = compute_correlation_rowwise(batch, reconstructed)
            correlations.append(corr.cpu())

            # Per-sample Cosine Similarity
            cos_sim = compute_cosine_similarity_rowwise(batch, reconstructed)
            cosine_sims.append(cos_sim.cpu())

            # Per-sample R²
            r2 = compute_r2_rowwise(batch, reconstructed)
            r2_scores.append(r2.cpu())

            # Update progress bar
            if len(mse_losses) > 0:
                current_mse = torch.cat(mse_losses).mean().item()
                current_corr = torch.cat(correlations).mean().item()
                pbar.set_postfix(MSE=f"{current_mse:.4f}", Corr=f"{current_corr:.4f}")

    if not mse_losses:
        return pd.DataFrame()

    return pd.DataFrame({
        "MSE": torch.cat(mse_losses).numpy(),
        "Correlation": torch.cat(correlations).numpy(),
        "CosineSim": torch.cat(cosine_sims).numpy(),
        "R2": torch.cat(r2_scores).numpy()
    })


def collect_embeddings(model, dataset, device, n_samples=5000, batch_size=2048):
    """
    Collect latent embeddings for visualization.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_fn_static,
        pin_memory=True
    )

    embeddings = []
    originals = []
    reconstructions = []
    n_collected = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting embeddings"):
            batch = batch.to(device)
            reconstructed, z = model(batch)

            embeddings.append(z.cpu())
            originals.append(batch.cpu())
            reconstructions.append(reconstructed.cpu())

            n_collected += len(batch)
            if n_collected >= n_samples:
                break

    return (
        torch.cat(embeddings)[:n_samples],
        torch.cat(originals)[:n_samples],
        torch.cat(reconstructions)[:n_samples]
    )


def run_benchmark(
    ckpt_path: str,
    data_dir: str = None,
    output_dir: str = None,
    batch_size: int = 2048,
    max_batches: int = 200,
    n_vis_samples: int = 5000,
    device: str = "cuda",
    seed: int = 42
):
    """
    Run benchmark for Autoencoder model.

    Args:
        ckpt_path: Path to model checkpoint (.ckpt file)
        data_dir: Path to data directory containing test_shards/ and ood_shards/
                  If not specified, tries to infer from checkpoint config
        output_dir: Directory to save results. Defaults to benchmarks/results/<ckpt_name>/
        batch_size: Batch size for evaluation
        max_batches: Maximum batches to evaluate (None for full dataset)
        n_vis_samples: Number of samples for embedding visualization
        device: 'cuda' or 'cpu'
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup paths
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Auto-generate output directory if not specified
    if output_dir is None:
        ckpt_name = ckpt_path.stem
        output_dir = Path("benchmarks/results") / ckpt_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")

    # Load model
    model = load_model(str(ckpt_path), device)

    # Determine data directory
    if data_dir is None:
        # Try to infer from model config
        if hasattr(model.cfg, 'data') and hasattr(model.cfg.data, 'processed_path'):
            # Use parent directory of test_shards
            test_path = Path(model.cfg.data.processed_path.test)
            data_dir = test_path.parent
            print(f"Inferred data directory from config: {data_dir}")
        else:
            # Default fallback
            data_dir = Path("/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens/.parquet")
            print(f"Using default data directory: {data_dir}")
    else:
        data_dir = Path(data_dir)

    # Setup dataset paths
    test_path = data_dir / "test_shards"
    ood_path = data_dir / "ood_shards"

    # Load datasets
    datasets = {}
    if test_path.exists():
        print(f"\nLoading Test Dataset from {test_path}...")
        datasets["Test"] = ParquetIterableDataset(test_path, verbose=True, shuffle_shards=False, equal_length=False)
    else:
        print(f"Warning: Test path not found: {test_path}")

    if ood_path.exists():
        print(f"\nLoading OOD Dataset from {ood_path}...")
        datasets["OOD"] = ParquetIterableDataset(ood_path, verbose=True, shuffle_shards=False, equal_length=False)
    else:
        print(f"Warning: OOD path not found: {ood_path}")

    if not datasets:
        print("Error: No datasets found!")
        return

    # Evaluate each dataset
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    results = {}
    summary_stats = []

    for name, dataset in datasets.items():
        print(f"\nEvaluating {name} set...")
        df = evaluate_dataset(model, dataset, device, batch_size=batch_size,
                             max_batches=max_batches, desc=f"Eval {name}")
        df["Dataset"] = name
        results[name] = df

        # Print summary
        stats = {
            "Dataset": name,
            "N_Samples": len(df),
            "MSE_Mean": df['MSE'].mean(),
            "MSE_Std": df['MSE'].std(),
            "Corr_Mean": df['Correlation'].mean(),
            "Corr_Std": df['Correlation'].std(),
            "CosineSim_Mean": df['CosineSim'].mean(),
            "CosineSim_Std": df['CosineSim'].std(),
            "R2_Mean": df['R2'].mean(),
            "R2_Std": df['R2'].std(),
        }
        summary_stats.append(stats)

        print(f"  {name} Results:")
        print(f"    MSE:        {stats['MSE_Mean']:.4f} ± {stats['MSE_Std']:.4f}")
        print(f"    Correlation: {stats['Corr_Mean']:.4f} ± {stats['Corr_Std']:.4f}")
        print(f"    Cosine Sim:  {stats['CosineSim_Mean']:.4f} ± {stats['CosineSim_Std']:.4f}")
        print(f"    R²:          {stats['R2_Mean']:.4f} ± {stats['R2_Std']:.4f}")

    # Combine results
    full_df = pd.concat(results.values(), ignore_index=True)

    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / "summary_stats.csv", index=False)
    print(f"\nSummary saved to {output_dir / 'summary_stats.csv'}")

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    # 1. Correlation Distribution
    print("Generating correlation distribution plot...")
    plt.figure(figsize=(10, 6))
    for name in results.keys():
        data = results[name]['Correlation']
        sns.kdeplot(data, label=f"{name} (μ={data.mean():.3f})", fill=True, alpha=0.3)
    plt.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='0.9 threshold')
    plt.title("Distribution of Reconstruction Correlation")
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Density")
    plt.legend()
    plt.xlim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_distribution.png", dpi=150)
    plt.close()

    # 2. MSE Distribution
    print("Generating MSE distribution plot...")
    plt.figure(figsize=(10, 6))
    for name in results.keys():
        data = results[name]['MSE']
        sns.kdeplot(data, label=f"{name} (μ={data.mean():.4f})", fill=True, alpha=0.3)
    plt.title("Distribution of Reconstruction MSE")
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Density")
    plt.legend()
    mse_max = min(0.5, full_df['MSE'].quantile(0.99))
    plt.xlim(0, mse_max)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "mse_distribution.png", dpi=150)
    plt.close()

    # 3. Boxplot Comparison
    print("Generating boxplot comparison...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    metrics = ['Correlation', 'MSE', 'CosineSim', 'R2']
    titles = ['Correlation', 'MSE', 'Cosine Similarity', 'R² Score']

    for ax, metric, title in zip(axes, metrics, titles):
        sns.boxplot(data=full_df, x="Dataset", y=metric, ax=ax, palette="Set2")
        ax.set_title(title)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_boxplot.png", dpi=150)
    plt.close()

    # 4. Scatter: Correlation vs MSE
    print("Generating correlation vs MSE scatter plot...")
    plt.figure(figsize=(10, 8))
    for name in results.keys():
        data = results[name]
        plt.scatter(data['Correlation'], data['MSE'], alpha=0.3, s=5, label=name)
    plt.xlabel("Correlation")
    plt.ylabel("MSE")
    plt.title("Reconstruction Quality: Correlation vs MSE")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_vs_mse.png", dpi=150)
    plt.close()

    # 5. Latent Space Visualization (PCA)
    print("Generating latent space visualization...")

    # Collect embeddings from first available dataset
    first_dataset_name = list(datasets.keys())[0]
    first_dataset = datasets[first_dataset_name]
    embeddings, originals, reconstructions = collect_embeddings(
        model, first_dataset, device, n_samples=n_vis_samples, batch_size=batch_size
    )

    # PCA on embeddings
    embeddings_np = embeddings.numpy()
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_np)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Latent space colored by reconstruction quality
    corrs = compute_correlation_rowwise(originals, reconstructions).numpy()
    sc = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=corrs, cmap='RdYlGn', s=5, alpha=0.6, vmin=0.5, vmax=1.0)
    plt.colorbar(sc, ax=axes[0], label='Correlation')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title(f'Latent Space ({first_dataset_name}) - Colored by Reconstruction Quality')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # Plot 2: Latent space colored by expression sum (proxy for cell size)
    expr_sum = originals.sum(dim=1).numpy()
    sc2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=expr_sum, cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(sc2, ax=axes[1], label='Total Expression')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title(f'Latent Space ({first_dataset_name}) - Colored by Total Expression')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "latent_space_pca.png", dpi=150)
    plt.close()

    # 6. Sample Reconstruction Examples
    print("Generating reconstruction examples...")
    n_examples = 6
    fig, axes = plt.subplots(2, n_examples, figsize=(3*n_examples, 6))

    # Sample random indices with different correlation levels
    corr_quantiles = np.percentile(corrs, [10, 30, 50, 70, 90, 99])
    example_indices = []
    for q in corr_quantiles:
        idx = np.argmin(np.abs(corrs - q))
        example_indices.append(idx)

    for i, idx in enumerate(example_indices):
        orig = originals[idx].numpy()
        recon = reconstructions[idx].numpy()
        corr_val = corrs[idx]

        # Top row: scatter plot
        axes[0, i].scatter(orig, recon, alpha=0.3, s=2)
        max_val = max(orig.max(), recon.max())
        axes[0, i].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        axes[0, i].set_xlabel('Original')
        axes[0, i].set_ylabel('Reconstructed')
        axes[0, i].set_title(f'Corr={corr_val:.3f}')
        axes[0, i].set_aspect('equal', adjustable='box')

        # Bottom row: bar plot of top genes
        n_top_genes = 20
        top_gene_idx = np.argsort(orig)[-n_top_genes:]
        x = np.arange(n_top_genes)
        width = 0.35
        axes[1, i].bar(x - width/2, orig[top_gene_idx], width, label='Original', alpha=0.7)
        axes[1, i].bar(x + width/2, recon[top_gene_idx], width, label='Reconstructed', alpha=0.7)
        axes[1, i].set_xlabel('Top Genes')
        axes[1, i].set_ylabel('Expression')
        if i == 0:
            axes[1, i].legend(fontsize=8)

    plt.suptitle('Reconstruction Examples (Low to High Correlation)', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_examples.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save raw metrics
    full_df.to_csv(output_dir / "benchmark_metrics.csv", index=False)

    # Print final summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles generated:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")

    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    fire.Fire(run_benchmark)
