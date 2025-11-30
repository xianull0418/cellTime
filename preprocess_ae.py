import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import uuid
import shutil
try:
    import zarr
    import numcodecs
except ImportError:
    zarr = None

# Suppress warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable for workers
shared_target_genes = None

def init_worker(genes):
    """Initialize worker process with shared target genes."""
    global shared_target_genes
    shared_target_genes = genes

def load_gene_vocab(vocab_path):
    """Load gene vocabulary from file."""
    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    
    logger.info(f"Loading vocabulary from {path}")
    with open(path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    return genes

def process_and_write_task(args):
    """
    Worker task to process a single h5ad file and write directly to a temporary Zarr or Parquet file.
    """
    file_path, is_ood, min_genes, target_sum, temp_dir, output_format = args
    
    global shared_target_genes
    if shared_target_genes is None:
        return None

    try:
        # Read h5ad
        adata = sc.read_h5ad(file_path)
        
        # Fix gene matching: Map Ensembl IDs (index) to Gene Symbols (gene_symbols column)
        if "gene_symbols" in adata.var.columns:
            # Check overlap with vocab using symbols
            # We want to index adata by symbols to match our vocab
            adata.var_names = adata.var["gene_symbols"].astype(str)
            adata.var_names_make_unique()
        else:
            # Fallback or already correct
            adata.var_names_make_unique()
        
        # 1. Filter cells
        if min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)
        
        if adata.shape[0] == 0:
            return None

        # 2. Normalize
        sc.pp.normalize_total(adata, target_sum=target_sum)

        # 3. Log1p
        sc.pp.log1p(adata)

        # 4. Align genes
        if hasattr(adata.X, 'toarray'):
            data = adata.X.toarray()
        else:
            data = adata.X
            
        df = pd.DataFrame(
            data, 
            index=adata.obs_names, 
            columns=adata.var_names
        )
        
        df = df.astype(np.float32)

        # Reindex
        df_aligned = df.reindex(columns=shared_target_genes, fill_value=0.0)
        df_aligned = df_aligned.astype(np.float32, copy=False)
        
        if df_aligned.isna().values.any():
             df_aligned = df_aligned.fillna(0.0)
        
        # Write to temp files
        unique_id = str(uuid.uuid4())
        stats = {'train': 0, 'val': 0, 'test': 0, 'ood': 0}
        created_files = {}

        def write_shard(df_shard, prefix):
            if len(df_shard) == 0: return
            
            if output_format == "zarr":
                fname = temp_dir / f"{prefix}_{unique_id}.zarr"
                # Create Zarr group and dataset
                store = zarr.DirectoryStore(str(fname))
                root = zarr.group(store=store, overwrite=True)
                
                # Standardize on "X" for data
                # Chunks: (chunk_size, n_genes). Default auto chunking is often fine, but let's be explicit
                # Given typical shard size ~6000 rows, maybe 1 chunk or a few.
                root.create_dataset("X", data=df_shard.values, chunks=(None, None), dtype="float32")
                
                # Optionally save obs/var for compatibility (simplified)
                # root.create_dataset("obs", data=df_shard.index.values.astype(str))
                # root.create_dataset("var", data=df_shard.columns.values.astype(str))
                
                created_files[prefix] = str(fname)
                stats[prefix] += len(df_shard)
                
            else: # parquet
                fname = temp_dir / f"{prefix}_{unique_id}.parquet"
                pq.write_table(pa.Table.from_pandas(df_shard), fname, compression='snappy')
                created_files[prefix] = str(fname)
                stats[prefix] += len(df_shard)

        if is_ood:
            write_shard(df_aligned, 'ood')
        else:
            n_cells = len(df_aligned)
            indices = np.random.permutation(n_cells)
            
            n_train = int(n_cells * 0.90)
            n_val = int(n_cells * 0.05)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]
            
            if len(train_idx) > 0: write_shard(df_aligned.iloc[train_idx], 'train')
            if len(val_idx) > 0: write_shard(df_aligned.iloc[val_idx], 'val')
            if len(test_idx) > 0: write_shard(df_aligned.iloc[test_idx], 'test')

        return stats, created_files

    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="Preprocess single-cell data for Autoencoder (Parallel I/O Sharded)")
    parser.add_argument("--csv_path", type=str, default="data_info/ae_data_info.csv", help="Path to data info CSV")
    parser.add_argument("--vocab_path", type=str, default="data_info/gene_order.tsv", help="Path to gene vocabulary")
    parser.add_argument("--output_dir", type=str, default="data/ae_processed", help="Output directory for processed files")
    parser.add_argument("--min_genes", type=int, default=200, help="Minimum genes per cell")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--format", type=str, default="zarr", choices=["parquet", "zarr"], help="Output format")
    
    args = parser.parse_args()
    
    if args.format == "zarr" and zarr is None:
        raise ImportError("Please install zarr and numcodecs to use zarr format.")
    
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    # Append format to output dir to avoid conflicts if needed, or just assume user handles it
    # output_dir = output_dir / args.format 
    
    temp_dir = output_dir / "temp_chunks"
    
    # Cleanup and create dirs
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    target_genes = load_gene_vocab(args.vocab_path)
    logger.info(f"Target vocabulary size: {len(target_genes)}")
    
    # Load CSV
    if not Path(args.csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    
    info_df = pd.read_csv(args.csv_path)
    logger.info(f"Found {len(info_df)} files in CSV")
    
    # Prepare tasks
    tasks = []
    for idx, row in info_df.iterrows():
        file_path = row['file_path']
        is_ood = row.get('full_validation_dataset', 0) == 1
        
        if Path(file_path).exists():
            tasks.append((file_path, is_ood, args.min_genes, 1e4, temp_dir, args.format))
    
    logger.info(f"Starting processing with {args.num_workers} workers (Output: {args.format})...")
    
    # Output directories for shards
    shard_dirs = {
        'train': output_dir / "train_shards",
        'val': output_dir / "val_shards",
        'test': output_dir / "test_shards",
        'ood': output_dir / "ood_shards"
    }
    for d in shard_dirs.values():
        d.mkdir(exist_ok=True)

    total_stats = {k: 0 for k in shard_dirs.keys()}
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker, initargs=(target_genes,)) as executor:
        futures = [executor.submit(process_and_write_task, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                stats, created_files = result
                for key, count in stats.items():
                    total_stats[key] += count
                
                # Move temp files to final shard directories
                for key, temp_path in created_files.items():
                    temp_path = Path(temp_path)
                    final_path = shard_dirs[key] / temp_path.name
                    # For zarr directory, we need to move the whole directory
                    if temp_path.is_dir():
                        if final_path.exists(): shutil.rmtree(final_path)
                        shutil.move(str(temp_path), str(final_path))
                    else:
                        shutil.move(str(temp_path), str(final_path))

    logger.info(f"Processing statistics: {total_stats}")
    
    # Cleanup temp dir
    shutil.rmtree(temp_dir)
    
    logger.info("Workflow complete. Data is stored in sharded directories:")
    for k, v in shard_dirs.items():
        logger.info(f"  - {k}: {v}")

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    main()
