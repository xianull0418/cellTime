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

def process_single_h5ad_task(args):
    """
    Worker task to process a single h5ad file.
    Args:
        args: tuple (file_path, is_ood, min_genes, target_sum)
    Returns:
        tuple: (df_aligned, is_ood, file_path) or (None, is_ood, file_path)
    """
    file_path, is_ood, min_genes, target_sum = args
    
    # Access shared variable
    global shared_target_genes
    if shared_target_genes is None:
        return None, is_ood, file_path

    try:
        # Read h5ad
        adata = sc.read_h5ad(file_path)
        adata.var_names_make_unique()
        
        # 1. Filter cells
        if min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)
        
        if adata.shape[0] == 0:
            return None, is_ood, file_path

        # 2. Normalize
        sc.pp.normalize_total(adata, target_sum=target_sum)

        # 3. Log1p
        sc.pp.log1p(adata)

        # 4. Align genes
        # Use sparse to dense strategy to manage memory if needed
        # For now, assuming dense fits in worker memory
        if hasattr(adata.X, 'toarray'):
            data = adata.X.toarray()
        else:
            data = adata.X
            
        # Create DataFrame
        df = pd.DataFrame(
            data, 
            index=adata.obs_names, 
            columns=adata.var_names
        )
        
        # Optimize memory: convert to float32 immediately
        df = df.astype(np.float32)

        # Reindex to target genes (filling missing with 0)
        # copy=False to avoid extra copy if possible
        df_aligned = df.reindex(columns=shared_target_genes, fill_value=0.0)
        
        # Ensure float32 again (reindex with 0.0 might introduce float64)
        df_aligned = df_aligned.astype(np.float32, copy=False)
        
        # Handle NaNs if any
        if df_aligned.isna().values.any():
             df_aligned = df_aligned.fillna(0.0)

        return df_aligned, is_ood, file_path

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, is_ood, file_path

def main():
    parser = argparse.ArgumentParser(description="Preprocess single-cell data for Autoencoder (Parallel)")
    parser.add_argument("--csv_path", type=str, default="ae_data_info.csv", help="Path to data info CSV")
    parser.add_argument("--vocab_path", type=str, default="gene_order.tsv", help="Path to gene vocabulary")
    parser.add_argument("--output_dir", type=str, default="data/ae_processed", help="Output directory for parquet files")
    parser.add_argument("--min_genes", type=int, default=200, help="Minimum genes per cell")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    target_genes = load_gene_vocab(args.vocab_path)
    logger.info(f"Target vocabulary size: {len(target_genes)}")
    
    # Load CSV
    if not Path(args.csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    
    info_df = pd.read_csv(args.csv_path)
    logger.info(f"Found {len(info_df)} files in CSV")
    
    # Writers cache
    writers = {}
    
    def get_writer(split_name, schema):
        if split_name not in writers:
            outfile = output_dir / f"{split_name}.parquet"
            writers[split_name] = pq.ParquetWriter(
                outfile,
                schema=schema,
                compression='snappy'
            )
        return writers[split_name]

    # Stats
    stats = {k: 0 for k in ['train', 'val', 'test', 'ood']}

    # Prepare tasks
    tasks = []
    for idx, row in info_df.iterrows():
        file_path = row['file_path']
        is_ood = row.get('full_validation_dataset', 0) == 1
        
        if Path(file_path).exists():
            tasks.append((file_path, is_ood, args.min_genes, 1e4))
    
    logger.info(f"Starting processing with {args.num_workers} workers...")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker, initargs=(target_genes,)) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_h5ad_task, task) for task in tasks]
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            df_aligned, is_ood, fpath = future.result()
            
            if df_aligned is None or df_aligned.empty:
                continue
                
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(df_aligned)
            schema = table.schema
            
            if is_ood:
                # OOD Data -> ood.parquet
                writer = get_writer('ood', schema)
                writer.write_table(table)
                stats['ood'] += len(df_aligned)
            else:
                # Training Data -> Split 90/5/5
                n_cells = len(df_aligned)
                indices = np.random.permutation(n_cells)
                
                n_train = int(n_cells * 0.90)
                n_val = int(n_cells * 0.05)
                # Remainder to test
                
                train_idx = indices[:n_train]
                val_idx = indices[n_train:n_train+n_val]
                test_idx = indices[n_train+n_val:]
                
                if len(train_idx) > 0:
                    t_df = df_aligned.iloc[train_idx]
                    w = get_writer('train', schema)
                    w.write_table(pa.Table.from_pandas(t_df, schema=schema))
                    stats['train'] += len(t_df)
                    
                if len(val_idx) > 0:
                    v_df = df_aligned.iloc[val_idx]
                    w = get_writer('val', schema)
                    w.write_table(pa.Table.from_pandas(v_df, schema=schema))
                    stats['val'] += len(v_df)
                    
                if len(test_idx) > 0:
                    test_df = df_aligned.iloc[test_idx]
                    w = get_writer('test', schema)
                    w.write_table(pa.Table.from_pandas(test_df, schema=schema))
                    stats['test'] += len(test_df)

    # Close all writers
    for w in writers.values():
        w.close()
        
    logger.info("Processing complete.")
    logger.info(f"Statistics: {stats}")

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)  # Ensure fork is used on Linux
    main()
