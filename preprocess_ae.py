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
import shutil
import json
import gc
from scipy.sparse import csr_matrix, coo_matrix

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

def process_single_file(args):
    """
    Worker task to process a single h5ad file.
    Returns processed DataFrames.
    """
    file_path, is_ood, min_genes, target_sum = args

    global shared_target_genes
    if shared_target_genes is None:
        return None

    try:
        # Read h5ad
        adata = sc.read_h5ad(file_path)

        # Fix gene matching
        if "gene_symbols" in adata.var.columns:
            adata.var_names = adata.var["gene_symbols"].astype(str)
            adata.var_names_make_unique()
        else:
            adata.var_names_make_unique()

        # Filter cells
        if min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)

        if adata.shape[0] == 0:
            return None

        # Normalize and log1p
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

        # Align genes
        if hasattr(adata.X, 'toarray'):
            data = adata.X.toarray()
        else:
            data = adata.X

        df = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
        df = df.astype(np.float32)

        # Reindex to target genes
        df_aligned = df.reindex(columns=shared_target_genes, fill_value=0.0)
        df_aligned = df_aligned.astype(np.float32, copy=False)

        if df_aligned.isna().values.any():
             df_aligned = df_aligned.fillna(0.0)

        # Split into train/val/test
        if is_ood:
            return {'ood': df_aligned}
        else:
            n_cells = len(df_aligned)
            indices = np.random.permutation(n_cells)

            n_train = int(n_cells * 0.90)
            n_val = int(n_cells * 0.05)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]

            result = {}
            if len(train_idx) > 0:
                result['train'] = df_aligned.iloc[train_idx]
            if len(val_idx) > 0:
                result['val'] = df_aligned.iloc[val_idx]
            if len(test_idx) > 0:
                result['test'] = df_aligned.iloc[test_idx]

            return result

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def process_file_batch(batch_args):
    """
    Worker task to process multiple h5ad files in one go.
    Writes results to temporary parquet files to avoid large data serialization.
    Returns list of temporary file paths and metadata.
    """
    task_list, min_genes, target_sum, temp_dir, worker_id = batch_args

    global shared_target_genes
    if shared_target_genes is None:
        return []

    results = []
    for file_idx, (file_path, is_ood) in enumerate(task_list):
        try:
            # Read h5ad
            adata = sc.read_h5ad(file_path)

            # Fix gene matching
            if "gene_symbols" in adata.var.columns:
                adata.var_names = adata.var["gene_symbols"].astype(str)
                adata.var_names_make_unique()
            else:
                adata.var_names_make_unique()

            # Filter cells
            if min_genes > 0:
                sc.pp.filter_cells(adata, min_genes=min_genes)

            if adata.shape[0] == 0:
                continue

            # Normalize and log1p
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)

            # Align genes
            if hasattr(adata.X, 'toarray'):
                data = adata.X.toarray()
            else:
                data = adata.X

            df = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
            df = df.astype(np.float32)

            # Reindex to target genes
            df_aligned = df.reindex(columns=shared_target_genes, fill_value=0.0)
            df_aligned = df_aligned.astype(np.float32, copy=False)

            if df_aligned.isna().values.any():
                df_aligned = df_aligned.fillna(0.0)

            # Split into train/val/test and write to temp files
            if is_ood:
                temp_file = temp_dir / f"temp_worker{worker_id}_file{file_idx}_ood.parquet"
                table = pa.Table.from_pandas(df_aligned.reset_index(drop=True))
                pq.write_table(table, temp_file, compression='snappy')
                results.append({'split': 'ood', 'path': str(temp_file), 'n_cells': len(df_aligned)})
            else:
                n_cells = len(df_aligned)
                indices = np.random.permutation(n_cells)

                n_train = int(n_cells * 0.90)
                n_val = int(n_cells * 0.05)

                train_idx = indices[:n_train]
                val_idx = indices[n_train:n_train+n_val]
                test_idx = indices[n_train+n_val:]

                if len(train_idx) > 0:
                    temp_file = temp_dir / f"temp_worker{worker_id}_file{file_idx}_train.parquet"
                    table = pa.Table.from_pandas(df_aligned.iloc[train_idx].reset_index(drop=True))
                    pq.write_table(table, temp_file, compression='snappy')
                    results.append({'split': 'train', 'path': str(temp_file), 'n_cells': len(train_idx)})

                if len(val_idx) > 0:
                    temp_file = temp_dir / f"temp_worker{worker_id}_file{file_idx}_val.parquet"
                    table = pa.Table.from_pandas(df_aligned.iloc[val_idx].reset_index(drop=True))
                    pq.write_table(table, temp_file, compression='snappy')
                    results.append({'split': 'val', 'path': str(temp_file), 'n_cells': len(val_idx)})

                if len(test_idx) > 0:
                    temp_file = temp_dir / f"temp_worker{worker_id}_file{file_idx}_test.parquet"
                    table = pa.Table.from_pandas(df_aligned.iloc[test_idx].reset_index(drop=True))
                    pq.write_table(table, temp_file, compression='snappy')
                    results.append({'split': 'test', 'path': str(temp_file), 'n_cells': len(test_idx)})

            # Clean up
            del adata, df, df_aligned
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    return results


def create_tiledb_from_parquet_shards(shard_dir: Path, output_dir: Path, split_name: str, target_genes: list):
    """
    Convert Parquet shards to TileDB format (CellArr-style).

    Creates:
    - counts: Sparse matrix TileDB (cell_index, gene_index, value)
    - cell_metadata: Cell metadata TileDB
    - gene_annotation: Gene annotation TileDB

    This enables efficient random access like scimilarity.
    """
    try:
        import tiledb
    except ImportError:
        logger.error("TileDB not installed. Run: pip install tiledb")
        return None

    logger.info(f"Converting {split_name} shards to TileDB format...")

    # Create output directory
    tiledb_dir = output_dir / f"{split_name}_tiledb"
    if tiledb_dir.exists():
        shutil.rmtree(tiledb_dir)
    tiledb_dir.mkdir(parents=True)

    # Read all parquet files and accumulate data
    parquet_files = sorted(shard_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {shard_dir}")
        return None

    # Collect all data
    all_data = []
    cell_indices = []
    current_cell_idx = 0

    for pq_file in tqdm(parquet_files, desc=f"Reading {split_name} shards"):
        table = pq.read_table(pq_file)
        df = table.to_pandas()
        n_cells = len(df)

        # Convert to sparse and collect COO data
        data_matrix = df.values.astype(np.float32)

        for local_idx in range(n_cells):
            row = data_matrix[local_idx]
            nonzero_mask = row != 0
            gene_indices = np.where(nonzero_mask)[0]
            values = row[nonzero_mask]

            if len(values) > 0:
                all_data.append({
                    'cell_index': np.full(len(values), current_cell_idx, dtype=np.int64),
                    'gene_index': gene_indices.astype(np.int64),
                    'value': values.astype(np.float32)
                })

            cell_indices.append(current_cell_idx)
            current_cell_idx += 1

        del df, data_matrix
        gc.collect()

    total_cells = current_cell_idx
    n_genes = len(target_genes)
    logger.info(f"  Total cells: {total_cells:,}, Genes: {n_genes:,}")

    # Concatenate all COO data
    if all_data:
        all_cell_indices = np.concatenate([d['cell_index'] for d in all_data])
        all_gene_indices = np.concatenate([d['gene_index'] for d in all_data])
        all_values = np.concatenate([d['value'] for d in all_data])
    else:
        all_cell_indices = np.array([], dtype=np.int64)
        all_gene_indices = np.array([], dtype=np.int64)
        all_values = np.array([], dtype=np.float32)

    del all_data
    gc.collect()

    nnz = len(all_values)
    sparsity = 1.0 - (nnz / (total_cells * n_genes)) if total_cells * n_genes > 0 else 0
    logger.info(f"  Non-zero elements: {nnz:,}, Sparsity: {sparsity:.2%}")

    # Create counts TileDB (sparse matrix)
    counts_uri = str(tiledb_dir / "counts")

    # Define sparse array schema
    dom = tiledb.Domain(
        tiledb.Dim(name="cell_index", domain=(0, total_cells - 1), tile=min(10000, total_cells), dtype=np.int64),
        tiledb.Dim(name="gene_index", domain=(0, n_genes - 1), tile=n_genes, dtype=np.int64),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=True,
        attrs=[tiledb.Attr(name="data", dtype=np.float32)],
        allows_duplicates=False,
    )

    tiledb.Array.create(counts_uri, schema)

    # Write data in chunks to avoid memory issues
    chunk_size = 10000000  # 10M elements per chunk
    with tiledb.open(counts_uri, 'w') as arr:
        for i in range(0, nnz, chunk_size):
            end_idx = min(i + chunk_size, nnz)
            arr[all_cell_indices[i:end_idx], all_gene_indices[i:end_idx]] = all_values[i:end_idx]

    logger.info(f"  ✓ Created counts TileDB: {counts_uri}")

    del all_cell_indices, all_gene_indices, all_values
    gc.collect()

    # Create cell_metadata TileDB
    cell_metadata_uri = str(tiledb_dir / "cell_metadata")
    cell_df = pd.DataFrame({
        'cell_index': np.arange(total_cells, dtype=np.int64),
        'split': split_name,
    })

    # Create schema for cell metadata
    dom = tiledb.Domain(
        tiledb.Dim(name="cell_index", domain=(0, total_cells - 1), tile=min(10000, total_cells), dtype=np.int64),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        attrs=[
            tiledb.Attr(name="split", dtype='ascii', var=True),
        ],
    )
    tiledb.Array.create(cell_metadata_uri, schema)

    with tiledb.open(cell_metadata_uri, 'w') as arr:
        arr[:] = {'split': np.array([split_name] * total_cells, dtype=object)}

    logger.info(f"  ✓ Created cell_metadata TileDB: {cell_metadata_uri}")

    # Create gene_annotation TileDB
    gene_annotation_uri = str(tiledb_dir / "gene_annotation")
    gene_df = pd.DataFrame({
        'gene_index': np.arange(n_genes, dtype=np.int64),
        'gene_symbol': target_genes,
    })

    dom = tiledb.Domain(
        tiledb.Dim(name="gene_index", domain=(0, n_genes - 1), tile=n_genes, dtype=np.int64),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        attrs=[
            tiledb.Attr(name="gene_symbol", dtype='ascii', var=True),
        ],
    )
    tiledb.Array.create(gene_annotation_uri, schema)

    with tiledb.open(gene_annotation_uri, 'w') as arr:
        arr[:] = {'gene_symbol': np.array(target_genes, dtype=object)}

    logger.info(f"  ✓ Created gene_annotation TileDB: {gene_annotation_uri}")

    # Save metadata
    metadata = {
        'n_cells': total_cells,
        'n_genes': n_genes,
        'nnz': nnz,
        'sparsity': sparsity,
        'format': 'tiledb_sparse',
        'counts_uri': 'counts',
        'cell_metadata_uri': 'cell_metadata',
        'gene_annotation_uri': 'gene_annotation',
    }
    with open(tiledb_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  ✓ Metadata saved to {tiledb_dir / 'metadata.json'}")

    return {
        'n_cells': total_cells,
        'n_genes': n_genes,
        'nnz': nnz,
        'sparsity': sparsity,
        'path': str(tiledb_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess single-cell data to sharded parquet or TileDB files")
    parser.add_argument("--csv_path", type=str, default="data_info/ae_data_info.csv")
    parser.add_argument("--vocab_path", type=str, default="data_info/gene_order.tsv")
    parser.add_argument("--output_dir", type=str, default="data/ae_processed")
    parser.add_argument("--output_format", type=str, default="parquet", choices=["parquet", "tiledb"],
                        help="Output format: parquet (default) or tiledb (CellArr-style)")
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers (recommend 32-64)")
    parser.add_argument("--batch_size", type=int, default=100, help="Files per batch for parallel processing")
    parser.add_argument("--files_per_worker_batch", type=int, default=10, help="Number of files each worker processes in one task (reduces I/O)")
    parser.add_argument("--cells_per_shard", type=int, default=200000, help="Target cells per shard (like scGPT)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temp directory for worker outputs
    temp_dir = output_dir / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)

    # Create shard directories
    shard_dirs = {}
    for split_name in ['train', 'val', 'test', 'ood']:
        shard_dir = output_dir / f"{split_name}_shards"
        shard_dir.mkdir(exist_ok=True)
        shard_dirs[split_name] = shard_dir

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
            tasks.append((file_path, is_ood, args.min_genes, 1e4))

    total_files = len(tasks)
    logger.info(f"Processing {total_files} files with {args.num_workers} workers")
    logger.info(f"Batch size: {args.batch_size} files per batch")
    logger.info(f"Files per worker batch: {args.files_per_worker_batch} (reduces I/O overhead)")
    logger.info(f"Target cells per shard: {args.cells_per_shard:,} (like scGPT)")
    logger.info("=" * 80)

    # Track statistics
    total_stats = {'train': 0, 'val': 0, 'test': 0, 'ood': 0}
    shard_counters = {'train': 0, 'val': 0, 'test': 0, 'ood': 0}

    # Accumulate temp file paths until reaching cells_per_shard
    shard_accumulators = {'train': [], 'val': [], 'test': [], 'ood': []}
    shard_cell_counts = {'train': 0, 'val': 0, 'test': 0, 'ood': 0}

    num_batches = (total_files + args.batch_size - 1) // args.batch_size

    def write_shard_if_ready(split_name, force=False):
        """Write shard if accumulator has enough cells or force=True"""
        if not shard_accumulators[split_name]:
            return

        if force or shard_cell_counts[split_name] >= args.cells_per_shard:
            # Read and concatenate accumulated temp files
            dfs = []
            for temp_file in shard_accumulators[split_name]:
                df = pq.read_table(temp_file).to_pandas()
                dfs.append(df)

            combined_df = pd.concat(dfs, axis=0, ignore_index=True)
            n_samples = len(combined_df)

            # Shuffle within shard
            combined_df = combined_df.sample(frac=1.0, random_state=args.seed + shard_counters[split_name]).reset_index(drop=True)

            # Write shard
            shard_file = shard_dirs[split_name] / f"shard_{shard_counters[split_name]:04d}.parquet"
            table = pa.Table.from_pandas(combined_df)
            pq.write_table(
                table,
                shard_file,
                compression='snappy',
                use_dictionary=True,
                write_statistics=True
            )

            shard_counters[split_name] += 1
            total_stats[split_name] += n_samples

            logger.info(f"  → {split_name}: shard_{shard_counters[split_name]-1:04d}.parquet ({n_samples:,} cells)")

            # Delete temp files
            for temp_file in shard_accumulators[split_name]:
                Path(temp_file).unlink(missing_ok=True)

            # Reset accumulator
            shard_accumulators[split_name] = []
            shard_cell_counts[split_name] = 0

            # Clean up
            del combined_df, table, dfs
            gc.collect()

    with tqdm(total=total_files, desc="Processing files") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, total_files)
            batch_tasks = tasks[start_idx:end_idx]

            logger.info(f"Batch {batch_idx+1}/{num_batches}: Processing files {start_idx+1}-{end_idx}")

            # Group tasks into mini-batches for each worker
            # Each worker will process files_per_worker_batch files in one go
            worker_batches = []
            for i in range(0, len(batch_tasks), args.files_per_worker_batch):
                mini_batch = batch_tasks[i:i + args.files_per_worker_batch]
                # Extract (file_path, is_ood) from tasks
                file_list = [(fp, is_ood) for fp, is_ood, _, _ in mini_batch]
                worker_id = batch_idx * 1000 + i  # Unique worker ID
                worker_batches.append((file_list, args.min_genes, 1e4, temp_dir, worker_id))

            # Process batch in parallel
            batch_results = []
            files_processed = 0
            with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker, initargs=(target_genes,)) as executor:
                futures_to_count = {}
                for batch_args in worker_batches:
                    future = executor.submit(process_file_batch, batch_args)
                    futures_to_count[future] = len(batch_args[0])  # Number of files in this mini-batch

                for future in as_completed(futures_to_count):
                    result_list = future.result()  # Returns list of {split, path, n_cells} dicts
                    num_files = futures_to_count[future]
                    files_processed += num_files

                    if result_list:
                        batch_results.extend(result_list)

                    # Update progress bar by number of files processed
                    pbar.update(num_files)

            # Add batch results (temp file paths) to accumulators
            for result in batch_results:
                split_name = result['split']
                shard_accumulators[split_name].append(result['path'])
                shard_cell_counts[split_name] += result['n_cells']

                # Write shard if accumulator is full
                write_shard_if_ready(split_name)

            # Clean up batch
            del batch_results
            gc.collect()

            logger.info(f"Batch {batch_idx+1} completed. Current status:")
            for split in total_stats:
                logger.info(f"  {split}: {total_stats[split]:,} cells written, "
                           f"{shard_cell_counts[split]:,} cells buffered, "
                           f"{shard_counters[split]} shards")

    # Write remaining data in accumulators
    logger.info("=" * 80)
    logger.info("Writing remaining buffered data...")
    for split_name in ['train', 'val', 'test', 'ood']:
        write_shard_if_ready(split_name, force=True)

    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temp directory: {temp_dir}")

    # Save metadata
    logger.info("=" * 80)
    logger.info("Writing metadata...")

    metadata = {
        'total_samples': total_stats,
        'n_genes': len(target_genes),
        'n_shards': shard_counters,
        'shard_dirs': {
            'train': str(shard_dirs['train']) if total_stats['train'] > 0 else None,
            'val': str(shard_dirs['val']) if total_stats['val'] > 0 else None,
            'test': str(shard_dirs['test']) if total_stats['test'] > 0 else None,
            'ood': str(shard_dirs['ood']) if total_stats['ood'] > 0 else None,
        }
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Generate metadata cache for DataLoader (important for multi-worker loading!)
    logger.info("=" * 80)
    logger.info("Generating metadata cache for fast DataLoader initialization...")

    for split_name in ['train', 'val', 'test', 'ood']:
        if total_stats[split_name] > 0:
            shard_dir = shard_dirs[split_name]
            cache_file = shard_dir / "metadata_cache.json"

            # Create cache with shard info
            cache = {
                'n_files': shard_counters[split_name],
                'files_hash': shard_counters[split_name],  # Simple file count hash
                'n_cells': total_stats[split_name],
                'shard_lengths': [],
                'shard_offsets': []
            }

            # Read lengths from shards
            cumulative = 0
            for shard_file in sorted(shard_dir.glob("*.parquet")):
                meta = pq.read_metadata(shard_file)
                n_rows = meta.num_rows
                cache['shard_lengths'].append(n_rows)
                cache['shard_offsets'].append(cumulative)
                cumulative += n_rows

            # Write cache
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)

            logger.info(f"  ✓ {split_name}_shards/metadata_cache.json created")

    logger.info("=" * 80)
    logger.info("Preprocessing COMPLETE!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Generated shard directories:")
    for split, count in total_stats.items():
        if count > 0:
            shard_dir = shard_dirs[split]
            total_size = sum(f.stat().st_size for f in shard_dir.glob("*.parquet")) / 1024 / 1024
            logger.info(f"  {split}_shards/: {count:,} samples, {shard_counters[split]} shards, {total_size:.1f} MB")
    logger.info("")
    logger.info("✅ Metadata caches generated - you can now use num_workers > 0 in training!")

    # Convert to TileDB if requested
    if args.output_format == "tiledb":
        logger.info("")
        logger.info("=" * 80)
        logger.info("Converting Parquet shards to TileDB format...")
        logger.info("=" * 80)

        tiledb_stats = {}
        for split_name in ['train', 'val', 'test', 'ood']:
            if total_stats[split_name] > 0:
                result = create_tiledb_from_parquet_shards(
                    shard_dirs[split_name],
                    output_dir,
                    split_name,
                    target_genes
                )
                if result:
                    tiledb_stats[split_name] = result

        logger.info("")
        logger.info("=" * 80)
        logger.info("TileDB conversion COMPLETE!")
        logger.info("")
        for split, stats in tiledb_stats.items():
            logger.info(f"  {split}_tiledb/: {stats['n_cells']:,} cells, "
                       f"{stats['nnz']:,} non-zeros, {stats['sparsity']:.2%} sparsity")
        logger.info("")
        logger.info("✅ TileDB format ready - use --data.dataset_type=tiledb in training!")

    logger.info("=" * 80)

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    main()
