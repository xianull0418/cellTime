"""
直接从 h5ad 文件构建 TileDB 数据集
参考 CellArr (https://github.com/CellArr/cellarr) 的方式

输出结构:
  output_dir/
  ├── train_tiledb/
  │   ├── counts/          # 稀疏矩阵 TileDB (cell_index, gene_index -> value)
  │   ├── cell_metadata/   # 细胞元数据
  │   ├── gene_annotation/ # 基因信息
  │   └── metadata.json
  ├── val_tiledb/
  ├── test_tiledb/
  └── ood_tiledb/
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import tiledb
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json
import gc
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_gene_vocab(vocab_path: str) -> List[str]:
    """Load gene vocabulary from file."""
    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")

    with open(path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def process_h5ad_file(args) -> Optional[Dict]:
    """
    Process a single h5ad file and return sparse COO data.
    Returns dict with cell_indices, gene_indices, values, n_cells
    """
    file_path, target_genes, min_genes, target_sum = args

    try:
        adata = sc.read_h5ad(file_path)

        # Fix gene names
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

        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X

        # Create gene name to index mapping for target genes
        target_gene_idx = {g: i for i, g in enumerate(target_genes)}
        source_gene_idx = {g: i for i, g in enumerate(adata.var_names)}

        # Find overlapping genes
        overlap_genes = set(adata.var_names) & set(target_genes)

        # Collect sparse data (COO format)
        cell_indices = []
        gene_indices = []
        values = []

        n_cells = X.shape[0]

        for cell_idx in range(n_cells):
            row = X[cell_idx]
            # Only keep non-zero values for overlapping genes
            for gene_name in overlap_genes:
                src_idx = source_gene_idx[gene_name]
                val = row[src_idx]
                if val != 0:
                    tgt_idx = target_gene_idx[gene_name]
                    cell_indices.append(cell_idx)  # Local cell index, will be remapped
                    gene_indices.append(tgt_idx)
                    values.append(float(val))

        return {
            'n_cells': n_cells,
            'cell_indices': np.array(cell_indices, dtype=np.int64),
            'gene_indices': np.array(gene_indices, dtype=np.int64),
            'values': np.array(values, dtype=np.float32),
            'file_path': str(file_path),
        }

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def create_tiledb_dataset(
    output_dir: Path,
    split_name: str,
    all_data: List[Dict],
    target_genes: List[str],
) -> Dict:
    """
    Create TileDB dataset from collected sparse data.

    Structure (CellArr-style):
    - counts/: Sparse matrix (cell_index, gene_index) -> value
    - cell_metadata/: Cell metadata
    - gene_annotation/: Gene info
    - metadata.json
    """
    tiledb_dir = output_dir / f"{split_name}_tiledb"
    if tiledb_dir.exists():
        import shutil
        shutil.rmtree(tiledb_dir)
    tiledb_dir.mkdir(parents=True)

    n_genes = len(target_genes)

    # Concatenate all data and remap cell indices
    logger.info(f"  Concatenating {len(all_data)} files...")

    all_cell_indices = []
    all_gene_indices = []
    all_values = []
    cell_offset = 0
    total_cells = 0

    for data in tqdm(all_data, desc=f"  Merging {split_name}"):
        n_cells = data['n_cells']
        total_cells += n_cells

        if len(data['cell_indices']) > 0:
            # Remap cell indices to global
            remapped_cells = data['cell_indices'] + cell_offset
            all_cell_indices.append(remapped_cells)
            all_gene_indices.append(data['gene_indices'])
            all_values.append(data['values'])

        cell_offset += n_cells

    if all_cell_indices:
        all_cell_indices = np.concatenate(all_cell_indices)
        all_gene_indices = np.concatenate(all_gene_indices)
        all_values = np.concatenate(all_values)
    else:
        all_cell_indices = np.array([], dtype=np.int64)
        all_gene_indices = np.array([], dtype=np.int64)
        all_values = np.array([], dtype=np.float32)

    nnz = len(all_values)
    sparsity = 1.0 - (nnz / (total_cells * n_genes)) if total_cells * n_genes > 0 else 0

    logger.info(f"  Total cells: {total_cells:,}, Genes: {n_genes:,}")
    logger.info(f"  Non-zero elements: {nnz:,}, Sparsity: {sparsity:.2%}")

    # Create counts TileDB (sparse matrix)
    counts_uri = str(tiledb_dir / "counts")

    dom = tiledb.Domain(
        tiledb.Dim(name="cell_index", domain=(0, max(total_cells - 1, 0)),
                   tile=min(10000, max(total_cells, 1)), dtype=np.int64),
        tiledb.Dim(name="gene_index", domain=(0, n_genes - 1),
                   tile=n_genes, dtype=np.int64),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=True,
        attrs=[tiledb.Attr(name="data", dtype=np.float32)],
        allows_duplicates=False,
    )

    tiledb.Array.create(counts_uri, schema)

    # Write data in chunks
    if nnz > 0:
        chunk_size = 10000000  # 10M elements per chunk
        with tiledb.open(counts_uri, 'w') as arr:
            for i in tqdm(range(0, nnz, chunk_size), desc=f"  Writing counts"):
                end_idx = min(i + chunk_size, nnz)
                arr[all_cell_indices[i:end_idx], all_gene_indices[i:end_idx]] = all_values[i:end_idx]

    logger.info(f"  ✓ Created counts TileDB")

    # Free memory
    del all_cell_indices, all_gene_indices, all_values
    gc.collect()

    # Create cell_metadata TileDB
    cell_metadata_uri = str(tiledb_dir / "cell_metadata")

    dom = tiledb.Domain(
        tiledb.Dim(name="cell_index", domain=(0, max(total_cells - 1, 0)),
                   tile=min(10000, max(total_cells, 1)), dtype=np.int64),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        attrs=[
            tiledb.Attr(name="split", dtype='ascii', var=True),
        ],
    )
    tiledb.Array.create(cell_metadata_uri, schema)

    if total_cells > 0:
        with tiledb.open(cell_metadata_uri, 'w') as arr:
            arr[:] = {'split': np.array([split_name] * total_cells, dtype=object)}

    logger.info(f"  ✓ Created cell_metadata TileDB")

    # Create gene_annotation TileDB
    gene_annotation_uri = str(tiledb_dir / "gene_annotation")

    dom = tiledb.Domain(
        tiledb.Dim(name="gene_index", domain=(0, n_genes - 1),
                   tile=n_genes, dtype=np.int64),
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

    logger.info(f"  ✓ Created gene_annotation TileDB")

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

    logger.info(f"  ✓ Metadata saved")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Preprocess h5ad files directly to TileDB format")
    parser.add_argument("--csv_path", type=str, default="data_info/ae_data_info.csv",
                        help="CSV file with file_path and full_validation_dataset columns")
    parser.add_argument("--vocab_path", type=str, default="data_info/gene_order.tsv",
                        help="Gene vocabulary file (one gene per line)")
    parser.add_argument("--output_dir", type=str, default="data/ae_tiledb",
                        help="Output directory for TileDB datasets")
    parser.add_argument("--min_genes", type=int, default=200,
                        help="Minimum genes per cell")
    parser.add_argument("--target_sum", type=float, default=1e4,
                        help="Target sum for normalization")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42)

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

    # Prepare tasks
    tasks = []
    for idx, row in info_df.iterrows():
        file_path = row['file_path']
        is_ood = row.get('full_validation_dataset', 0) == 1

        if Path(file_path).exists():
            tasks.append({
                'file_path': file_path,
                'is_ood': is_ood,
            })

    logger.info(f"Processing {len(tasks)} files with {args.num_workers} workers")
    logger.info("=" * 80)

    # Process all files in parallel
    logger.info("Step 1: Processing h5ad files...")

    process_args = [
        (t['file_path'], target_genes, args.min_genes, args.target_sum)
        for t in tasks
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_h5ad_file, arg): i for i, arg in enumerate(process_args)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing h5ad"):
            result = future.result()
            if result:
                idx = futures[future]
                result['is_ood'] = tasks[idx]['is_ood']
                results.append(result)

    logger.info(f"Successfully processed {len(results)} files")

    # Split results into train/val/test/ood
    logger.info("=" * 80)
    logger.info("Step 2: Splitting data...")

    ood_data = [r for r in results if r['is_ood']]
    other_data = [r for r in results if not r['is_ood']]

    # Shuffle and split non-OOD data
    np.random.shuffle(other_data)
    n_total = len(other_data)
    n_train = int(n_total * 0.90)
    n_val = int(n_total * 0.05)

    train_data = other_data[:n_train]
    val_data = other_data[n_train:n_train + n_val]
    test_data = other_data[n_train + n_val:]

    logger.info(f"  Train: {len(train_data)} files")
    logger.info(f"  Val: {len(val_data)} files")
    logger.info(f"  Test: {len(test_data)} files")
    logger.info(f"  OOD: {len(ood_data)} files")

    # Create TileDB datasets
    logger.info("=" * 80)
    logger.info("Step 3: Creating TileDB datasets...")

    stats = {}

    for split_name, split_data in [('train', train_data), ('val', val_data),
                                     ('test', test_data), ('ood', ood_data)]:
        if split_data:
            logger.info(f"\nCreating {split_name}_tiledb...")
            metadata = create_tiledb_dataset(output_dir, split_name, split_data, target_genes)
            stats[split_name] = metadata

    # Save global metadata
    global_metadata = {
        'n_genes': len(target_genes),
        'splits': stats,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(global_metadata, f, indent=2)

    # Summary
    logger.info("=" * 80)
    logger.info("Preprocessing COMPLETE!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    for split_name, meta in stats.items():
        logger.info(f"  {split_name}_tiledb/: {meta['n_cells']:,} cells, "
                   f"{meta['nnz']:,} non-zeros, {meta['sparsity']:.2%} sparsity")

    logger.info("")
    logger.info("To train with TileDB data:")
    logger.info(f"  python train_ae.py \\")
    logger.info(f"      --data.dataset_type=tiledb \\")
    logger.info(f"      --data.processed_path.train={output_dir}/train_tiledb \\")
    logger.info(f"      --data.processed_path.val={output_dir}/val_tiledb \\")
    logger.info(f"      --data.processed_path.ood={output_dir}/ood_tiledb")
    logger.info("=" * 80)


if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    main()
