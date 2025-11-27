#!/usr/bin/env python
"""
AE Data Processing Script (scBank Version)
Process single-cell data for Autoencoder training using scBank.
1. Read ae_data_info.csv
2. Split into Train and OOD file lists
3. Create scBank datasets for each
4. Performs QC (min_genes), Gene Mapping (align to gene_order.tsv), Normalization, Log1p
"""

import os
import argparse
import pandas as pd
import scanpy as sc
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path to allow imports
sys.path.append(os.getcwd())

from dataset.scbank.databank import DataBank, _map_ind
from dataset.scbank.gene_vocab import GeneVocab
from dataset.scbank.data import MetaInfo, DataTable
from dataset.scbank.setting import Setting
from datasets import load_dataset, Dataset, disable_progress_bar

# Suppress warnings
warnings.filterwarnings("ignore")

def process_single_file_custom(
    file_path: Path, 
    to_path: Path, 
    vocab: GeneVocab, 
    main_table_key: str, 
    index: int,
    min_genes: int = 200
):
    """
    Custom worker to process a single h5ad file.
    Performs: QC -> Gene Mapping -> Norm -> Log1p -> Tokenize -> Save Parquet
    """
    disable_progress_bar()
    
    try:
        # 1. Read Data
        adata = sc.read_h5ad(file_path)
        
        # 2. QC: Filter cells
        if min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)
            if adata.n_obs == 0:
                print(f"Warning: {file_path.name} has 0 cells after QC.")
                return None

        # 3. Gene Mapping Setup
        # We need to map adata.var_names to vocab indices
        # scBank's _map_ind does this: maps existing token -> new index
        # Extra genes in adata are ignored (not in vocab -> not in ind2ind)
        # Missing genes in adata are just not present in the sparse output (implicitly 0)
        
        # Ensure we use the correct column for gene names (usually index)
        tokens = adata.var_names.tolist()
        ind2ind = _map_ind(tokens, vocab)
        
        # 4. Normalization & Log1p
        try:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        except Exception as e:
            print(f"Warning: Preprocessing failed for {file_path.name}: {e}")

        # 5. Tokenize
        # Use a dummy DataBank to access _tokenize method
        # We use a dummy DataBank to reuse the robust tokenization logic (handling sparse/dense/numba)
        db_dummy = DataBank(meta_info=MetaInfo(), gene_vocab=vocab)
        
        data_key = main_table_key
        if data_key == "X":
            data = adata.X
        elif data_key in adata.layers:
            data = adata.layers[data_key]
        else:
            data = adata.X

        tokenized_data = db_dummy._tokenize(data, ind2ind)
        
        # 6. Save to Parquet
        ds = Dataset.from_dict(tokenized_data)
        shard_path = to_path / f"shard_{index}.parquet"
        ds.to_parquet(shard_path)
        
        # Clean up
        del adata
        del data
        del tokenized_data
        del ds
        
        return str(shard_path)

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_scbank_from_files(
    files: list,
    output_dir: Path,
    vocab: GeneVocab = None,
    main_table_key: str = "X",
    num_workers: int = 4,
    min_genes: int = 200,
):
    """
    Create a scBank from a list of files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vocab from gene_order.tsv if not provided
    if vocab is None:
        gene_order_path = Path("gene_order.tsv")
        if gene_order_path.exists():
            print(f"Loading vocabulary from {gene_order_path}...")
            vocab = GeneVocab.from_file(gene_order_path)
        else:
            print("Warning: gene_order.tsv not found. Building vocabulary from the first file...")
            first_adata = sc.read_h5ad(files[0])
            genes = first_adata.var_names.tolist()
            vocab = GeneVocab.from_dict({g: i for i, g in enumerate(genes)})
            del first_adata
            
    # Save vocab to scBank
    vocab.save_json(output_dir / "gene_vocab.json")

    # Initialize DataBank (MetaInfo only)
    db = DataBank(
        meta_info=MetaInfo(on_disk_path=output_dir, on_disk_format="parquet"),
        gene_vocab=vocab,
        settings=Setting(immediate_save=True),
    )
    
    # Process files
    parquet_files = []
    
    print(f"Processing {len(files)} files to {output_dir} with {num_workers} workers...")
    print(f"  - QC: min_genes={min_genes}")
    print(f"  - Gene Mapping: Aligned to {len(vocab)} genes")
    print(f"  - Norm: 1e4")
    print(f"  - Log1p: Yes")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(
                process_single_file_custom, 
                Path(f), output_dir, vocab, main_table_key, i, min_genes
            ): f for i, f in enumerate(files)
        }
        
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="Converting"):
            f = future_to_file[future]
            try:
                result = future.result()
                if result:
                    parquet_files.append(result)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    if not parquet_files:
        print("Warning: No files processed successfully.")
        return

    # Load all parquet files as a single Dataset
    print(f"Loading {len(parquet_files)} shards as a single dataset...")
    try:
        full_ds = load_dataset("parquet", data_files=parquet_files, split="train", cache_dir=str(output_dir))
        
        data_table = DataTable(
            name=main_table_key,
            data=full_ds,
        )

        db.main_table_key = main_table_key
        db.update_datatables(new_tables=[data_table], immediate_save=True)
        print(f"scBank created at {output_dir}")
        
    except Exception as e:
        print(f"Error creating final dataset: {e}")
        import traceback
        traceback.print_exc()

def process_data(
    csv_path: str,
    output_dir: str,
    min_genes: int = 200,
    num_workers: int = 4,
):
    """
    Process data defined in CSV and save as scBank datasets.
    """
    output_path = Path(output_dir)
    train_dir = output_path / "train.scbank"
    ood_dir = output_path / "ood.scbank"
    
    # Read CSV
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter valid files
    valid_df = df[df["file_path"].apply(os.path.exists)]
    if len(valid_df) < len(df):
        print(f"Warning: {len(df) - len(valid_df)} files not found.")
    
    # Split Train/OOD
    # full_validation_dataset == 1 -> OOD
    ood_files = valid_df[valid_df["full_validation_dataset"] == 1]["file_path"].tolist()
    train_files = valid_df[valid_df["full_validation_dataset"] != 1]["file_path"].tolist()
    
    print(f"Found {len(train_files)} training files and {len(ood_files)} OOD files.")
    
    # Create Train scBank
    if train_files:
        print("\n=== Creating Train scBank ===")
        create_scbank_from_files(
            train_files, 
            train_dir, 
            num_workers=num_workers,
            min_genes=min_genes
        )
        
    # Create OOD scBank
    if ood_files:
        print("\n=== Creating OOD scBank ===")
        # Use the same vocab as train if available (which should be gene_order.tsv)
        vocab = None
        if (train_dir / "gene_vocab.json").exists():
            vocab = GeneVocab.from_file(train_dir / "gene_vocab.json")
            
        create_scbank_from_files(
            ood_files, 
            ood_dir, 
            vocab=vocab,
            num_workers=num_workers,
            min_genes=min_genes
        )
    
    print("\nProcessing Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AE Data Processing (scBank)")
    parser.add_argument("--csv_path", type=str, default="ae_data_info.csv", help="Path to data info CSV")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--min_genes", type=int, default=200, help="Minimum genes for cell filtering")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    
    process_data(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        min_genes=args.min_genes,
        num_workers=args.num_workers
    )
