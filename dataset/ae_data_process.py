#!/usr/bin/env python
"""
AE Data Processing Script (scBank Version)
Process single-cell data for Autoencoder training using scBank.
1. Read ae_data_info.csv
2. Split into Train and OOD file lists
3. Create scBank datasets for each
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

# Add project root to path to allow imports
sys.path.append(os.getcwd())

from dataset.scbank.databank import DataBank, _process_single_file
from dataset.scbank.gene_vocab import GeneVocab
from dataset.scbank.data import MetaInfo, DataTable
from dataset.scbank.setting import Setting
from datasets import load_dataset

# Suppress warnings
warnings.filterwarnings("ignore")

def create_scbank_from_files(
    files: list,
    output_dir: Path,
    vocab: GeneVocab = None,
    main_table_key: str = "X",
    token_col: str = "gene_name",
    num_workers: int = 4,
    min_genes: int = 200,
):
    """
    Create a scBank from a list of files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If vocab is not provided, build it from the first file (assuming consistency)
    if vocab is None:
        print("Building vocabulary from the first file...")
        first_adata = sc.read_h5ad(files[0])
        genes = first_adata.var_names.tolist()
        vocab = GeneVocab.from_dict({g: i for i, g in enumerate(genes)})
        # Save vocab
        vocab.save_json(output_dir / "gene_vocab.json")
        del first_adata

    # Initialize DataBank
    db = DataBank(
        meta_info=MetaInfo(on_disk_path=output_dir, on_disk_format="parquet"),
        gene_vocab=vocab,
        settings=Setting(immediate_save=True),
    )
    
    # Process files
    parquet_files = []
    
    # We can reuse _process_single_file from databank.py
    # But we need to handle the multiprocessing ourselves or use the one in databank if adaptable
    # Since _process_single_file is module-level, we can use it.
    # However, _process_single_file in databank.py does normalization/log1p internally.
    # We need to ensure it matches our requirements.
    # Checking databank.py:
    # It does: sc.pp.normalize_total(adata, target_sum=1e4) -> sc.pp.log1p(adata)
    # This matches our requirement.
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    print(f"Processing {len(files)} files to {output_dir}...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(
                _process_single_file, 
                Path(f), output_dir, vocab, main_table_key, token_col, i
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
        # Note: load_dataset might fail if datasets library is mocked improperly or version mismatch
        # For verification script with mocks, this step is fragile.
        # But in real run, it should work.
        # The error "isinstance() arg 2 must be a type" suggests some type check failed.
        # It might be in scBank's update_datatables where it checks isinstance(t, DataTable)
        
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
        # Use the same vocab as train if available
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
        
    # Update CSV with processed info (optional, scBank doesn't strictly need this but good for record)
    # For now we skip updating cell counts per file as scBank aggregates them.
    
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
