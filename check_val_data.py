import argparse
from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm
import torch
from dataset.cell_dataset import ParquetIterableDataset

def check_val_dataset(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory {data_path} not found.")
        return

    print(f"Scanning validation directory: {data_path}")
    files = sorted(list(data_path.glob("*.parquet")))
    print(f"Found {len(files)} parquet files.")

    if len(files) == 0:
        print("No parquet files found!")
        return

    print("Attempting to read first few files...")
    for i, f in enumerate(files[:3]):
        try:
            table = pq.read_table(f)
            print(f"File {f.name}: {table.num_rows} rows, {table.num_columns} columns")
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    print("\nTesting ParquetIterableDataset loading...")
    try:
        dataset = ParquetIterableDataset(data_path, verbose=True, shuffle_shards=False, shuffle_rows=False)
        print(f"Dataset initialized. Estimated n_cells: {dataset.n_cells}")
        
        print("Iterating through first 10 items...")
        count = 0
        for batch in dataset:
            count += 1
            if count >= 10:
                break
            if count == 1:
                print(f"Sample shape: {batch.shape}")
        print("Iteration successful.")
        
    except Exception as e:
        print(f"Dataset iteration failed: {e}")

if __name__ == "__main__":
    # Hardcoded path based on previous context or argument
    # Using the path from your script
    val_path = "/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens/.parquet/val_shards"
    check_val_dataset(val_path)

