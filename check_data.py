
import argparse
from pathlib import Path
import json
import pyarrow.parquet as pq
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def scan_dataset(data_dir, num_workers=128):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory {data_path} not found.")
        return

    print(f"Scanning directory: {data_path}")
    files = sorted(list(data_path.glob("*.parquet")))
    print(f"Found {len(files)} parquet files.")

    if len(files) == 0:
        return

    def get_rows(f):
        try:
            return pq.read_metadata(f).num_rows
        except Exception as e:
            print(f"Error reading {f}: {e}")
            return 0

    print("Reading metadata from files...")
    total_cells = 0
    shard_lengths = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(get_rows, files), total=len(files)))
    
    total_cells = sum(results)
    shard_lengths = results

    print(f"Total cells found: {total_cells}")
    
    # Check existing cache
    cache_file = data_path / "metadata_cache.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"Existing cache n_cells: {cache.get('n_cells', 'Unknown')}")
        print(f"Existing cache n_files: {cache.get('n_files', 'Unknown')}")
    else:
        print("No metadata_cache.json found.")

    # Update cache option
    print("Updating cache with scanned values...")
    cache = {
        'n_files': len(files),
        'files_hash': len(files),
        'n_cells': total_cells,
        'shard_lengths': shard_lengths,
        'shard_offsets': [] # Optional, can be recomputed
    }
    
    # Recompute offsets
    cumulative = 0
    offsets = []
    for length in shard_lengths:
        offsets.append(cumulative)
        cumulative += length
    cache['shard_offsets'] = offsets

    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    print(f"Cache updated at {cache_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to shards directory (e.g., .../train_shards)")
    args = parser.parse_args()
    
    scan_dataset(args.path)

