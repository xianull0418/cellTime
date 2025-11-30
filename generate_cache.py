# generate_cache.py
import json
from pathlib import Path
import zarr # or import pyarrow.parquet as pq

# Update path as needed
base_dir = Path("/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens/.zarr")
splits = ["train_shards", "val_shards", "test_shards", "ood_shards"]

for split in splits:
    d = base_dir / split
    if not d.exists(): continue
    
    files = sorted(list(d.glob("*.zarr")))
    print(f"Scanning {split}: {len(files)} files...")
    
    total_cells = 0
    # Fast scan since we just need shape
    for f in files:
        try:
            # For Zarr
            store = zarr.open_group(str(f), mode='r')
            if "X" in store:
                 if isinstance(store["X"], zarr.hierarchy.Group): # sparse
                     total_cells += store["X"].attrs["shape"][0]
                 else: # dense
                     total_cells += store["X"].shape[0]
        except Exception as e:
            print(f"Error {f}: {e}")
            
    print(f"  Total cells: {total_cells}")
    
    cache = {
        'n_files': len(files),
        'files_hash': len(files),
        'n_cells': total_cells,
        'shard_lengths': [],
        'shard_offsets': []
    }
    
    with open(d / "metadata_cache_zarr.json", 'w') as f:
        json.dump(cache, f)
    print("  Saved cache.")