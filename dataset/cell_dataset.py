"""
单细胞数据集
支持从 Parquet (推荐) 或 AnnData 加载数据
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
import scanpy as sc
from pathlib import Path
from typing import Union, Optional, List, Dict
import json
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import math
import logging
import gc

try:
    import zarr
except ImportError:
    zarr = None

try:
    import anndata as ad
except ImportError:
    ad = None


def load_anndata(
    data: Union[str, Path, "ad.AnnData"],
    *,
    index_col: int = 0,
    verbose: bool = False,
    max_genes: Optional[int] = None,
    select_hvg: bool = True,
    target_genes: Optional[List[str]] = None,
) -> sc.AnnData:
    """
    加载 AnnData 对象
    """
    if ad is not None and isinstance(data, ad.AnnData):
        adata = data.copy()
    else:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        
        if verbose:
            print(f"从文件加载数据: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".h5ad":
            adata = sc.read_h5ad(path)
        elif suffix in {".h5", ".hdf5"}:
            adata = sc.read(path)
        elif suffix in {".csv", ".xlsx", ".xls"}:
            if suffix == ".csv":
                df = pd.read_csv(path, index_col=index_col)
            else:
                df = pd.read_excel(path, index_col=index_col)
            
            adata = sc.AnnData(
                X=df.to_numpy(),
                obs=pd.DataFrame(index=df.index),
                var=pd.DataFrame(index=df.columns),
            )
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
        
        if verbose:
            print(f"数据加载完成，维度: {adata.shape}")
    
    # 基因映射（优先级高于 max_genes）
    if target_genes is not None:
        if verbose:
            print(f"正在将数据映射到目标基因列表 ({len(target_genes)} 个基因)...")
        
        adata.var_names_make_unique()
        
        # 检查重叠
        overlap = len(set(adata.var_names) & set(target_genes))
        if verbose:
            print(f"  - 共有基因数: {overlap}")
        
        # 创建一个新的 AnnData
        if hasattr(adata.X, "toarray"):
            X_mat = adata.X.toarray()
        else:
            X_mat = adata.X
            
        X_df = pd.DataFrame(X_mat, index=adata.obs_names, columns=adata.var_names)
        X_new = X_df.reindex(columns=target_genes, fill_value=0.0)
        
        new_adata = sc.AnnData(
            X=X_new.values.astype(np.float32),
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=target_genes)
        )
        adata = new_adata

    # 基因数检查和高变基因选择
    elif max_genes is not None and adata.shape[1] > max_genes and select_hvg:
        if verbose:
            print(f"基因数 ({adata.shape[1]}) 超过上限 ({max_genes})，自动选择高变基因...")
            
        # Fix for potential KeyError: 'base' in adata.uns['log1p'] caused by some scanpy versions
        if 'log1p' in adata.uns and isinstance(adata.uns['log1p'], dict) and 'base' not in adata.uns['log1p']:
            adata.uns['log1p']['base'] = None
            
        sc.pp.highly_variable_genes(adata, n_top_genes=max_genes, subset=True)
    
    return adata


class ParquetDataset(Dataset):
    """
    高效加载预处理好的 Parquet 文件 (支持单个文件或分片目录)
    """
    def __init__(self, path: Union[str, Path], verbose: bool = False):
        self.path = Path(path)
        self.files = []
        
        if self.path.is_dir():
            # Load all parquet files in directory
            self.files = sorted(list(self.path.glob("*.parquet")))
            if not self.files:
                raise FileNotFoundError(f"No .parquet files found in directory: {self.path}")
            if verbose:
                print(f"Loading Parquet dataset from directory: {self.path} ({len(self.files)} shards)")
        elif self.path.is_file():
            self.files = [self.path]
            if verbose:
                print(f"Loading Parquet dataset: {self.path}")
        else:
             raise FileNotFoundError(f"Path not found: {self.path}")

        # Read metadata to get total length and valid gene columns
        # Optimization: Read schema from first file, assume consistent schema
        try:
            import pyarrow.parquet as pq
            self.shard_offsets = []
            self.shard_lengths = []
            total_len = 0
            
            # Use first file to get gene names/dim
            # Use read_table with limit to accurately determine columns (excluding index via to_pandas)
            # meta0 = pq.read_metadata(self.files[0]) # This can include index columns
            sample = pq.read_table(self.files[0]).slice(0, 1) # Slice is cheap
            self.n_genes = sample.to_pandas().shape[1]
            
            # Cache check
            cache_file = self.path / "metadata_cache.json" if self.path.is_dir() else None
            
            if cache_file and cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                    # Verify cache is valid (same file count)
                    if cache.get('n_files') == len(self.files) and cache.get('files_hash') == len(self.files): # Simple check
                        if verbose:
                            print("Loading metadata from cache...")
                        self.shard_lengths = cache['shard_lengths']
                        self.shard_offsets = cache['shard_offsets']
                        self.n_cells = cache['n_cells']
                        # Ensure types
                        self.shard_offsets = np.array(self.shard_offsets)
                        self.shard_lengths = np.array(self.shard_lengths)
                        
                        if verbose:
                            print(f"  - Total Cells: {self.n_cells}")
                            print(f"  - Genes: {self.n_genes}")
                        return
                except Exception as e:
                    if verbose:
                        print(f"Cache load failed, reloading: {e}")

            # Parallel metadata reading
            def get_meta(f):
                return pq.read_metadata(f).num_rows

            if verbose:
                print("Reading metadata from shards (parallel)...")
            
            with ThreadPoolExecutor(max_workers=32) as executor:
                results = list(tqdm(executor.map(get_meta, self.files), total=len(self.files), disable=not verbose))
            
            self.shard_lengths = results
            
            # Calculate offsets
            cumulative = 0
            for n_rows in self.shard_lengths:
                self.shard_offsets.append(cumulative)
                cumulative += n_rows
                
            total_len = cumulative
                
            self.n_cells = total_len
            self.shard_offsets = np.array(self.shard_offsets)
            self.shard_lengths = np.array(self.shard_lengths)
            
            # Save cache
            if cache_file:
                try:
                    cache = {
                        'n_files': len(self.files),
                        'files_hash': len(self.files), # Ideally use a timestamp/hash, simple count for now
                        'n_cells': self.n_cells,
                        'shard_lengths': self.shard_lengths.tolist(),
                        'shard_offsets': self.shard_offsets.tolist()
                    }
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f)
                    if verbose:
                        print("Metadata cache saved.")
                except Exception as e:
                    if verbose: print(f"Failed to save cache: {e}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ParquetDataset: {e}")
            
        if verbose:
            print(f"  - Total Cells: {self.n_cells}")
            print(f"  - Genes: {self.n_genes}")

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Find which shard contains the index
        # Binary search or simple loop (loop is fast enough for small N_shards)
        shard_idx = np.searchsorted(self.shard_offsets, idx, side='right') - 1
        
        if shard_idx < 0 or shard_idx >= len(self.files):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.n_cells}")
            
        local_idx = idx - self.shard_offsets[shard_idx]
        
        # Read specific row from specific file
        # This reads the whole row group containing the row, so it might be slow if row groups are large
        # Optimization: Keep file handle open or use memory mapping if possible? 
        # Pandas read_parquet is simple but reads whole file or columns.
        # PyArrow allows reading specific rows.
        
        # Optimized read:
        import pyarrow.parquet as pq
        # Reading a single row is inefficient in Parquet.
        # Ideally we use a DataLoader with BatchSampler that reads chunks.
        # For now, to keep API compatible, we read the row.
        # Performance Warning: This might be slow for random access training.
        
        # Alternative: Load shard into memory if it's small? 
        # With 15,000 files for 90M cells, avg shard is ~6000 cells. Small enough to cache?
        # Let's use a simple LRU cache for open file/table if memory permits.
        # For simplicity and reliability first implementation:
        
        fpath = self.files[shard_idx]
        # Use pyarrow to read specific row (batch of size 1)
        # table = pq.read_table(fpath) # Reading whole shard (approx 6k cells) is fast enough
        # row = table.slice(local_idx, length=1)
        
        # Optimization: Read only necessary row group?
        # Or better: Use Zarr which supports chunked reads natively and much faster for random access.
        # For Parquet, random access is inherently slow.
        
        # Optimized read:
        import pyarrow.parquet as pq
        
        # To optimize random access in Parquet, we should ideally cache the open file or table.
        # But given the scale, caching thousands of tables is memory intensive.
        # Let's try to read only the row we need more efficiently if possible.
        
        # If dataset is large, random access on parquet is a bottleneck.
        # Suggest converting to Zarr or Arrow (IPC) format for random access.
        
        pf = pq.ParquetFile(fpath)
        # Map global row index to row group and row index within group
        # Simplified: read the whole table. It's small enough (6k rows).
        table = pf.read() 
        row = table.slice(local_idx, length=1)

        # Convert to tensor
        # Use to_pandas() or direct buffer access
        x = torch.from_numpy(row.to_pandas().values[0]).float()
        return x


class ParquetIterableDataset(IterableDataset):
    """
    Iterative Parquet Dataset for efficient training on large sharded datasets.
    Avoids random access overhead by reading full shards and shuffling in memory.

    DDP Support:
    - Files are split among ranks (GPUs) using round-robin
    - Each rank's files are further split among DataLoader workers
    - Use `equal_length=True` to ensure all ranks produce the same number of samples
      (required to avoid DDP deadlocks when file sizes are uneven)
    """
    def __init__(self, path: Union[str, Path], verbose: bool = False, shuffle_shards: bool = True,
                 shuffle_rows: bool = True, debug: bool = False, equal_length: bool = True):
        self.path = Path(path)
        self.verbose = verbose
        self.shuffle_shards = shuffle_shards
        self.shuffle_rows = shuffle_rows
        self.debug = debug
        self.equal_length = equal_length  # Ensure all ranks produce equal samples for DDP
        self.files = []
        self.shard_lengths = None  # Will be populated from cache if available

        if self.path.is_dir():
            self.files = sorted(list(self.path.glob("*.parquet")))
            if not self.files:
                raise FileNotFoundError(f"No .parquet files found in directory: {self.path}")
            if verbose:
                print(f"Loading Parquet iterable dataset: {self.path} ({len(self.files)} shards)")
        elif self.path.is_file():
            self.files = [self.path]
        else:
             raise FileNotFoundError(f"Path not found: {self.path}")

        # Read metadata from first file to get n_genes
        try:
            import pyarrow.parquet as pq
            sample = pq.read_table(self.files[0]).slice(0, 1)
            self.n_genes = sample.to_pandas().shape[1]
            self.n_cells = 0

            # Cache check for n_cells and shard_lengths (critical for equal_length mode)
            cache_file = self.path / "metadata_cache.json" if self.path.is_dir() else None
            if cache_file and cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                    if cache.get('n_files') == len(self.files):
                        self.n_cells = cache['n_cells']
                        self.shard_lengths = cache.get('shard_lengths', None)
                        if self.shard_lengths:
                            self.shard_lengths = {str(f): l for f, l in zip(self.files, self.shard_lengths)}
                except: pass

        except Exception as e:
            raise RuntimeError(f"Failed to initialize ParquetIterableDataset: {e}")

    def _compute_target_samples_per_rank(self, world_size: int, num_workers: int) -> int:
        """
        Compute the target number of samples each rank should produce for equal_length mode.
        This ensures all ranks produce exactly the same number of samples to avoid DDP deadlock.
        """
        if self.n_cells <= 0:
            return 0

        # Each rank should produce ceil(n_cells / world_size) samples
        # This ensures we don't lose data, and all ranks produce the same amount
        samples_per_rank = int(math.ceil(self.n_cells / world_size))
        return samples_per_rank

    def __iter__(self):
        import torch.distributed as dist
        import pyarrow.parquet as pq

        # 1. Global Split (per Rank/GPU)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # 2. Local Split (per Worker)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # Compute target samples for equal_length mode
        target_samples_per_rank = self._compute_target_samples_per_rank(world_size, num_workers) if self.equal_length else 0
        target_samples_per_worker = int(math.ceil(target_samples_per_rank / num_workers)) if target_samples_per_rank > 0 else 0

        if self.debug and self.equal_length and target_samples_per_worker > 0:
            logging.debug(f"[Rank {rank} Worker {worker_id}] equal_length mode: target {target_samples_per_worker:,} samples/worker")

        # Special handling for edge cases
        if len(self.files) == 1:
            yield from self._iter_single_file(rank, world_size, worker_id, num_workers, target_samples_per_worker)
            return

        if len(self.files) < world_size:
            if self.debug or self.verbose:
                logging.debug(f"[Rank {rank}] Files ({len(self.files)}) < ranks ({world_size}), using row-based split")
            yield from self._iter_few_files(rank, world_size, worker_id, num_workers, target_samples_per_worker)
            return

        # Multi-file handling: Split files among ranks and workers
        my_files = self.files[rank::world_size]
        if worker_info is not None:
            my_files = my_files[worker_id::num_workers]

        if self.debug or self.verbose:
            logging.debug(f"[Rank {rank}/{world_size} Worker {worker_id}/{num_workers}] "
                         f"Processing {len(my_files)}/{len(self.files)} files from {self.path.name}")

        if self.shuffle_shards:
            my_files = list(my_files)
            np.random.shuffle(my_files)

        total_samples_yielded = 0
        all_samples_buffer = []

        show_progress = (rank == 0 and worker_id == 0)
        file_iter = tqdm(enumerate(my_files), total=len(my_files),
                        desc=f"Loading {self.path.name}",
                        disable=not show_progress,
                        unit="shard")

        # Smaller batch to reduce memory usage
        # 5000 rows × 19331 genes × 4 bytes ≈ 370 MB per batch
        BATCH_SIZE = 5000

        for shard_idx, fpath in file_iter:
            try:
                parquet_file = pq.ParquetFile(fpath)

                for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
                    # Fast path: Arrow -> NumPy directly (skip pandas)
                    # Combine all columns into a single array
                    arrays = [col.to_numpy(zero_copy_only=False) for col in batch.columns]
                    batch_data = np.column_stack(arrays).astype(np.float32, copy=False)
                    n_rows = len(batch_data)

                    # Pre-convert entire batch to tensor (faster than per-row)
                    batch_tensor = torch.from_numpy(batch_data)
                    del batch_data, arrays

                    indices = np.arange(n_rows)
                    if self.shuffle_rows:
                        np.random.shuffle(indices)

                    for idx in indices:
                        sample = batch_tensor[idx]
                        yield sample
                        total_samples_yielded += 1

                        if self.equal_length and target_samples_per_worker > 0:
                            if len(all_samples_buffer) < min(1000, target_samples_per_worker):
                                all_samples_buffer.append(sample.clone())

                        if self.equal_length and target_samples_per_worker > 0 and total_samples_yielded >= target_samples_per_worker:
                            return

                    del batch_tensor

                if show_progress:
                    file_iter.set_postfix(samples=f"{total_samples_yielded:,}")

            except Exception as e:
                logging.error(f"[Rank {rank} Worker {worker_id}] Error reading shard {fpath}: {e}")
                continue

        # Padding for equal_length mode
        if self.equal_length and target_samples_per_worker > 0 and total_samples_yielded < target_samples_per_worker:
            if all_samples_buffer:
                buffer_idx = 0
                while total_samples_yielded < target_samples_per_worker:
                    yield all_samples_buffer[buffer_idx % len(all_samples_buffer)]
                    total_samples_yielded += 1
                    buffer_idx += 1

        if self.debug or self.verbose:
            logging.debug(f"[Rank {rank} Worker {worker_id}] COMPLETED iteration, total samples: {total_samples_yielded}")

    def _iter_single_file(self, rank: int, world_size: int, worker_id: int, num_workers: int, target_samples_per_worker: int):
        """Handle iteration for single-file datasets by splitting rows using streaming."""
        import pyarrow.parquet as pq

        show_progress = (rank == 0 and worker_id == 0)

        if show_progress:
            print(f"Loading single file: {self.files[0].name}...")

        parquet_file = pq.ParquetFile(self.files[0])
        total_rows = parquet_file.metadata.num_rows

        # Calculate row range for this rank and worker
        rows_per_rank = total_rows // world_size
        start_idx = rank * rows_per_rank
        end_idx = total_rows if rank == world_size - 1 else start_idx + rows_per_rank

        rank_rows = end_idx - start_idx
        rows_per_worker = rank_rows // num_workers if num_workers > 1 else rank_rows
        worker_start = start_idx + worker_id * rows_per_worker
        worker_end = end_idx if worker_id == num_workers - 1 else worker_start + rows_per_worker

        if self.debug or self.verbose:
            logging.debug(f"[Rank {rank} Worker {worker_id}] Processing rows {worker_start:,}-{worker_end:,}")

        total_samples_yielded = 0
        current_row = 0
        BATCH_SIZE = 5000  # ~370 MB per batch

        row_iter = tqdm(range(worker_end - worker_start),
                       desc=f"Processing {self.path.name}",
                       disable=not show_progress,
                       unit="sample")

        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
            batch_start = current_row
            batch_len = batch.num_rows
            batch_end = current_row + batch_len

            if batch_end <= worker_start:
                current_row = batch_end
                continue
            if batch_start >= worker_end:
                break

            # Fast Arrow -> NumPy -> Tensor conversion
            arrays = [col.to_numpy(zero_copy_only=False) for col in batch.columns]
            batch_data = np.column_stack(arrays).astype(np.float32, copy=False)
            batch_tensor = torch.from_numpy(batch_data)
            del batch_data, arrays

            # Calculate overlap
            local_start = max(0, worker_start - batch_start)
            local_end = min(batch_len, worker_end - batch_start)

            indices = np.arange(local_start, local_end)
            if self.shuffle_rows:
                np.random.shuffle(indices)

            for idx in indices:
                yield batch_tensor[idx]
                total_samples_yielded += 1
                row_iter.update(1)

                if self.equal_length and target_samples_per_worker > 0 and total_samples_yielded >= target_samples_per_worker:
                    del batch_tensor
                    row_iter.close()
                    return

            current_row = batch_end
            del batch_tensor

        row_iter.close()

        if self.debug or self.verbose:
            logging.debug(f"[Rank {rank} Worker {worker_id}] COMPLETED iteration, total samples: {total_samples_yielded:,}")

    def _iter_few_files(self, rank: int, world_size: int, worker_id: int, num_workers: int, target_samples_per_worker: int):
        """
        Handle iteration when files < ranks by streaming through all files and splitting by rows.
        Memory efficient: processes one batch at a time.
        """
        import pyarrow.parquet as pq

        show_progress = (rank == 0 and worker_id == 0)

        # First pass: count total rows (lightweight metadata reads)
        total_rows = 0
        file_row_counts = []
        for fpath in self.files:
            try:
                pf = pq.ParquetFile(fpath)
                count = pf.metadata.num_rows
                file_row_counts.append((fpath, count))
                total_rows += count
            except Exception as e:
                logging.error(f"[Rank {rank} Worker {worker_id}] Error reading metadata {fpath}: {e}")

        if total_rows == 0:
            return

        # Calculate row range for this rank and worker
        rows_per_rank = total_rows // world_size
        start_idx = rank * rows_per_rank
        end_idx = total_rows if rank == world_size - 1 else start_idx + rows_per_rank

        rank_rows = end_idx - start_idx
        if rank_rows > 0 and num_workers > 1:
            rows_per_worker = rank_rows // num_workers
            worker_start = start_idx + worker_id * rows_per_worker
            worker_end = end_idx if worker_id == num_workers - 1 else worker_start + rows_per_worker
        else:
            worker_start = start_idx
            worker_end = end_idx

        if self.debug or self.verbose:
            logging.debug(f"[Rank {rank} Worker {worker_id}] Processing rows {worker_start:,}-{worker_end:,}")

        total_samples_yielded = 0
        global_row = 0
        BATCH_SIZE = 5000  # ~370 MB per batch

        row_iter = tqdm(range(worker_end - worker_start),
                       desc=f"Processing {self.path.name}",
                       disable=not show_progress,
                       unit="sample")

        for fpath, file_rows in file_row_counts:
            file_start = global_row
            file_end = global_row + file_rows

            if file_end <= worker_start:
                global_row = file_end
                continue
            if file_start >= worker_end:
                break

            try:
                parquet_file = pq.ParquetFile(fpath)

                for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
                    batch_start = global_row
                    batch_len = batch.num_rows
                    batch_end = global_row + batch_len

                    if batch_end <= worker_start:
                        global_row = batch_end
                        continue
                    if batch_start >= worker_end:
                        break

                    # Fast Arrow -> NumPy -> Tensor conversion
                    arrays = [col.to_numpy(zero_copy_only=False) for col in batch.columns]
                    batch_data = np.column_stack(arrays).astype(np.float32, copy=False)
                    batch_tensor = torch.from_numpy(batch_data)
                    del batch_data, arrays

                    local_start = max(0, worker_start - batch_start)
                    local_end = min(batch_len, worker_end - batch_start)

                    indices = np.arange(local_start, local_end)
                    if self.shuffle_rows:
                        np.random.shuffle(indices)

                    for idx in indices:
                        yield batch_tensor[idx]
                        total_samples_yielded += 1
                        row_iter.update(1)

                        if self.equal_length and target_samples_per_worker > 0 and total_samples_yielded >= target_samples_per_worker:
                            del batch_tensor
                            row_iter.close()
                            return

                    global_row = batch_end
                    del batch_tensor

            except Exception as e:
                logging.error(f"[Rank {rank} Worker {worker_id}] Error streaming {fpath}: {e}")
                global_row = file_end

        row_iter.close()

        if self.debug or self.verbose:
            logging.debug(f"[Rank {rank} Worker {worker_id}] COMPLETED few_files iteration, total samples: {total_samples_yielded:,}")

    # Re-added __len__ for progress bar, but safeguard against 0
    def __len__(self):
        if self.n_cells > 0:
            # Important: When using multi-GPU (DDP), the dataset is split.
            # Each rank sees a subset of files.
            # However, PyTorch Lightning usually expects __len__ to return the *total* length 
            # if using DistributedSampler, but for IterableDataset with custom splitting,
            # it expects the length *on this process*.
            
            # Estimate per-rank length
            import torch.distributed as dist
            world_size = 1
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            
            return int(math.ceil(self.n_cells / world_size))
        return 1000000 # Fallback estimate if unknown to avoid early stopping


class ZarrIterableDataset(IterableDataset):
    """
    Iterative Dataset for efficient training on large Zarr datasets (sharded or consolidated).
    Reads chunks efficiently.
    """
    def __init__(self, path: Union[str, Path], verbose: bool = False, shuffle_shards: bool = True, shuffle_rows: bool = True, debug: bool = False):
        self.path = Path(path)
        self.verbose = verbose
        self.shuffle_shards = shuffle_shards
        self.shuffle_rows = shuffle_rows
        self.debug = debug  # Add debug flag

        if zarr is None:
            raise ImportError("Please install zarr: pip install zarr")

        self.files = []
        if self.path.is_dir():
            # Look for .zarr directories
            self.files = sorted(list(self.path.glob("*.zarr")))
            if not self.files:
                # Fallback: maybe the directory IS the zarr store?
                if (self.path / ".zgroup").exists():
                    self.files = [self.path]
                else:
                    raise FileNotFoundError(f"No .zarr files found in directory: {self.path}")
            if verbose:
                print(f"Loading Zarr iterable dataset: {self.path} ({len(self.files)} shards)")
        elif str(self.path).endswith(".zarr"): # Treat as single zarr
             self.files = [self.path]
        else:
             raise FileNotFoundError(f"Path not found: {self.path}")

        # Metadata read from first file
        try:
            store = zarr.open_group(str(self.files[0]), mode='r')
            # Auto-detect X structure
            if "X" in store:
                if isinstance(store["X"], zarr.hierarchy.Group):
                    # X/data pattern (csc/csr)
                    if "data" in store["X"]:
                        # Estimate n_genes from shape attribute
                        self.n_genes = store["X"].attrs["shape"][1]
                    else:
                        # Fallback if no standard sparse structure
                        raise ValueError("Zarr X group exists but structure unknown (no 'data')")
                else:
                    # X is a direct array (dense)
                    self.n_genes = store["X"].shape[1]
            else:
                # Try to find array directly if not in group 'X' (unlikely for anndata structure)
                # Assuming structure is root -> X
                raise ValueError("Zarr store must contain 'X' array/group")

            self.n_cells = 0 # Unknown unless scanned

            # Cache check
            cache_file = self.path / "metadata_cache_zarr.json" if self.path.is_dir() else None
            if cache_file and cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                    if cache.get('n_files') == len(self.files):
                        self.n_cells = cache['n_cells']
                except: pass

        except Exception as e:
            raise RuntimeError(f"Failed to initialize ZarrIterableDataset: {e}")

    def __iter__(self):
        # DDP and Worker splitting
        import torch.distributed as dist

        # 1. Global Split (per Rank/GPU)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Split files among ranks (Round Robin)
        my_files = self.files[rank::world_size]

        # 2. Local Split (per Worker)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if worker_info is not None:
            my_files = my_files[worker_id::num_workers]

        # DEBUG: Print file distribution
        if self.debug or self.verbose:
            logging.debug(f"[Rank {rank}/{world_size} Worker {worker_id}/{num_workers}] "
                         f"Processing {len(my_files)}/{len(self.files)} Zarr files from {self.path.name}")

        if self.shuffle_shards:
            np.random.shuffle(my_files)

        total_samples_yielded = 0
        total_files = len(self.files)  # Store total file count for consistent logging
        for shard_idx, fpath in enumerate(my_files):
            try:
                if self.debug:
                    logging.debug(f"[Rank {rank} Worker {worker_id}] Reading Zarr shard {shard_idx+1}/{len(my_files)} (assigned), {total_files} total: {fpath.name}")

                store = zarr.open_group(str(fpath), mode='r')
                # Assume X is the data
                X = store["X"]

                # We can read the whole X if shard is small, or chunk by chunk
                # For simplicity and speed (given shards are small ~6k rows), read full X

                # Handle Sparse Zarr (as saved by scimilarity/anndata)
                if isinstance(X, zarr.hierarchy.Group) and "data" in X and "indices" in X and "indptr" in X:
                    from scipy.sparse import csr_matrix
                    data = X["data"][:]
                    indices = X["indices"][:]
                    indptr = X["indptr"][:]
                    shape = X.attrs["shape"]
                    # Reconstruct CSR
                    mat = csr_matrix((data, indices, indptr), shape=shape)
                    data = mat.toarray()
                else:
                    # Dense Zarr
                    # Converting to numpy array loads into memory
                    data = X[:]

                if hasattr(data, "toarray"): # Handle sparse if stored as sparse object (unlikely with standard zarr without wrapper)
                    data = data.toarray()

                data = data.astype(np.float32, copy=False)

                n_rows = len(data)
                indices = np.arange(n_rows)

                if self.shuffle_rows:
                    np.random.shuffle(indices)

                for idx in indices:
                    yield torch.from_numpy(data[idx])
                    total_samples_yielded += 1

                # Cleanup
                del data

                if self.debug:
                    logging.debug(f"[Rank {rank} Worker {worker_id}] Completed Zarr shard {shard_idx+1}/{len(my_files)}, "
                                 f"total samples yielded: {total_samples_yielded}")

            except Exception as e:
                logging.error(f"[Rank {rank} Worker {worker_id}] Error reading zarr shard {fpath}: {e}")
                continue

        # DEBUG: Mark completion
        if self.debug or self.verbose:
            logging.debug(f"[Rank {rank} Worker {worker_id}] COMPLETED Zarr iteration, "
                         f"total samples: {total_samples_yielded}")

    # Re-added __len__ for progress bar, but safeguard against 0
    def __len__(self):
        if self.n_cells > 0:
            # See ParquetIterableDataset.__len__ note
            import torch.distributed as dist
            world_size = 1
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            return int(math.ceil(self.n_cells / world_size))
        return 1000000 # Fallback


class TileDBDataset(Dataset):
    """
    TileDB Dataset for efficient random access training (like scimilarity).

    Uses TileDB sparse arrays for lazy loading - only reads data when accessed.
    This is the most memory-efficient approach for large datasets.

    TileDB Structure (CellArr-style):
    - counts/: Sparse matrix (cell_index, gene_index) -> value
    - cell_metadata/: Cell metadata
    - gene_annotation/: Gene info

    Usage:
        dataset = TileDBDataset("data/train_tiledb")
        # Data is NOT loaded until __getitem__ is called
        sample = dataset[0]  # Reads only this row from TileDB
    """

    def __init__(
        self,
        path: Union[str, Path],
        verbose: bool = False,
        lognorm: bool = False,  # Data is already log-normalized from preprocessing
        target_sum: float = 1e4,
    ):
        try:
            import tiledb
            self.tiledb = tiledb
        except ImportError:
            raise ImportError("TileDB not installed. Run: pip install tiledb")

        self.path = Path(path)
        self.verbose = verbose
        self.lognorm = lognorm
        self.target_sum = target_sum

        if not self.path.exists():
            raise FileNotFoundError(f"TileDB path not found: {self.path}")

        # Load metadata
        metadata_file = self.path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            self.n_cells = self.metadata['n_cells']
            self.n_genes = self.metadata['n_genes']
        else:
            raise FileNotFoundError(f"TileDB metadata not found: {metadata_file}")

        # Configure TileDB for single-threaded access (better for DataLoader workers)
        self.cfg = tiledb.Config({
            "sm.compute_concurrency_level": 1,
            "sm.io_concurrency_level": 1,
        })

        # Open counts TileDB (will be used for random access)
        counts_uri = str(self.path / "counts")
        self.counts_tdb = tiledb.open(counts_uri, 'r', config=self.cfg)

        if verbose:
            print(f"TileDB Dataset: {self.path}")
            print(f"  Cells: {self.n_cells:,}, Genes: {self.n_genes:,}")
            print(f"  Sparsity: {self.metadata.get('sparsity', 'N/A')}")

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single cell's expression vector.
        This is the key lazy-loading method - only reads one row from TileDB.
        """
        # Query TileDB for this cell's data
        results = self.counts_tdb.multi_index[idx, :]

        # Reconstruct dense vector from sparse data
        gene_indices = results['gene_index']
        values = results['data']

        # Create dense vector
        x = np.zeros(self.n_genes, dtype=np.float32)
        if len(gene_indices) > 0:
            x[gene_indices] = values

        return torch.from_numpy(x)

    def __del__(self):
        if hasattr(self, 'counts_tdb'):
            self.counts_tdb.close()


class TileDBCollator:
    """
    Collator that fetches data from TileDB in batches.
    More efficient than single-row access for training.

    Usage:
        collator = TileDBCollator("data/train_tiledb")
        dataloader = DataLoader(dataset, collate_fn=collator)
    """

    def __init__(
        self,
        tiledb_path: Union[str, Path],
        lognorm: bool = False,
        target_sum: float = 1e4,
    ):
        try:
            import tiledb
            self.tiledb = tiledb
        except ImportError:
            raise ImportError("TileDB not installed. Run: pip install tiledb")

        self.path = Path(tiledb_path)
        self.lognorm = lognorm
        self.target_sum = target_sum

        # Load metadata
        with open(self.path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        self.n_genes = self.metadata['n_genes']

        # Configure TileDB
        self.cfg = tiledb.Config({
            "sm.mem.total_budget": 10000000000,  # 10GB
            "sm.compute_concurrency_level": 1,
            "sm.io_concurrency_level": 1,
        })

        # Open counts TileDB
        counts_uri = str(self.path / "counts")
        self.counts_tdb = tiledb.open(counts_uri, 'r', config=self.cfg)

    def __call__(self, batch_indices: list) -> torch.Tensor:
        """
        Collate a batch of cell indices into a tensor.
        Fetches all cells in one TileDB query (efficient!).
        """
        from scipy.sparse import coo_matrix

        # Batch query TileDB
        cell_indices = list(batch_indices)
        results = self.counts_tdb.multi_index[cell_indices, :]

        # Build sparse matrix from results
        n_cells = len(cell_indices)
        cell_idx_map = {orig: new for new, orig in enumerate(cell_indices)}

        # Map original cell indices to batch indices
        batch_cell_indices = np.array([cell_idx_map[c] for c in results['cell_index']])
        gene_indices = results['gene_index']
        values = results['data']

        # Create sparse matrix and convert to dense
        sparse_mat = coo_matrix(
            (values, (batch_cell_indices, gene_indices)),
            shape=(n_cells, self.n_genes)
        ).tocsr()

        # Convert to dense tensor
        X = torch.from_numpy(sparse_mat.toarray().astype(np.float32))

        return X

    def __del__(self):
        if hasattr(self, 'counts_tdb'):
            self.counts_tdb.close()


class TileDBIterableDataset(IterableDataset):
    """
    Iterable TileDB Dataset for DDP training.

    Combines the efficiency of TileDB with IterableDataset for distributed training.
    Each rank/worker gets a subset of cell indices, and data is loaded lazily.
    """

    def __init__(
        self,
        path: Union[str, Path],
        verbose: bool = False,
        shuffle: bool = True,
        debug: bool = False,
        equal_length: bool = True,
        batch_size: int = 1000,  # Read this many cells at once from TileDB
    ):
        try:
            import tiledb
            self.tiledb = tiledb
        except ImportError:
            raise ImportError("TileDB not installed. Run: pip install tiledb")

        self.path = Path(path)
        self.verbose = verbose
        self.shuffle = shuffle
        self.debug = debug
        self.equal_length = equal_length
        self.batch_size = batch_size

        if not self.path.exists():
            raise FileNotFoundError(f"TileDB path not found: {self.path}")

        # Load metadata
        with open(self.path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        self.n_cells = self.metadata['n_cells']
        self.n_genes = self.metadata['n_genes']

        if verbose:
            print(f"TileDB Iterable Dataset: {self.path}")
            print(f"  Cells: {self.n_cells:,}, Genes: {self.n_genes:,}")

    def __iter__(self):
        import torch.distributed as dist
        from scipy.sparse import coo_matrix

        # Get rank and world size
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Get worker info
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # Calculate cell range for this rank and worker
        cells_per_rank = self.n_cells // world_size
        rank_start = rank * cells_per_rank
        rank_end = self.n_cells if rank == world_size - 1 else rank_start + cells_per_rank

        rank_cells = rank_end - rank_start
        cells_per_worker = rank_cells // num_workers
        worker_start = rank_start + worker_id * cells_per_worker
        worker_end = rank_end if worker_id == num_workers - 1 else worker_start + cells_per_worker

        # Generate indices
        indices = np.arange(worker_start, worker_end)
        if self.shuffle:
            np.random.shuffle(indices)

        if self.debug or self.verbose:
            logging.debug(f"[Rank {rank} Worker {worker_id}] TileDB cells {worker_start}-{worker_end} ({len(indices):,} cells)")

        # Open TileDB
        cfg = self.tiledb.Config({
            "sm.compute_concurrency_level": 1,
            "sm.io_concurrency_level": 1,
        })
        counts_tdb = self.tiledb.open(str(self.path / "counts"), 'r', config=cfg)

        try:
            total_yielded = 0
            show_progress = (rank == 0 and worker_id == 0)

            # Process in batches for efficiency
            for batch_start in tqdm(range(0, len(indices), self.batch_size),
                                    desc=f"Loading TileDB",
                                    disable=not show_progress,
                                    unit="batch"):
                batch_end = min(batch_start + self.batch_size, len(indices))
                batch_indices = indices[batch_start:batch_end].tolist()

                # Batch query TileDB
                results = counts_tdb.multi_index[batch_indices, :]

                # Build batch data
                n_batch = len(batch_indices)
                idx_map = {orig: new for new, orig in enumerate(batch_indices)}

                if len(results['cell_index']) > 0:
                    batch_cell_idx = np.array([idx_map[c] for c in results['cell_index']])
                    gene_idx = results['gene_index']
                    values = results['data']

                    sparse_mat = coo_matrix(
                        (values, (batch_cell_idx, gene_idx)),
                        shape=(n_batch, self.n_genes)
                    ).tocsr()
                    batch_data = sparse_mat.toarray().astype(np.float32)
                else:
                    batch_data = np.zeros((n_batch, self.n_genes), dtype=np.float32)

                # Yield samples
                for i in range(n_batch):
                    yield torch.from_numpy(batch_data[i])
                    total_yielded += 1

                del batch_data

            if self.debug or self.verbose:
                logging.debug(f"[Rank {rank} Worker {worker_id}] TileDB iteration complete, {total_yielded:,} samples")

        finally:
            counts_tdb.close()

    def __len__(self):
        import torch.distributed as dist
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        return int(math.ceil(self.n_cells / world_size))


# Alias for backward compatibility if needed, or use ParquetDataset directly
StaticCellDataset = ParquetDataset


class TemporalCellDataset(Dataset):
    """
    时序单细胞数据集
    """
    def __init__(
        self,
        data: Union[str, Path, "ad.AnnData"],
        *,
        layer: Optional[str] = "log1p",
        max_genes: Optional[int] = None,
        target_genes: Optional[List[str]] = None,
        valid_pairs_only: bool = True,
        time_col: str = "time",
        next_cell_col: str = "next_cell_id",
        index_col: int = 0,
        verbose: bool = False,
    ):
        if isinstance(data, (str, Path)):
            self.adata = load_anndata(
                data, 
                index_col=index_col, 
                verbose=verbose, 
                max_genes=max_genes,
                target_genes=target_genes
            )
        else:
            self.adata = data
            if target_genes is not None:
                self.adata = load_anndata(self.adata, verbose=verbose, target_genes=target_genes)
            elif max_genes is not None and self.adata.shape[1] > max_genes:
                self.adata = load_anndata(self.adata, verbose=verbose, max_genes=max_genes)
        
        self.n_cells, self.n_genes = self.adata.shape
        self.time_col = time_col
        self.next_cell_col = next_cell_col
        
        # 检查必要列
        missing = [c for c in (time_col, next_cell_col) if c not in self.adata.obs.columns]
        if missing:
            raise ValueError(f"缺少必要的 obs 列: {missing}")
        
        self.valid_indices = np.where(~pd.isna(self.adata.obs[next_cell_col]))[0] if valid_pairs_only else np.arange(self.n_cells)
        
        # 读取表达矩阵
        X = self.adata.layers[layer] if (layer and layer in self.adata.layers) else self.adata.X
        X = X.toarray() if hasattr(X, "toarray") else X
        self.expressions = torch.tensor(X, dtype=torch.float32)
        
        # 时间信息
        self.times = torch.tensor(self.adata.obs[time_col].to_numpy(), dtype=torch.float32)
        if self.times.max() > 10: self.times = self.times / 100.0
        
        self.next_cell_ids = self.adata.obs[next_cell_col].to_numpy()
        # Handle NaN/-1 for next_cell_ids if necessary
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        current_idx = int(self.valid_indices[idx])
        x_cur = self.expressions[current_idx]
        t_cur = self.times[current_idx]
        
        # Simple next index resolution (assuming next_cell_col contains integer indices or can be mapped)
        # For simplicity, assuming it's an index here as per original code implication
        next_id = self.next_cell_ids[current_idx]
        next_idx = int(next_id) if (not pd.isna(next_id) and next_id != -1) else current_idx
        
        x_next = self.expressions[next_idx]
        t_next = self.times[next_idx]
        
        return {
            "x_cur": x_cur, "x_next": x_next,
            "t_cur": t_cur, "t_next": t_next,
        }


class MultiCellDataset(Dataset):
    """
    多细胞数据集
    """
    def __init__(
        self,
        data: Union[str, Path, "ad.AnnData"],
        *,
        n_cells_per_sample: int = 5,
        sampling_with_replacement: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        if isinstance(data, (str, Path)):
            self.adata = load_anndata(data, verbose=verbose, **kwargs)
        else:
            self.adata = data
            
        self.n_cells, self.n_genes = self.adata.shape
        self.n_cells_per_sample = n_cells_per_sample
        self.sampling_with_replacement = sampling_with_replacement
        
        X = self.adata.X
        X = X.toarray() if hasattr(X, "toarray") else X
        self.expressions = torch.tensor(X, dtype=torch.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_cells
    
    def __getitem__(self, idx: int):
        if self.sampling_with_replacement:
            indices = self.rng.integers(0, self.n_cells, size=self.n_cells_per_sample)
        else:
            indices = self.rng.choice(self.n_cells, size=self.n_cells_per_sample, replace=False)
            
        return {
            "expressions": self.expressions[indices],
            "cell_indices": torch.tensor(indices, dtype=torch.long),
        }


def collate_fn_static(batch):
    return torch.stack(batch, dim=0)


def collate_fn_temporal(batch):
    return {
        "x_cur": torch.stack([b["x_cur"] for b in batch]),
        "x_next": torch.stack([b["x_next"] for b in batch]),
        "t_cur": torch.stack([b["t_cur"] for b in batch]),
        "t_next": torch.stack([b["t_next"] for b in batch]),
    }


def collate_fn_multi_cell(batch):
    return {
        "expressions": torch.stack([b["expressions"] for b in batch]),
        "cell_indices": torch.stack([b["cell_indices"] for b in batch]),
    }
