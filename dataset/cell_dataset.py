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
        table = pq.read_table(fpath) # Reading whole shard (approx 6k cells) is fast enough
        row = table.slice(local_idx, length=1)
        
        # Convert to tensor
        # Use to_pandas() or direct buffer access
        x = torch.from_numpy(row.to_pandas().values[0]).float()
        return x


class ParquetIterableDataset(IterableDataset):
    """
    Iterative Parquet Dataset for efficient training on large sharded datasets.
    Avoids random access overhead by reading full shards and shuffling in memory.
    """
    def __init__(self, path: Union[str, Path], verbose: bool = False, shuffle_shards: bool = True, shuffle_rows: bool = True):
        self.path = Path(path)
        self.verbose = verbose
        self.shuffle_shards = shuffle_shards
        self.shuffle_rows = shuffle_rows
        self.files = []
        
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
        # We don't need total length for IterableDataset usually, but good to have estimate
        # We skip full metadata scan for speed unless cached
        try:
            import pyarrow.parquet as pq
            # Use read_table with slice to accurately determine columns (excluding index via to_pandas)
            sample = pq.read_table(self.files[0]).slice(0, 1)
            self.n_genes = sample.to_pandas().shape[1]
            self.n_cells = 0 # Unknown unless scanned
            
             # Cache check for n_cells (optional but helpful for logging)
            cache_file = self.path / "metadata_cache.json" if self.path.is_dir() else None
            if cache_file and cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                    if cache.get('n_files') == len(self.files):
                        self.n_cells = cache['n_cells']
                except: pass
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ParquetIterableDataset: {e}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:  # Single-process data loading
            my_files = list(self.files)
        else:  # Worker split
            # Split files among workers
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            my_files = self.files[iter_start:iter_end]
            
        if self.shuffle_shards:
            np.random.shuffle(my_files)
            
        import pyarrow.parquet as pq
        
        for fpath in my_files:
            try:
                # Read whole shard (much faster than random access)
                # Optimization: Read only needed columns if known (not implemented here as we need all genes)
                # Optimization: Use memory mapping if possible? parquet doesn't support mmap directly well
                
                # To avoid soft lockups with huge files:
                # 1. Limit the number of rows processed at once if files are huge?
                #    (Current shards are ~6k rows, which is fine, but let's be safe)
                
                table = pq.read_table(fpath)
                # Convert to numpy
                # We assume all columns are features
                # Use copy=False to avoid extra allocation if possible
                df = table.to_pandas()
                data = df.values.astype(np.float32, copy=False)
                
                # Explicitly delete table/df to free pyarrow memory early
                del table
                del df
                
                n_rows = len(data)
                indices = np.arange(n_rows)
                
                if self.shuffle_rows:
                    np.random.shuffle(indices)
                
                for idx in indices:
                    yield torch.from_numpy(data[idx])
                    
            except Exception as e:
                print(f"Error reading shard {fpath}: {e}")
                continue

    def __len__(self):
        # Return estimated length if known, else warning
        return self.n_cells


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
