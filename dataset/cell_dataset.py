"""
单细胞数据集
支持从 Parquet (推荐) 或 AnnData 加载数据
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scanpy as sc
from pathlib import Path
from typing import Union, Optional, List, Dict

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
    高效加载预处理好的 Parquet 文件 (AE 训练主要使用此 Dataset)
    """
    def __init__(self, path: Union[str, Path], verbose: bool = False):
        self.path = Path(path)
        if not self.path.exists():
            # 尝试寻找 path 目录下的 train.parquet 等（如果是目录）
            # 但为了简单，我们要求 path 必须是文件
            raise FileNotFoundError(f"Parquet file not found: {self.path}")
            
        if verbose:
            print(f"Loading Parquet dataset: {self.path}")
            
        # Read parquet using pandas
        try:
            df = pd.read_parquet(self.path)
            # Ensure float32
            self.data = torch.from_numpy(df.values).float()
            self.n_cells, self.n_genes = self.data.shape
            self.gene_names = df.columns.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet {self.path}: {e}")
            
        if verbose:
            print(f"  - Cells: {self.n_cells}")
            print(f"  - Genes: {self.n_genes}")

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


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
