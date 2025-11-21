"""
单细胞数据集
支持静态和时序单细胞数据加载
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scanpy as sc
from pathlib import Path
from typing import Union, Optional

try:
    import anndata as ad
except ImportError:
    ad = None


def load_anndata(
    data: Union[str, Path, "ad.AnnData"],
    *,
    index_col: int = 0,
    verbose: bool = False,
) -> sc.AnnData:
    """
    加载 AnnData 对象
    
    支持格式：
      - .h5ad, .h5, .hdf5（scanpy 读取）
      - .csv（第一列为细胞 id）
      - .xlsx/.xls（第一列为细胞 id）
      - 已经是 AnnData 的对象（直接返回）
    
    Args:
        data: 数据路径或 AnnData 对象
        index_col: CSV/Excel 文件的索引列
        verbose: 是否打印信息
    
    Returns:
        AnnData 对象
    """
    # 已经是 AnnData 对象
    if ad is not None and isinstance(data, ad.AnnData):
        return data
    
    # 文件路径
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
    
    return adata


class StaticCellDataset(Dataset):
    """
    静态单细胞数据集（用于 Autoencoder 训练）
    每次返回单个细胞的表达向量
    """
    
    def __init__(
        self,
        data: Union[str, Path, "ad.AnnData"],
        *,
        layer: Optional[str] = "log1p",
        index_col: int = 0,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            data: 数据路径或 AnnData 对象
            layer: 使用的 layer 名称，默认为 'log1p'。如果为 None 或不存在，则使用 adata.X
            index_col: CSV/Excel 文件的索引列
            verbose: 是否打印信息
            seed: 随机种子
        """
        if isinstance(data, (str, Path)):
            self.adata = load_anndata(data, index_col=index_col, verbose=verbose)
        else:
            self.adata = data
        
        self.n_cells, self.n_genes = self.adata.shape
        
        # 从指定的 layer 读取表达矩阵
        if layer is not None and layer in self.adata.layers:
            X = self.adata.layers[layer]
            if verbose:
                print(f"从 layer '{layer}' 读取表达数据")
        else:
            X = self.adata.X
            if verbose:
                if layer is not None:
                    print(f"Warning: layer '{layer}' 不存在，使用 adata.X")
                else:
                    print(f"使用 adata.X 读取表达数据")
        
        # 转换为 tensor（支持稀疏矩阵）
        X = X.toarray() if hasattr(X, "toarray") else X
        self.expressions = torch.tensor(X, dtype=torch.float32)
        
        # 随机数生成器
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        if verbose:
            print(f"StaticCellDataset 初始化完成:")
            print(f"  - 细胞数: {self.n_cells}")
            print(f"  - 基因数: {self.n_genes}")
    
    def __len__(self) -> int:
        return self.n_cells
    
    def __getitem__(self, idx: int):
        """
        返回单个细胞的表达向量
        
        Returns:
            x: [n_genes] 基因表达向量
        """
        # 随机采样（忽略 idx）
        rand_idx = int(self.rng.integers(0, self.n_cells))
        x = self.expressions[rand_idx]
        return x


class TemporalCellDataset(Dataset):
    """
    时序单细胞数据集（用于 Rectified Flow 训练）
    每个样本包含一个细胞及其下一个时间点的细胞
    """
    
    def __init__(
        self,
        data: Union[str, Path, "ad.AnnData"],
        *,
        layer: Optional[str] = "log1p",
        valid_pairs_only: bool = True,
        time_col: str = "time",
        next_cell_col: str = "next_cell_id",
        index_col: int = 0,
        verbose: bool = False,
    ):
        """
        Args:
            data: 数据路径或 AnnData 对象
            layer: 使用的 layer 名称，默认为 'log1p'。如果为 None 或不存在，则使用 adata.X
            valid_pairs_only: 是否只使用有效的时序对
            time_col: 时间列名
            next_cell_col: 下一个细胞 ID 列名
            index_col: CSV/Excel 文件的索引列
            verbose: 是否打印信息
        """
        if isinstance(data, (str, Path)):
            self.adata = load_anndata(data, index_col=index_col, verbose=verbose)
        else:
            self.adata = data
        
        self.n_cells, self.n_genes = self.adata.shape
        self.time_col = time_col
        self.next_cell_col = next_cell_col
        
        # 检查必要列
        missing = [c for c in (time_col, next_cell_col) if c not in self.adata.obs.columns]
        if missing:
            raise ValueError(f"缺少必要的 obs 列: {missing}")
        
        # 细胞名和索引映射
        self.cell_names = self.adata.obs.index.to_list()
        self.name_to_idx = {name: i for i, name in enumerate(self.cell_names)}
        
        # 有效时序对索引
        if valid_pairs_only:
            valid_mask = ~pd.isna(self.adata.obs[next_cell_col])
            self.valid_indices = np.where(valid_mask)[0]
        else:
            self.valid_indices = np.arange(self.n_cells)
        
        if verbose:
            print(f"TemporalCellDataset 初始化完成:")
            print(f"  - 细胞数: {self.n_cells}")
            print(f"  - 基因数: {self.n_genes}")
            print(f"  - 有效时序对: {len(self.valid_indices)} / {self.n_cells}")
        
        # 从指定的 layer 读取表达矩阵
        if layer is not None and layer in self.adata.layers:
            X = self.adata.layers[layer]
            if verbose:
                print(f"  - 从 layer '{layer}' 读取表达数据")
        else:
            X = self.adata.X
            if verbose:
                if layer is not None:
                    print(f"  - Warning: layer '{layer}' 不存在，使用 adata.X")
                else:
                    print(f"  - 使用 adata.X 读取表达数据")
        
        # 转换为 tensor（支持稀疏矩阵）
        X = X.toarray() if hasattr(X, "toarray") else X
        self.expressions = torch.tensor(X, dtype=torch.float32)
        
        # 时间信息
        self.times = torch.tensor(self.adata.obs[time_col].to_numpy(), dtype=torch.float32)
        # TODO: 根据实际数据调整时间归一化
        if self.times.max() > 10:  # 如果时间值很大，进行归一化
            self.times = self.times / 100.0
        
        # 下一个细胞 ID
        self.next_cell_ids = self.adata.obs[next_cell_col].to_numpy()
        # TODO: 临时处理，将 -1 替换为 0
        self.next_cell_ids[self.next_cell_ids == -1] = 0
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def _resolve_next_index(self, raw_id, fallback_idx: int) -> int:
        """
        解析下一个细胞的索引
        
        Args:
            raw_id: 原始 ID（可能是名称或索引）
            fallback_idx: 备用索引
        
        Returns:
            解析后的索引
        """
        # TODO: 根据实际数据格式完善
        idx = int(raw_id) if not pd.isna(raw_id) else fallback_idx
        return idx
    
    def __getitem__(self, idx: int):
        """
        返回时序对
        
        Returns:
            dict: 包含以下键
                - x_cur: [n_genes] 当前细胞表达
                - x_next: [n_genes] 下一个细胞表达
                - t_cur: 当前时间
                - t_next: 下一个时间
        """
        # 当前细胞索引
        current_idx = int(self.valid_indices[idx])
        
        x_cur = self.expressions[current_idx]
        t_cur = self.times[current_idx]
        
        # 解析下一个细胞索引
        raw_next_id = self.next_cell_ids[current_idx]
        next_idx = self._resolve_next_index(raw_next_id, fallback_idx=current_idx)
        
        x_next = self.expressions[next_idx]
        t_next = self.times[next_idx]
        
        return {
            "x_cur": x_cur,
            "x_next": x_next,
            "t_cur": t_cur,
            "t_next": t_next,
        }


class MultiCellDataset(Dataset):
    """
    多细胞数据集
    每次采样多个细胞的表达向量（用于集合编码器）
    """
    
    def __init__(
        self,
        data: Union[str, Path, "ad.AnnData"],
        *,
        layer: Optional[str] = "log1p",
        n_cells_per_sample: int = 5,
        sampling_with_replacement: bool = True,
        index_col: int = 0,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            data: 数据路径或 AnnData 对象
            layer: 使用的 layer 名称，默认为 'log1p'。如果为 None 或不存在，则使用 adata.X
            n_cells_per_sample: 每次采样的细胞数量
            sampling_with_replacement: 是否有放回采样
            index_col: CSV/Excel 文件的索引列
            verbose: 是否打印信息
            seed: 随机种子
        """
        if isinstance(data, (str, Path)):
            self.adata = load_anndata(data, index_col=index_col, verbose=verbose)
        else:
            self.adata = data
        
        self.n_cells, self.n_genes = self.adata.shape
        self.n_cells_per_sample = n_cells_per_sample
        self.sampling_with_replacement = sampling_with_replacement
        
        if not sampling_with_replacement and n_cells_per_sample > self.n_cells:
            raise ValueError(
                f"无放回采样时，每次采样细胞数 ({n_cells_per_sample}) "
                f"不能超过总细胞数 ({self.n_cells})"
            )
        
        # 从指定的 layer 读取表达矩阵
        if layer is not None and layer in self.adata.layers:
            X = self.adata.layers[layer]
            if verbose:
                print(f"从 layer '{layer}' 读取表达数据")
        else:
            X = self.adata.X
            if verbose:
                if layer is not None:
                    print(f"Warning: layer '{layer}' 不存在，使用 adata.X")
                else:
                    print(f"使用 adata.X 读取表达数据")
        
        # 转换为 tensor（支持稀疏矩阵）
        X = X.toarray() if hasattr(X, "toarray") else X
        self.expressions = torch.tensor(X, dtype=torch.float32)
        
        # 随机数生成器
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        if verbose:
            print(f"MultiCellDataset 初始化完成:")
            print(f"  - 细胞数: {self.n_cells}")
            print(f"  - 基因数: {self.n_genes}")
            print(f"  - 每次采样细胞数: {n_cells_per_sample}")
            print(f"  - 采样方式: {'有放回' if sampling_with_replacement else '无放回'}")
    
    def __len__(self) -> int:
        return self.n_cells
    
    def __getitem__(self, idx: int):
        """
        返回多个细胞的表达向量
        
        Returns:
            dict: 包含以下键
                - expressions: [n_cells_per_sample, n_genes] 表达矩阵
                - cell_indices: [n_cells_per_sample] 细胞索引
        """
        if self.sampling_with_replacement:
            sampled_indices = self.rng.integers(
                0, self.n_cells, size=self.n_cells_per_sample
            )
        else:
            sampled_indices = self.rng.choice(
                self.n_cells, size=self.n_cells_per_sample, replace=False
            )
        
        sampled_expressions = self.expressions[sampled_indices]
        
        return {
            "expressions": sampled_expressions,
            "cell_indices": torch.tensor(sampled_indices, dtype=torch.long),
        }


# ==================== Collate Functions ====================

def collate_fn_static(batch):
    """
    静态数据集的 collate 函数
    
    Args:
        batch: list of tensors [n_genes]
    
    Returns:
        tensor: [batch_size, n_genes]
    """
    return torch.stack(batch, dim=0)


def collate_fn_temporal(batch):
    """
    时序数据集的 collate 函数
    
    Args:
        batch: list of dicts
    
    Returns:
        dict: 批量数据
    """
    return {
        "x_cur": torch.stack([b["x_cur"] for b in batch]),
        "x_next": torch.stack([b["x_next"] for b in batch]),
        "t_cur": torch.stack([b["t_cur"] for b in batch]),
        "t_next": torch.stack([b["t_next"] for b in batch]),
    }


def collate_fn_multi_cell(batch):
    """
    多细胞数据集的 collate 函数
    
    Args:
        batch: list of dicts
    
    Returns:
        dict: 批量数据
    """
    return {
        "expressions": torch.stack([b["expressions"] for b in batch]),
        "cell_indices": torch.stack([b["cell_indices"] for b in batch]),
    }


