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
from typing import Union, Optional, List
from .scbank.databank import DataBank
from .scbank.gene_vocab import GeneVocab

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
    
    支持格式：
      - .h5ad, .h5, .hdf5（scanpy 读取）
      - .csv（第一列为细胞 id）
      - .xlsx/.xls（第一列为细胞 id）
      - 已经是 AnnData 的对象（直接返回）
    
    Args:
        data: 数据路径或 AnnData 对象
        index_col: CSV/Excel 文件的索引列
        verbose: 是否打印信息
        max_genes: 最大基因数。如果数据基因数超过此值，自动选择高变基因
        select_hvg: 是否在基因数超标时选择高变基因
        target_genes: 目标基因列表。如果提供，将数据映射到此基因列表（缺失填0）
    
    Returns:
        AnnData 对象
    """
    # 已经是 AnnData 对象
    if ad is not None and isinstance(data, ad.AnnData):
        adata = data.copy()  # 复制一份，以免修改原对象
    else:
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
    
    # 基因映射（优先级高于 max_genes）
    if target_genes is not None:
        if verbose:
            print(f"正在将数据映射到目标基因列表 ({len(target_genes)} 个基因)...")
        
        # 确保 var 索引是唯一的
        adata.var_names_make_unique()
        
        # 检查重叠 (参考 scimilarity)
        overlap = len(set(adata.var_names) & set(target_genes))
        if verbose:
            print(f"  - 共有基因数: {overlap}")
        
        if overlap < 100 and verbose:
            print(f"Warning: 共有基因数 ({overlap}) 较少，请检查基因命名是否一致")
            
        # 记录原始基因
        orig_genes = adata.var_names.values
        
        # 创建一个新的 AnnData，包含目标基因
        # 使用 pandas 的 reindex 功能
        
        # 为了优化内存，强制使用 float32
        if hasattr(adata.X, "toarray"):
            X_mat = adata.X.toarray()
        else:
            X_mat = adata.X
            
        X_df = pd.DataFrame(X_mat, index=adata.obs_names, columns=adata.var_names)
            
        # Reindex
        # fill_value=0.0
        X_new = X_df.reindex(columns=target_genes, fill_value=0.0)
        
        # 创建新 AnnData
        new_adata = sc.AnnData(
            X=X_new.values.astype(np.float32),
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=target_genes)
        )
        
        # 复制 layers
        for layer_name, layer_data in adata.layers.items():
            if hasattr(layer_data, "toarray"):
                layer_mat = layer_data.toarray()
            else:
                layer_mat = layer_data
                
            layer_df = pd.DataFrame(layer_mat, index=adata.obs_names, columns=adata.var_names)
            new_layer = layer_df.reindex(columns=target_genes, fill_value=0.0)
            new_adata.layers[layer_name] = new_layer.values.astype(np.float32)
            
        new_adata.uns["orig_genes"] = orig_genes
            
        adata = new_adata
        if verbose:
            print(f"基因映射完成，新维度: {adata.shape}")
            
    # 基因数检查和高变基因选择（仅当没有提供 target_genes 时）
    elif max_genes is not None and adata.shape[1] > max_genes and select_hvg:
        if verbose:
            print(f"基因数 ({adata.shape[1]}) 超过上限 ({max_genes})，自动选择 {max_genes} 个高变基因...")
        
        # 选择高变基因
        adata = _select_highly_variable_genes(adata, n_top_genes=max_genes, verbose=verbose)
        
        if verbose:
            print(f"高变基因选择完成，新维度: {adata.shape}")
    
    return adata


def _select_highly_variable_genes(
    adata: sc.AnnData,
    n_top_genes: int,
    verbose: bool = False,
) -> sc.AnnData:
    """
    选择高变基因
    
    Args:
        adata: AnnData 对象
        n_top_genes: 要选择的基因数量
        verbose: 是否打印信息
    
    Returns:
        筛选后的 AnnData 对象
    """
    # 保存原始数据（如果还没有保存）
    if 'counts' not in adata.layers:
        # 如果 X 是稀疏矩阵，需要先转换
        if hasattr(adata.X, "toarray"):
            adata.layers['counts'] = adata.X.toarray().copy()
        else:
            adata.layers['counts'] = adata.X.copy()
    
    # 计算高变基因
    # 使用 Seurat 方法，这是单细胞数据中最常用的方法
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor='seurat_v3',
            layer='counts' if 'counts' in adata.layers else None,
            subset=False,  # 先不筛选，只标记
        )
    except Exception as e:
        if verbose:
            print(f"Warning: Seurat v3 方法失败 ({e})，使用默认方法...")
        # 如果失败，使用默认的 seurat 方法
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor='seurat',
            subset=False,
        )
    
    # 筛选高变基因
    adata = adata[:, adata.var['highly_variable']].copy()
    
    if verbose:
        print(f"已选择 {adata.shape[1]} 个高变基因")
    
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
        max_genes: Optional[int] = None,
        target_genes: Optional[List[str]] = None,
        index_col: int = 0,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            data: 数据路径或 AnnData 对象
            layer: 使用的 layer 名称
            max_genes: 最大基因数（自动选择高变基因）
            target_genes: 目标基因列表（用于映射）
            index_col: CSV/Excel 文件的索引列
            verbose: 是否打印信息
            seed: 随机种子
        """
        self.use_scbank = False
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.is_dir():
                # Check for existing scBank
                if (path / "manifest.json").exists():
                    if verbose: print(f"Loading from scBank: {path}")
                    self.use_scbank = True
                    self.db = DataBank.from_path(path)
                # Check for h5ad directory to auto-convert/load cache
                elif any(path.glob("*.h5ad")):
                    cache_dir = path / ".scbank_cache"
                    # Check for gene vocabulary
                    vocab_file = path / "gene_vocab.json"
                    if not vocab_file.exists():
                        vocab_file = path / "gene_order.tsv"
                        
                    if (cache_dir / "manifest.json").exists():
                         if verbose: print(f"Loading from cached scBank: {cache_dir}")
                         self.use_scbank = True
                         self.db = DataBank.from_path(cache_dir)
                    elif vocab_file.exists():
                         if verbose: print(f"Converting h5ad directory to scBank cache at {cache_dir}...")
                         vocab = GeneVocab.from_file(vocab_file)
                         self.db = DataBank.from_h5ad_dir(path, vocab, to=cache_dir)
                         self.use_scbank = True
        
        if self.use_scbank:
             self.ds = self.db.main_data.data
             self.n_cells = len(self.ds)
             self.n_genes = len(self.db.gene_vocab)
             if verbose:
                print(f"StaticCellDataset (scBank) 初始化完成:")
                print(f"  - 细胞数: {self.n_cells}")
                print(f"  - 基因数: {self.n_genes}")
             return

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
            # 基因映射
            if target_genes is not None:
                self.adata = load_anndata(self.adata, verbose=verbose, target_genes=target_genes)
            # 基因数检查
            elif max_genes is not None and self.adata.shape[1] > max_genes:
                if verbose:
                    print(f"基因数 ({self.adata.shape[1]}) 超过上限 ({max_genes})，自动选择 {max_genes} 个高变基因...")
                self.adata = _select_highly_variable_genes(self.adata, n_top_genes=max_genes, verbose=verbose)
        
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
        
        if self.use_scbank:
            item = self.ds[rand_idx]
            # item keys: 'id', 'genes', 'expressions'
            x = torch.zeros(self.n_genes, dtype=torch.float32)
            if len(item['genes']) > 0:
                genes_idx = torch.tensor(item['genes'], dtype=torch.long)
                exprs = torch.tensor(item['expressions'], dtype=torch.float32)
                x[genes_idx] = exprs
            return x

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
        max_genes: Optional[int] = None,
        target_genes: Optional[List[str]] = None,
        valid_pairs_only: bool = True,
        time_col: str = "time",
        next_cell_col: str = "next_cell_id",
        index_col: int = 0,
        verbose: bool = False,
    ):
        """
        Args:
            data: 数据路径或 AnnData 对象
            layer: 使用的 layer 名称
            max_genes: 最大基因数
            target_genes: 目标基因列表
            valid_pairs_only: 是否只使用有效的时序对
            time_col: 时间列名
            next_cell_col: 下一个细胞 ID 列名
            index_col: CSV/Excel 文件的索引列
            verbose: 是否打印信息
        """
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
            # 基因映射
            if target_genes is not None:
                self.adata = load_anndata(self.adata, verbose=verbose, target_genes=target_genes)
            # 基因数检查
            elif max_genes is not None and self.adata.shape[1] > max_genes:
                if verbose:
                    print(f"基因数 ({self.adata.shape[1]}) 超过上限 ({max_genes})，自动选择 {max_genes} 个高变基因...")
                self.adata = _select_highly_variable_genes(self.adata, n_top_genes=max_genes, verbose=verbose)
        
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
        max_genes: Optional[int] = None,
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
            max_genes: 最大基因数。如果数据基因数超过此值，自动选择高变基因
            n_cells_per_sample: 每次采样的细胞数量
            sampling_with_replacement: 是否有放回采样
            index_col: CSV/Excel 文件的索引列
            verbose: 是否打印信息
            seed: 随机种子
        """
        if isinstance(data, (str, Path)):
            self.adata = load_anndata(data, index_col=index_col, verbose=verbose, max_genes=max_genes)
        else:
            self.adata = data
            # 如果是 AnnData 对象，也需要检查基因数
            if max_genes is not None and self.adata.shape[1] > max_genes:
                if verbose:
                    print(f"基因数 ({self.adata.shape[1]}) 超过上限 ({max_genes})，自动选择 {max_genes} 个高变基因...")
                self.adata = _select_highly_variable_genes(self.adata, n_top_genes=max_genes, verbose=verbose)
        
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


