import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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
    将各种输入统一为 AnnData：

    支持：
      - .h5ad, .h5, .hdf5（scanpy 读取）
      - .csv（第一列为细胞 id）
      - .xlsx/.xls（第一列为细胞 id）
      - 已经是 AnnData 的对象（直接返回）
    """
    # 已经是 AnnData 对象
    if ad is not None and isinstance(data, ad.AnnData):
        return data  # type: ignore[return-value]

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
        adata = sc.read_h5(path)
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
    随机细胞数据集（用于 autoencoder 训练）：
    每次 __getitem__ 随机返回一个细胞的表达向量（而不是按索引顺序）。
    仍然支持从文件或 AnnData 对象构造。
    """

    def __init__(
        self,
        data: Union[str, Path, "ad.AnnData"],
        *,
        index_col: int = 0,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            data: 文件路径（.h5ad/.csv/.xlsx）或已加载的 AnnData。
            index_col: 若 data 为 CSV/Excel 文件，指定索引列。
            verbose: 是否打印加载信息。
            seed: 随机种子（可选），用于可复现采样。
        """
        if isinstance(data, (str, Path)):
            self.adata = load_anndata(data, index_col=index_col, verbose=verbose)
        else:
            self.adata = data

        self.n_cells, self.n_genes = self.adata.shape

        # 将表达矩阵转换为 dense numpy -> tensor（支持稀疏矩阵）
        X = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X
        self.expressions = torch.tensor(X, dtype=torch.float32)

        # 随机数生成器（用于可复现）
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        *,
        index_col: int = 0,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> "StaticCellDataset":
        return cls(file_path, index_col=index_col, verbose=verbose, seed=seed)

    def __len__(self) -> int:
        # 返回一个较大的值以便 DataLoader 可以持续采样；
        # 也可以返回 self.n_cells 如果希望 epoch 内限定步数。
        return self.n_cells

    def __getitem__(self, idx: int):
        """
        忽略输入的 idx（或将其作为 rng 的种子变体），随机选取一个细胞并返回表达向量。
        返回仅包含当前细胞表达（用于 autoencoder 的输入和目标）。
        """
        # 使用 rng 随机采样一个索引
        rand_idx = int(self.rng.integers(0, self.n_cells))
        x = self.expressions[rand_idx]
        return x


class TemporalCellDataset(Dataset):
    """
    时序细胞数据集：每个样本包含一个细胞及其“下一个”细胞的信息。
    """

    def __init__(
        self,
        data: Union[str, Path, "ad.AnnData"],
        *,
        valid_pairs_only: bool = True,
        time_col: str = "time",
        next_cell_col: str = "next_cell_id",
        index_col: int = 0,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            data:
                - str/Path: .h5ad/.h5/.csv/.xlsx/.xls 文件路径
                - AnnData: 已加载对象，包含表达矩阵和时间信息
            valid_pairs_only: 是否只使用有下一个细胞的样本
            time_col: 时间信息所在的 obs 列名
            next_cell_col: 下一个细胞 ID 所在的 obs 列名
            index_col: 若 data 为 CSV/Excel 文件，指定索引列
            verbose: 是否打印加载信息
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
            print(f"有效的时序对数量: {len(self.valid_indices)} / {self.n_cells}")

        # 表达矩阵 -> tensor（支持稀疏 / 稠密）
        X = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X
        self.expressions = torch.tensor(X, dtype=torch.float32)

        # 时间与 next_cell_id
        self.times = torch.tensor(self.adata.obs[time_col].to_numpy(), dtype=torch.float32)
        self.times = self.times/100 # TODO 临时，需要修改
        self.next_cell_ids = self.adata.obs[next_cell_col].to_numpy()
        # TODO 临时处理，后续需要解决
        self.next_cell_ids[self.next_cell_ids==-1] = 0

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        *,
        valid_pairs_only: bool = True,
        time_col: str = "time",
        next_cell_col: str = "next_cell_id",
        index_col: int = 0,
        verbose: bool = False,
    ) -> "TemporalCellDataset":
        """从文件创建数据集的便捷方法。"""
        return cls(
            data=file_path,
            valid_pairs_only=valid_pairs_only,
            time_col=time_col,
            next_cell_col=next_cell_col,
            index_col=index_col,
            verbose=verbose,
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _resolve_next_index(self, raw_id, fallback_idx: int) -> int:
        """
        将 next_cell_col 中的值解析为索引：
        - 若是合法的 cell 名称，使用 name_to_idx
        - 否则尝试转为整数索引
        - 若失败或越界，返回 fallback_idx
        """
        # if pd.isna(raw_id):
        #     return fallback_idx
        #
        # # 先当作 cell 名称
        # if isinstance(raw_id, str):
        #     idx = self.name_to_idx.get(raw_id)
        #     if idx is not None:
        #         return idx
        #
        # # 再尝试整数索引
        # try:
        #     idx = int(raw_id)
        # except (TypeError, ValueError):
        #     return fallback_idx
        #
        # if 0 <= idx < self.n_cells:
        #     return idx
        # return fallback_idx
        # TODO 待确认和完善
        idx = raw_id
        return idx

    def __getitem__(self, idx: int):
        # 当前细胞索引（映射到全局）
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
    多细胞数据集：每次 __getitem__ 随机采样多个细胞的表达向量。
    可以指定每次采样的细胞数量，支持有放回或无放回采样。
    """

    def __init__(
        self,
        data: Union[str, Path, "ad.AnnData"],
        *,
        n_cells_per_sample: int = 5,
        sampling_with_replacement: bool = True,
        index_col: int = 0,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            data: 文件路径（.h5ad/.csv/.xlsx）或已加载的 AnnData。
            n_cells_per_sample: 每次采样的细胞数量。
            sampling_with_replacement: 是否有放回采样。
            index_col: 若 data 为 CSV/Excel 文件，指定索引列。
            verbose: 是否打印加载信息。
            seed: 随机种子（可选），用于可复现采样。
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

        # 将表达矩阵转换为 dense numpy -> tensor（支持稀疏矩阵）
        X = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X
        self.expressions = torch.tensor(X, dtype=torch.float32)

        # 随机数生成器（用于可复现）
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        if verbose:
            print(f"MultiCellDataset 初始化完成:")
            print(f"  - 总细胞数: {self.n_cells}")
            print(f"  - 基因数: {self.n_genes}")
            print(f"  - 每次采样细胞数: {n_cells_per_sample}")
            print(f"  - 采样方式: {'有放回' if sampling_with_replacement else '无放回'}")

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        *,
        n_cells_per_sample: int = 5,
        sampling_with_replacement: bool = True,
        index_col: int = 0,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> "MultiCellDataset":
        return cls(
            file_path,
            n_cells_per_sample=n_cells_per_sample,
            sampling_with_replacement=sampling_with_replacement,
            index_col=index_col,
            verbose=verbose,
            seed=seed,
        )

    def __len__(self) -> int:
        # 返回一个较大的值以便 DataLoader 可以持续采样
        return self.n_cells

    def __getitem__(self, idx: int):
        """
        忽略输入的 idx，随机选取 n_cells_per_sample 个细胞并返回它们的表达向量。

        Returns:
            dict: 包含以下键值对
                - "expressions": shape (n_cells_per_sample, n_genes) 的张量
                - "cell_indices": shape (n_cells_per_sample,) 的张量，包含被采样细胞的原始索引
        """
        if self.sampling_with_replacement:
            # 有放回采样
            sampled_indices = self.rng.integers(
                0, self.n_cells, size=self.n_cells_per_sample
            )
        else:
            # 无放回采样
            sampled_indices = self.rng.choice(
                self.n_cells, size=self.n_cells_per_sample, replace=False
            )

        # 获取对应的表达向量
        sampled_expressions = self.expressions[sampled_indices]

        return {
            "expressions": sampled_expressions,
            "cell_indices": torch.tensor(sampled_indices, dtype=torch.long),
        }


# TODO 这个代码有问题，需要修改，起点和终点应该是同一个时间点
class MultiTemporalCellDataset(Dataset):
    """
    多细胞时序数据集：每次 __getitem__ 采样多个细胞及其对应的时序信息。
    支持两种模式：
    1. 采样多个独立的时序对 (current_cell -> next_cell)
    2. 采样一个时序链条 (cell1 -> cell2 -> cell3 -> ...)
    """

    def __init__(
        self,
        data: Union[str, Path, "ad.AnnData"],
        *,
        n_cells_per_sample: int = 5,
        sampling_mode: str = "independent_pairs",  # "independent_pairs" 或 "sequential_chain"
        valid_pairs_only: bool = True,
        time_col: str = "time",
        next_cell_col: str = "next_cell_id",
        index_col: int = 0,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            data: 文件路径或已加载的 AnnData
            n_cells_per_sample: 每次采样的细胞数量
            sampling_mode: 采样模式
                - "independent_pairs": 采样多个独立的时序对
                - "sequential_chain": 采样一个连续的时序链条
            valid_pairs_only: 是否只使用有下一个细胞的样本
            time_col: 时间信息所在的 obs 列名
            next_cell_col: 下一个细胞 ID 所在的 obs 列名
            index_col: 若 data 为 CSV/Excel 文件，指定索引列
            verbose: 是否打印加载信息
            seed: 随机种子
        """
        if isinstance(data, (str, Path)):
            self.adata = load_anndata(data, index_col=index_col, verbose=verbose)
        else:
            self.adata = data

        self.n_cells, self.n_genes = self.adata.shape
        self.n_cells_per_sample = n_cells_per_sample
        self.sampling_mode = sampling_mode
        self.time_col = time_col
        self.next_cell_col = next_cell_col

        if sampling_mode not in ["independent_pairs", "sequential_chain"]:
            raise ValueError(f"不支持的采样模式: {sampling_mode}")

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
            print(f"MultiTemporalCellDataset 初始化完成:")
            print(f"  - 总细胞数: {self.n_cells}")
            print(f"  - 有效时序对数量: {len(self.valid_indices)} / {self.n_cells}")
            print(f"  - 每次采样细胞数: {n_cells_per_sample}")
            print(f"  - 采样模式: {sampling_mode}")

        # 表达矩阵 -> tensor
        X = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X
        self.expressions = torch.tensor(X, dtype=torch.float32)

        # 时间与 next_cell_id
        self.times = torch.tensor(self.adata.obs[time_col].to_numpy(), dtype=torch.float32)
        self.next_cell_ids = self.adata.obs[next_cell_col].to_numpy()
        # TODO 临时处理，后续需要解决
        self.next_cell_ids[self.next_cell_ids == -1] = 0

        # 随机数生成器
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # 如果是sequential_chain模式，需要构建时序链条
        if sampling_mode == "sequential_chain":
            self._build_sequential_chains()

    def _build_sequential_chains(self):
        """构建可用的时序链条"""
        self.chains = []
        visited = set()

        for start_idx in self.valid_indices:
            if start_idx in visited:
                continue

            chain = []
            current_idx = start_idx

            while current_idx is not None and current_idx not in visited:
                chain.append(current_idx)
                visited.add(current_idx)

                # 找下一个细胞
                raw_next_id = self.next_cell_ids[current_idx]
                next_idx = self._resolve_next_index(raw_next_id, fallback_idx=None)

                if next_idx is None or next_idx == current_idx:
                    break

                current_idx = next_idx

            # 只保留长度足够的链条
            if len(chain) >= self.n_cells_per_sample:
                self.chains.append(chain)

        if len(self.chains) == 0:
            raise ValueError(f"找不到长度 >= {self.n_cells_per_sample} 的时序链条")

        if hasattr(self, 'verbose') and self.verbose:
            print(f"  - 可用时序链条数量: {len(self.chains)}")
            print(f"  - 链条长度分布: {[len(c) for c in self.chains[:5]]}")

    def _resolve_next_index(self, raw_id, fallback_idx: Optional[int]) -> Optional[int]:
        """解析下一个细胞索引"""
        if pd.isna(raw_id):
            return fallback_idx

        # 先当作 cell 名称
        if isinstance(raw_id, str):
            idx = self.name_to_idx.get(raw_id)
            if idx is not None:
                return idx

        # 再尝试整数索引
        try:
            idx = int(raw_id)
        except (TypeError, ValueError):
            return fallback_idx

        if 0 <= idx < self.n_cells:
            return idx
        return fallback_idx

    def __len__(self) -> int:
        if self.sampling_mode == "sequential_chain":
            return len(self.chains)
        else:
            return len(self.valid_indices)

    def __getitem__(self, idx: int):
        if self.sampling_mode == "independent_pairs":
            return self._sample_independent_pairs(idx)
        else:
            return self._sample_sequential_chain(idx)

    def _sample_independent_pairs(self, idx: int):
        """采样多个独立的时序对"""
        # 随机采样 n_cells_per_sample 个有效的时序对
        sampled_indices = self.rng.choice(
            len(self.valid_indices),
            size=self.n_cells_per_sample,
            replace=False if len(self.valid_indices) >= self.n_cells_per_sample else True
        )

        current_indices = self.valid_indices[sampled_indices]

        # 收集当前细胞和下一个细胞的信息
        x_current_list = []
        x_next_list = []
        t_current_list = []
        t_next_list = []
        current_idx_list = []
        next_idx_list = []

        for current_idx in current_indices:
            current_idx = int(current_idx)

            # 当前细胞信息
            x_cur = self.expressions[current_idx]
            t_cur = self.times[current_idx]

            # 解析下一个细胞索引
            raw_next_id = self.next_cell_ids[current_idx]
            next_idx = self._resolve_next_index(raw_next_id, fallback_idx=current_idx)

            x_next = self.expressions[next_idx]
            t_next = self.times[next_idx]

            x_current_list.append(x_cur)
            x_next_list.append(x_next)
            t_current_list.append(t_cur)
            t_next_list.append(t_next)
            current_idx_list.append(current_idx)
            next_idx_list.append(next_idx)

        return {
            "x_current": torch.stack(x_current_list),  # (n_cells_per_sample, n_genes)
            "x_next": torch.stack(x_next_list),        # (n_cells_per_sample, n_genes)
            "t_current": torch.stack(t_current_list),  # (n_cells_per_sample,)
            "t_next": torch.stack(t_next_list),        # (n_cells_per_sample,)
            "current_indices": torch.tensor(current_idx_list, dtype=torch.long),
            "next_indices": torch.tensor(next_idx_list, dtype=torch.long),
        }

    def _sample_sequential_chain(self, idx: int):
        """采样一个连续的时序链条"""
        chain = self.chains[idx]

        # 如果链条长度超过需要的数量，随机选择起始位置
        if len(chain) > self.n_cells_per_sample:
            start_pos = self.rng.integers(0, len(chain) - self.n_cells_per_sample + 1)
            selected_chain = chain[start_pos:start_pos + self.n_cells_per_sample]
        else:
            selected_chain = chain[:self.n_cells_per_sample]

        # 收集链条中所有细胞的信息
        x_sequence = []
        t_sequence = []

        for cell_idx in selected_chain:
            x_sequence.append(self.expressions[cell_idx])
            t_sequence.append(self.times[cell_idx])

        return {
            "x_sequence": torch.stack(x_sequence),     # (n_cells_per_sample, n_genes)
            "t_sequence": torch.stack(t_sequence),     # (n_cells_per_sample,)
            "sequence_indices": torch.tensor(selected_chain, dtype=torch.long),
        }

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        *,
        n_cells_per_sample: int = 5,
        sampling_mode: str = "independent_pairs",
        valid_pairs_only: bool = True,
        time_col: str = "time",
        next_cell_col: str = "next_cell_id",
        index_col: int = 0,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> "MultiTemporalCellDataset":
        return cls(
            data=file_path,
            n_cells_per_sample=n_cells_per_sample,
            sampling_mode=sampling_mode,
            valid_pairs_only=valid_pairs_only,
            time_col=time_col,
            next_cell_col=next_cell_col,
            index_col=index_col,
            verbose=verbose,
            seed=seed,
        )


def collate_fn_multi_temporal_pairs(batch):
    """
    多时序对数据集的批量处理函数 (independent_pairs 模式)
    """
    return {
        "x_current": torch.stack([b["x_current"] for b in batch]),    # (batch_size, n_cells_per_sample, n_genes)
        "x_next": torch.stack([b["x_next"] for b in batch]),          # (batch_size, n_cells_per_sample, n_genes)
        "t_current": torch.stack([b["t_current"] for b in batch]),    # (batch_size, n_cells_per_sample)
        "t_next": torch.stack([b["t_next"] for b in batch]),          # (batch_size, n_cells_per_sample)
        "current_indices": torch.stack([b["current_indices"] for b in batch]),
        "next_indices": torch.stack([b["next_indices"] for b in batch]),
    }


def collate_fn_multi_temporal_chains(batch):
    """
    多时序链条数据集的批量处理函数 (sequential_chain 模式)
    """
    return {
        "x_sequence": torch.stack([b["x_sequence"] for b in batch]),  # (batch_size, n_cells_per_sample, n_genes)
        "t_sequence": torch.stack([b["t_sequence"] for b in batch]),  # (batch_size, n_cells_per_sample)
        "sequence_indices": torch.stack([b["sequence_indices"] for b in batch]),
    }

def collate_fn_multi_cell(batch):
    """
    多细胞数据集的批量处理函数。

    Args:
        batch: list of dicts, 每个 dict 包含 "expressions" 和 "cell_indices"

    Returns:
        dict: 包含批量处理后的数据
            - "expressions": shape (batch_size, n_cells_per_sample, n_genes)
            - "cell_indices": shape (batch_size, n_cells_per_sample)
    """
    return {
        "expressions": torch.stack([b["expressions"] for b in batch]),
        "cell_indices": torch.stack([b["cell_indices"] for b in batch]),
    }

# language: python
def collate_fn_static(batch):
    """
    将 DataLoader 中的 list[tensor] 转为 batch tensor。
    batch: list of tensors shape (n_genes,)
    返回 tensor shape (batch_size, n_genes)
    """
    return torch.stack(batch, dim=0)


def collate_fn_temperal(batch):
    """
    自定义批量处理函数，将列表中的 dict 合并为 batch dict。
    """
    return {
        "x_cur": torch.stack([b["x_cur"] for b in batch]),
        "x_next": torch.stack([b["x_next"] for b in batch]),
        "t_cur": torch.stack([b["t_cur"] for b in batch]),
        "t_next": torch.stack([b["t_next"] for b in batch]),
    }

if __name__ == "__main__":
    # 示例用法（路径请按需修改）
    path = "data/hs_AD_Brain_Cerebellum[Organoid]_36-a.link.h5ad"

    print("=== TemporalCellDataset 示例 ===")
    dataset = TemporalCellDataset(path, valid_pairs_only=True, verbose=True)
    print(f"数据集大小: {len(dataset)}")
    print(f"基因数量: {dataset.n_genes}")

    sample = dataset[0]
    print("\n样本示例：")
    print("当前细胞表达量形状:", sample["x_cur"].shape)
    print("下一个细胞表达量形状:", sample["x_next"].shape)
    print("当前时间点:", sample["t_cur"].item())
    print("下一个时间点:", sample["t_next"].item())

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn_temperal,
        num_workers=0,
    )
    print("\nDataLoader 创建成功")

    for batch in dataloader:
        print("一个 batch 的 x_cur 形状:", batch["x_cur"].shape)
        print("时间差范围:",
              (batch["t_next"] - batch["t_cur"]).min().item(),
              "->",
              (batch["t_next"] - batch["t_cur"]).max().item())
        break

    print("\n" + "="*70 + "\n")

    print("=== MultiCellDataset 示例 ===")
    # 测试多细胞数据集
    multi_dataset = MultiCellDataset(
        path,
        n_cells_per_sample=5,
        sampling_with_replacement=True,
        verbose=True
    )
    print(f"多细胞数据集大小: {len(multi_dataset)}")

    multi_sample = multi_dataset[0]
    print("\n多细胞样本示例：")
    print("采样细胞表达量形状:", multi_sample["expressions"].shape)
    print("采样细胞索引:", multi_sample["cell_indices"])

    multi_dataloader = DataLoader(
        multi_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn_multi_cell,
        num_workers=0,
    )
    print("\n多细胞 DataLoader 创建成功")

    for batch in multi_dataloader:
        print("一个 batch 的 expressions 形状:", batch["expressions"].shape)
        print("一个 batch 的 cell_indices 形状:", batch["cell_indices"].shape)
        print("batch 内第一个样本的细胞索引:", batch["cell_indices"][0])
        break

    print("\n" + "="*70 + "\n")

    print("=== MultiTemporalCellDataset - Independent Pairs 模式 ===")
    # 测试多细胞时序数据集 - 独立时序对模式
    multi_temporal_pairs = MultiTemporalCellDataset(
        path,
        n_cells_per_sample=4,
        sampling_mode="independent_pairs",
        verbose=True
    )
    print(f"多时序对数据集大小: {len(multi_temporal_pairs)}")

    pairs_sample = multi_temporal_pairs[0]
    print("\n独立时序对样本示例：")
    print("当前细胞表达量形状:", pairs_sample["x_current"].shape)
    print("下一个细胞表达量形状:", pairs_sample["x_next"].shape)
    print("当前时间点形状:", pairs_sample["t_current"].shape)
    print("时间点值:", pairs_sample["t_current"])
    print("时间差:", pairs_sample["t_next"] - pairs_sample["t_current"])

    pairs_dataloader = DataLoader(
        multi_temporal_pairs,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn_multi_temporal_pairs,
        num_workers=0,
    )

    for batch in pairs_dataloader:
        print("Independent Pairs batch 的 x_current 形状:", batch["x_current"].shape)
        print("Independent Pairs batch 的 t_current 形状:", batch["t_current"].shape)
        print("时间差范围:",
              (batch["t_next"] - batch["t_current"]).min().item(),
              "->",
              (batch["t_next"] - batch["t_current"]).max().item())
        break

    print("\n" + "="*70 + "\n")

    print("=== MultiTemporalCellDataset - Sequential Chain 模式 ===")
    # 测试多细胞时序数据集 - 时序链条模式
    try:
        multi_temporal_chains = MultiTemporalCellDataset(
            path,
            n_cells_per_sample=3,
            sampling_mode="sequential_chain",
            verbose=True
        )
        print(f"时序链条数据集大小: {len(multi_temporal_chains)}")

        chain_sample = multi_temporal_chains[0]
        print("\n时序链条样本示例：")
        print("时序表达量形状:", chain_sample["x_sequence"].shape)
        print("时间序列形状:", chain_sample["t_sequence"].shape)
        print("时间序列值:", chain_sample["t_sequence"])
        print("时间差:", chain_sample["t_sequence"][1:] - chain_sample["t_sequence"][:-1])

        chains_dataloader = DataLoader(
            multi_temporal_chains,
            batch_size=3,
            shuffle=True,
            collate_fn=collate_fn_multi_temporal_chains,
            num_workers=0,
        )

        for batch in chains_dataloader:
            print("Sequential Chain batch 的 x_sequence 形状:", batch["x_sequence"].shape)
            print("Sequential Chain batch 的 t_sequence 形状:", batch["t_sequence"].shape)

            # 计算每个链条内的时间差
            time_diffs = batch["t_sequence"][:, 1:] - batch["t_sequence"][:, :-1]
            print("链条内时间差范围:", time_diffs.min().item(), "->", time_diffs.max().item())
            break

    except ValueError as e:
        print(f"Sequential Chain 模式失败: {e}")
        print("这可能是因为数据中没有足够长的时序链条")

    print("\n=== StaticCellDataset 示例 ===")
    # 测试静态细胞数据集
    static_dataset = StaticCellDataset(path, verbose=True, seed=42)
    print(f"静态数据集大小: {len(static_dataset)}")

    static_sample = static_dataset[0]
    print("静态样本形状:", static_sample.shape)

    static_dataloader = DataLoader(
        static_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn_static,
        num_workers=0,
    )

    for batch in static_dataloader:
        print("静态 DataLoader 一个 batch 形状:", batch.shape)
        break