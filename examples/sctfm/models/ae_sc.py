#!/usr/bin/env python
import logging
import fire
import numpy as np
import pandas as pd
import scanpy as sc

# from scimilarity.cell_embedding import CellEmbedding
from scimilarity.utils import align_dataset, lognorm_counts
from scimilarity.nn_models import Encoder
from scimilarity.nn_models import Decoder

from typing import Optional, Tuple, Union

LOGGER_NAME = "scimilarity_embed"
LOGGER = logging.getLogger(LOGGER_NAME)


def setup_logger(log_level: str = "INFO", log_file: str | None = None):
    """
    配置全局日志 LOGGER：
      - log_level: 字符串级别，比如 'DEBUG', 'INFO', 'WARNING', 'ERROR'
      - log_file:  如果给路径，则同时写到文件；否则只打到 stderr
    """
    # 防止重复添加 handler（多次调用 encode/decode 的情况下）
    if LOGGER.handlers:
        LOGGER.handlers.clear()

    # 转成真正的 level
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    LOGGER.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)

    # 可选：写文件
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        LOGGER.addHandler(fh)

    return LOGGER


class CellEmbedding:
    """A class that embeds cell gene expression data using an ML model.

    Parameters
    ----------
    model_path: str
        Path to the directory containing model files.
    use_gpu: bool, default: False
        Use GPU instead of CPU.

    Examples
    --------
    >>> ce = CellEmbedding(model_path="/opt/data/model")
    """

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,  # TODO: 改为用device，指定哪个GPU
    ):
        import json
        import os
        import pandas as pd

        self.model_path = model_path
        self.use_gpu = use_gpu

        self.filenames = {
            "model": os.path.join(self.model_path, "encoder.ckpt"),
            "gene_order": os.path.join(self.model_path, "gene_order.tsv"),
        }

        # get gene order
        with open(self.filenames["gene_order"], "r") as fh:
            self.gene_order = [line.strip() for line in fh]

        # get neural network model and infer network size
        with open(os.path.join(self.model_path, "layer_sizes.json"), "r") as fh:
            layer_sizes = json.load(fh)
        # keys: network.1.weight, network.2.weight, ..., network.n.weight
        layers = [
            (key, layer_sizes[key])
            for key in sorted(list(layer_sizes.keys()))
            if "weight" in key and len(layer_sizes[key]) > 1
        ]
        parameters = {
            "latent_dim": layers[-1][1][0],  # last
            "hidden_dim": [layer[1][0] for layer in layers][0:-1],  # all but last
        }

        self.n_genes = len(self.gene_order)
        self.latent_dim = parameters["latent_dim"]
        self.model = Encoder(
            n_genes=self.n_genes,
            latent_dim=parameters["latent_dim"],
            hidden_dim=parameters["hidden_dim"],
        )
        if self.use_gpu is True:
            self.model.cuda()
        self.model.load_state(self.filenames["model"])
        self.model.eval()

        self.int2label = pd.read_csv(
            os.path.join(self.model_path, "label_ints.csv"), index_col=0
        )["0"].to_dict()
        self.label2int = {value: key for key, value in self.int2label.items()}

    def get_embeddings(
        self,
        X: Union["scipy.sparse.csr_matrix", "scipy.sparse.csc_matrix", "numpy.ndarray"],
        num_cells: int = -1,
        buffer_size: int = 10000,
    ) -> "numpy.ndarray":
        """Calculate embeddings for lognormed gene expression matrix.

        Parameters
        ----------
        X: scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, numpy.ndarray
            Gene space aligned and log normalized (tp10k) gene expression matrix.
        num_cells: int, default: -1
            The number of cells to embed, starting from index 0.
            A value of -1 will embed all cells.
        buffer_size: int, default: 10000
            The number of cells to embed in one batch.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of embeddings [num_cells x latent_space_dimensions].

        Examples
        --------
        >>> from scimilarity.utils import align_dataset, lognorm_counts
        >>> ce = CellEmbedding(model_path="/opt/data/model")
        >>> data = align_dataset(data, ce.gene_order)
        >>> data = lognorm_counts(data)
        >>> embeddings = ce.get_embeddings(data.X)
        """

        import numpy as np
        from scipy.sparse import csr_matrix, csc_matrix
        import torch
        import zarr

        if num_cells == -1:
            num_cells = X.shape[0]

        if (
            (isinstance(X, csr_matrix) or isinstance(X, csc_matrix))
            and (
                isinstance(X.data, zarr.core.Array)
                or isinstance(X.indices, zarr.core.Array)
                or isinstance(X.indptr, zarr.core.Array)
            )
            and num_cells <= buffer_size
        ):
            X.data = X.data[...]
            X.indices = X.indices[...]
            X.indptr = X.indptr[...]

        embedding_parts = []
        with torch.inference_mode():  # disable gradients, not needed for inference
            for i in range(0, num_cells, buffer_size):
                profiles = None
                if isinstance(X, np.ndarray):
                    profiles = torch.Tensor(X[i : i + buffer_size])
                elif isinstance(X, torch.Tensor):
                    profiles = X[i : i + buffer_size]
                elif isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
                    profiles = torch.Tensor(X[i : i + buffer_size].toarray())

                if profiles is None:
                    raise RuntimeError(f"Unknown data type {type(X)}.")

                if self.use_gpu is True:
                    profiles = profiles.cuda()
                embedding_parts.append(self.model(profiles))

        if not embedding_parts:
            raise RuntimeError("No valid cells detected.")

        import torch as _torch  # avoid shadowing

        if self.use_gpu:
            # detach, move from gpu into cpu, return as numpy array
            embedding = _torch.vstack(embedding_parts).detach().cpu().numpy()
        else:
            # detach, return as numpy array
            embedding = _torch.vstack(embedding_parts).detach().numpy()

        if np.isnan(embedding).any():
            raise RuntimeError(
                "NaN detected in embeddings.", np.isnan(embedding).sum()
            )

        return embedding


class CellDecoder:
    """A class that decodes latent embeddings back to gene expression using an ML model.

    Parameters
    ----------
    model_path: str
        Path to the directory containing model files (decoder.ckpt, gene_order.tsv, layer_sizes.json).
    use_gpu: bool, default: False
        Use GPU instead of CPU.

    Examples
    --------
    >>> cd = CellDecoder(model_path="/opt/data/model")
    >>> decoded = cd.decode(latent_X)
    """

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,  # TODO: 改为用 device，指定哪个 GPU
    ):
        import json
        import os

        self.model_path = model_path
        self.use_gpu = use_gpu

        self.filenames = {
            "model": os.path.join(self.model_path, "decoder.ckpt"),
            "gene_order": os.path.join(self.model_path, "gene_order.tsv"),
        }

        # get gene order
        with open(self.filenames["gene_order"], "r") as fh:
            self.gene_order = [line.strip() for line in fh]

        # infer network sizes from layer_sizes.json
        with open(os.path.join(self.model_path, "layer_sizes.json"), "r") as fh:
            layer_sizes = json.load(fh)
        # keys: something.1.weight, something.2.weight, ...
        layers = [
            (key, layer_sizes[key])
            for key in sorted(list(layer_sizes.keys()))
            if "weight" in key
            and isinstance(layer_sizes[key], (list, tuple))
            and len(layer_sizes[key]) > 1
        ]
        # 与 encoder 一样推断：最后一层的 out_dim 当 latent_dim，
        # 其余层的 out_dim 当 hidden_dim
        parameters = {
            "latent_dim": layers[-1][1][0],  # last
            "hidden_dim": [layer[1][0] for layer in layers][0:-1],  # all but last
        }

        self.n_genes = len(self.gene_order)
        self.latent_dim = parameters["latent_dim"]
        self.hidden_dim = parameters["hidden_dim"]

        # build decoder network
        self.model = Decoder(
            n_genes=self.n_genes,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        )
        if self.use_gpu:
            self.model.cuda()
        self.model.load_state(self.filenames["model"])
        self.model.eval()

    def decode(
        self,
        Z: Union["numpy.ndarray", "scipy.sparse.csr_matrix", "scipy.sparse.csc_matrix"],
        num_cells: int = -1,
        buffer_size: int = 10000,
    ) -> "numpy.ndarray":
        """Decode latent embeddings back to gene expression.

        Parameters
        ----------
        Z: numpy.ndarray or scipy sparse matrix
            Latent embeddings [num_cells x latent_dim].
        num_cells: int, default: -1
            The number of cells to decode, starting from index 0.
            A value of -1 will decode all cells.
        buffer_size: int, default: 10000
            The number of cells to decode in one batch.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of decoded expressions [num_cells x n_genes].
        """
        import numpy as np
        from scipy.sparse import csr_matrix, csc_matrix
        import torch
        import zarr

        if num_cells == -1:
            num_cells = Z.shape[0]

        # 特别处理 zarr-backed 的稀疏矩阵，和 encoder 保持风格一致
        if (
            (isinstance(Z, csr_matrix) or isinstance(Z, csc_matrix))
            and (
                isinstance(Z.data, zarr.core.Array)
                or isinstance(Z.indices, zarr.core.Array)
                or isinstance(Z.indptr, zarr.core.Array)
            )
            and num_cells <= buffer_size
        ):
            Z.data = Z.data[...]
            Z.indices = Z.indices[...]
            Z.indptr = Z.indptr[...]

        decoded_parts = []
        with torch.inference_mode():
            for i in range(0, num_cells, buffer_size):
                profiles = None
                if isinstance(Z, np.ndarray):
                    profiles = torch.Tensor(Z[i : i + buffer_size])
                elif isinstance(Z, torch.Tensor):
                    profiles = Z[i : i + buffer_size]
                elif isinstance(Z, csr_matrix) or isinstance(Z, csc_matrix):
                    profiles = torch.Tensor(Z[i : i + buffer_size].toarray())

                if profiles is None:
                    raise RuntimeError(f"Unknown data type {type(Z)}.")

                if self.use_gpu:
                    profiles = profiles.cuda()

                decoded_batch = self.model(profiles)
                decoded_parts.append(decoded_batch)

        if not decoded_parts:
            raise RuntimeError("No valid cells detected for decoding.")

        import torch as _torch  # avoid shadowing

        if self.use_gpu:
            decoded = _torch.vstack(decoded_parts).detach().cpu().numpy()
        else:
            decoded = _torch.vstack(decoded_parts).detach().numpy()

        if np.isnan(decoded).any():
            raise RuntimeError("NaN detected in decoded expressions.")

        return decoded


def build_aligned_adata(adata, gene_order, min_overlap: int = 5000):
    """
    根据模型 gene_order 对齐基因。
    - 如果重叠基因 < min_overlap，则补零构建完整矩阵
    - 否则调用 align_dataset
    """
    adata_genes = adata.var_names
    gene_order_index = pd.Index(gene_order)

    overlap_genes = adata_genes.intersection(gene_order_index)
    overlap = len(overlap_genes)
    LOGGER.info(f"重叠基因数: {overlap}")

    if overlap >= min_overlap:
        LOGGER.info("重叠基因足够，使用 align_dataset 对齐...")
        adata_aligned = align_dataset(adata, gene_order)
    else:
        # 情况 2：重叠基因不足，补零
        LOGGER.warning(f"重叠基因不足 ({overlap} < {min_overlap})，将补全缺失基因为 0")

        # 构建完整矩阵（按 gene_order 排列）
        X = adata[:, overlap_genes].X
        # 创建空矩阵并填入已有基因数据
        full_X = np.zeros((adata.n_obs, len(gene_order_index)))
        full_X[:, gene_order_index.get_indexer(overlap_genes)] = np.nan_to_num(X, nan=0.0)

        # 构建新 AnnData
        adata_aligned = sc.AnnData(
            X=full_X,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=gene_order_index),
        )
    return adata_aligned


def project_decoded(decoded, gene_order, target_genes):
    """
    将 decoded (n_cells, len(gene_order))
    投影到 target_genes (例如 adata.var_names)。

    返回形状为 (n_cells, len(target_genes)) 的矩阵。
    """
    import numpy as np
    import pandas as pd

    gene_order = pd.Index(gene_order)
    target_genes = pd.Index(target_genes)

    # 计算 target_genes 在 gene_order 中的位置
    idx = gene_order.get_indexer(target_genes)

    # 初始化输出
    out = np.zeros((decoded.shape[0], target_genes.size), dtype=decoded.dtype)

    # 有效基因（存在于 gene_order）
    valid = idx >= 0
    out[:, valid] = decoded[:, idx[valid]]

    return out


def decode(
    input_file: str,
    output_file: str,
    model_path: str = "/gpfs/flash/home/yr/data/public/SCimilarity/model/model_v1.1/",
    col_source: str = "X_scimilarity",
    col_target: str = "X_scimilarity_decoded",
    buffer_size: int = 10000,
    log_level: str = "WARNING",
    log_file: str | None = None,
):
    """
    根据 scimilarity latent embedding 解码出基因表达，并写回 h5ad 文件。

    参数：
        input_file:   输入 h5ad 文件路径（里面含有 obsm[col_source] 的 latent）
        output_file:  输出 h5ad 文件路径
        model_path:   模型路径（包含 decoder.ckpt, gene_order.tsv, layer_sizes.json）
        col_source:   隐变量所在的 obsm 列名
        col_target:   解码后写入的层名（adata.layers[col_target]）
        buffer_size:  decode 时的 buffer_size
        log_level:    日志级别, e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR'
        log_file:     日志文件路径（可选），如果不填则只输出到 stderr
    """
    setup_logger(log_level=log_level, log_file=log_file)

    LOGGER.info(f"加载解码模型: {model_path}")
    cd = CellDecoder(model_path=model_path, use_gpu=False)

    # 3. 读取数据
    LOGGER.info(f"读取数据: {input_file}")
    adata = sc.read(input_file)
    LOGGER.info(f"原始数据维度: {adata.shape}")
    LOGGER.info(f"模型要求基因数: {len(cd.gene_order)}")

    # 5. 取出 latent
    if col_source not in adata.obsm_keys():
        raise KeyError(
            f"找不到 latent embedding: adata.obsm['{col_source}'] 不存在。"
        )

    Z = adata.obsm[col_source].astype(np.float32)
    LOGGER.info(f"latent 维度: {Z.shape}")

    if Z.shape[1] != cd.latent_dim:
        raise RuntimeError(
            f"latent 维度不匹配: Z.shape[1]={Z.shape[1]}, "
            f"但模型 latent_dim={cd.latent_dim}"
        )

    # 6. 解码
    LOGGER.info(
        f"开始解码 latent 为基因表达，buffer_size={buffer_size}, num_cells=-1"
    )
    decoded = cd.decode(
        Z,
        num_cells=-1,
        buffer_size=buffer_size,
    )

    LOGGER.info(
        f"解码完成，decoded 形状: {decoded.shape}, "
        f"范围: [{decoded.min():.2f}, {decoded.max():.2f}]"
    )

    decoded_aligned = project_decoded(
        decoded,
        cd.gene_order,
        adata.var_names,  # 仅传基因，不传整个 adata
    )

    adata.layers[col_target] = decoded_aligned

    # 8. 保存结果
    adata.write(output_file)
    LOGGER.info(
        f"保存至: {output_file}，layer='{col_target}' 已写入解码后的基因表达矩阵。"
    )


def encode(
    input_file: str,
    output_file: str,
    model_path: str = "/gpfs/flash/home/yr/data/public/SCimilarity/model/model_v1.1/",
    col_target: str = "X_scimilarity",
    buffer_size: int = 10000,
    log_level: str = "WARNING",
    log_file: str | None = None,
):
    """
    计算 scimilarity embedding 并写回 h5ad 文件。

    参数：
        input_file:   输入 h5ad 文件路径
        output_file:  输出 h5ad 文件路径
        model_path:   模型路径
        buffer_size:  get_embeddings 时的 buffer_size
        log_level:    日志级别，e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR'
        log_file:     日志文件路径（可选），如果不填则只输出到 stderr
    """
    setup_logger(log_level=log_level, log_file=log_file)

    LOGGER.info(f"加载模型: {model_path}")
    ce = CellEmbedding(model_path=model_path, use_gpu=False)

    LOGGER.info(f"读取数据: {input_file}")
    adata = sc.read(input_file)
    LOGGER.info(f"原始数据维度: {adata.shape}")
    LOGGER.info(f"模型要求基因数: {len(ce.gene_order)}")

    # 基因对齐（可能补零）
    adata_aligned = build_aligned_adata(adata, ce.gene_order)

    # 数据已是 log1p，直接使用或进行 lognorm
    if np.issubdtype(adata_aligned.X.dtype, np.integer):
        if "counts" not in adata_aligned.layers:
            adata_aligned.layers["counts"] = adata_aligned.X.copy()
        adata_aligned = lognorm_counts(adata_aligned)
        X = adata_aligned.X
    else:
        # adata_aligned.X = np.nan_to_num(adata_aligned.X, nan=0.0)
        X = adata_aligned.X.astype(np.float32)

    LOGGER.info(
        "数据已为 log1p 格式或已标准化，"
        f"范围: [{X.min():.2f}, {X.max():.2f}]"
    )

    # 计算 embedding
    LOGGER.info(
        f"开始计算 embeddings，buffer_size={buffer_size}, num_cells=-1"
    )
    embeddings = ce.get_embeddings(
        X,
        num_cells=-1,
        buffer_size=buffer_size,
    )

    # 写回原始 adata（保留原来的 obs/var 等信息）
    adata.obsm[col_target] = embeddings
    adata.write(output_file)
    LOGGER.info(
        f"保存至: {output_file}，embeddings 形状: {embeddings.shape}"
    )


if __name__ == "__main__":
    """
    统一入口，可以选择 encode 或 decode。

    用法示例：
      编码：
        python script.py encode input.h5ad output.h5ad \
            --log_level=INFO --log_file=encode.log

      解码：
        python script.py decode input.h5ad output_decoded.h5ad \
            --col_source=X_scimilarity \
            --col_target=X_scimilarity_decoded \
            --log_level=INFO --log_file=decode.log
    """
    # 使用 Fire 的“子命令”风格：
    #   script.py encode ...
    #   script.py decode ...
    fire.Fire(
        {
            "encode": encode,
            "decode": decode,
        }
    )
