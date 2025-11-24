#!/usr/bin/env python
"""
cellTime 推理脚本
使用训练好的 AE + RTF 模型进行单细胞时序预测
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fire
import torch
import numpy as np
import scanpy as sc
from typing import Optional, List, Union
from tqdm import tqdm

from models.ae import AESystem
from models.rtf import RTFSystem
from dataset import load_anndata

# 添加可视化相关导入
import matplotlib.pyplot as plt
import seaborn as sns

class CellTimeInference:
    """cellTime 推理引擎"""
    
    def __init__(
        self,
        ae_checkpoint: str,
        rtf_checkpoint: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化推理引擎
        
        Args:
            ae_checkpoint: AE 模型 checkpoint 路径
            rtf_checkpoint: RTF 模型 checkpoint 路径
            device: 设备（cuda/cpu）
        """
        self.device = device
        print(f"Using device: {device}")
        
        # 加载 AE 模型
        print(f"Loading AE model: {ae_checkpoint}")
        self.ae_system = AESystem.load_from_checkpoint(ae_checkpoint)
        self.ae_system.to(device)
        self.ae_system.eval()
        self.ae_system.freeze()
        
        # 加载 RTF 模型
        print(f"Loading RTF model: {rtf_checkpoint}")
        # 先加载 checkpoint（不传递 ae_encoder，避免 OmegaConf 序列化问题）
        checkpoint = torch.load(rtf_checkpoint, map_location=device)
        
        # 从 checkpoint 中提取配置和状态
        # 注意：需要传递 ae_encoder 给 RTFSystem 初始化
        # 但我们需要手动处理，避免 Lightning 的自动参数合并
        hparams = checkpoint.get('hyper_parameters', {})
        
        # 创建 RTFSystem 实例（传递 ae_encoder）
        self.rtf_system = RTFSystem(
            cfg=hparams,
            ae_encoder=self.ae_system.autoencoder.encoder
        )
        
        # 加载模型权重
        self.rtf_system.load_state_dict(checkpoint['state_dict'], strict=False)
        self.rtf_system.to(device)
        self.rtf_system.eval()
        self.rtf_system.freeze()
        
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码到潜空间
        
        Args:
            x: 基因表达 [B, n_genes]
        
        Returns:
            z: 潜空间表示 [B, latent_dim]
        """
        return self.ae_system.autoencoder.encode(x.to(self.device))
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜空间解码
        
        Args:
            z: 潜空间表示 [B, latent_dim]
        
        Returns:
            x: 基因表达 [B, n_genes]
        """
        return self.ae_system.autoencoder.decode(z.to(self.device))
    
    @torch.no_grad()
    def predict_latent(
        self,
        z_start: torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
        sample_steps: int = 50,
        cfg_scale: float = 2.0,
        cond: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        在潜空间进行时序预测
        
        Args:
            z_start: 起始潜空间 [B, latent_dim]
            t_start: 起始时间
            t_end: 目标时间
            sample_steps: 采样步数
            cfg_scale: CFG 强度
            cond: 条件信息
        
        Returns:
            trajectory: 潜空间轨迹 [z_0, z_1, ..., z_T]
        """
        z_start = z_start.to(self.device)
        batch_size = z_start.shape[0]
        
        # 准备时间条件
        t_start_tensor = torch.full((batch_size,), t_start, device=self.device)
        t_end_tensor = torch.full((batch_size,), t_end, device=self.device)
        
        # 准备条件信息
        if cond is None and self.rtf_system.cfg.model.use_cond:
            cond = torch.stack([t_start_tensor, t_end_tensor], dim=-1)
        
        null_cond = None
        if cond is not None:
            null_cond = torch.zeros_like(cond)
        
        # 根据 RTF 模式选择采样方法
        if self.rtf_system.cfg.model.mode == "direct":
            trajectory = self.rtf_system.model.sample(
                z_start,
                sample_steps=sample_steps,
                cond=cond,
                null_cond=null_cond,
                cfg_scale=cfg_scale,
            )
        else:  # inversion
            cond_start = torch.stack([t_start_tensor], dim=-1) if cond is not None else None
            cond_target = torch.stack([t_end_tensor], dim=-1) if cond is not None else None
            trajectory = self.rtf_system.model.sample(
                z_start,
                sample_steps=sample_steps,
                cond_start=cond_start,
                cond_target=cond_target,
                null_cond=null_cond,
                cfg_scale=cfg_scale,
            )
        
        return trajectory
    
    @torch.no_grad()
    def predict(
        self,
        x_start: torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
        sample_steps: int = 50,
        cfg_scale: float = 2.0,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        端到端预测：x_start -> x_end
        """
        # 编码到潜空间
        z_start = self.encode(x_start)
        
        # 在潜空间预测
        trajectory = self.predict_latent(
            z_start,
            t_start=t_start,
            t_end=t_end,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
        )
        
        # 解码最终状态
        z_end = trajectory[-1].to(self.device)
        x_end = self.decode(z_end)
        
        if return_trajectory:
            # 解码整个轨迹
            x_trajectory = [self.decode(z.to(self.device)).cpu() for z in trajectory]
            return x_end.cpu(), x_trajectory
        else:
            return x_end.cpu()


def predict(
    ae_checkpoint: str,
    rtf_checkpoint: str,
    input_data: str,
    output_path: str,
    target_time: float = 1.0,
    start_time: float = 0.0,
    sample_steps: int = 50,
    cfg_scale: float = 2.0,
    batch_size: int = 64,
    device: str = "auto",
):
    """单次时序预测"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    engine = CellTimeInference(ae_checkpoint, rtf_checkpoint, device)
    
    print(f"\nLoading input data: {input_data}")
    adata = load_anndata(input_data, verbose=True)
    
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    print(f"\nStart predicting ({start_time} -> {target_time})...")
    predictions = []
    
    n_batches = (len(X_tensor) + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc="Prediction progress"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(X_tensor))
        x_batch = X_tensor[batch_start:batch_end]
        
        x_pred = engine.predict(
            x_batch,
            t_start=start_time,
            t_end=target_time,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
        )
        
        predictions.append(x_pred.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    adata_pred = sc.AnnData(
        X=predictions,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    
    adata_pred.uns["prediction_info"] = {
        "start_time": start_time,
        "target_time": target_time,
        "sample_steps": sample_steps,
        "cfg_scale": cfg_scale,
        "ae_checkpoint": ae_checkpoint,
        "rtf_checkpoint": rtf_checkpoint,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_pred.write(output_path)
    
    print(f"\nPrediction completed!")
    print(f"Results saved to: {output_path}")


def predict_trajectory(
    ae_checkpoint: str,
    rtf_checkpoint: str,
    input_data: str,
    output_path: str,
    time_points: str = "[0.0, 0.5, 1.0, 1.5, 2.0]",
    sample_steps: int = 50,
    cfg_scale: float = 2.0,
    batch_size: int = 64,
    device: str = "auto",
):
    """多时间点轨迹预测"""
    import json
    time_points = json.loads(time_points)
    print(f"Predicting time points: {time_points}")
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    engine = CellTimeInference(ae_checkpoint, rtf_checkpoint, device)
    
    print(f"\nLoading input data: {input_data}")
    adata = load_anndata(input_data, verbose=True)
    
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    all_predictions = {}
    
    for t in time_points:
        print(f"\nPredicting time point t={t}...")
        predictions = []
        
        n_batches = (len(X_tensor) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc=f"t={t}"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(X_tensor))
            x_batch = X_tensor[batch_start:batch_end]
            
            x_pred = engine.predict(
                x_batch,
                t_start=time_points[0],
                t_end=t,
                sample_steps=sample_steps,
                cfg_scale=cfg_scale,
            )
            
            predictions.append(x_pred.numpy())
        
        all_predictions[f"t_{t}"] = np.concatenate(predictions, axis=0)
    
    last_time = time_points[-1]
    adata_pred = sc.AnnData(
        X=all_predictions[f"t_{last_time}"],
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    
    for t in time_points[:-1]:
        adata_pred.layers[f"t_{t}"] = all_predictions[f"t_{t}"]
    
    adata_pred.uns["trajectory_info"] = {
        "time_points": time_points,
        "sample_steps": sample_steps,
        "cfg_scale": cfg_scale,
        "ae_checkpoint": ae_checkpoint,
        "rtf_checkpoint": rtf_checkpoint,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_pred.write(output_path)
    
    print(f"\nTrajectory prediction completed!")
    print(f"Results saved to: {output_path}")


def encode_data(
    ae_checkpoint: str,
    input_data: str,
    output_path: str,
    batch_size: int = 64,
    device: str = "auto",
):
    """将单细胞数据编码到潜空间"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading AE model: {ae_checkpoint}")
    ae_system = AESystem.load_from_checkpoint(ae_checkpoint)
    ae_system.to(device)
    ae_system.eval()
    ae_system.freeze()
    
    print(f"\nLoading input data: {input_data}")
    adata = load_anndata(input_data, verbose=True)
    
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    print(f"\nEncoding to latent space...")
    latents = []
    
    n_batches = (len(X_tensor) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Encoding progress"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(X_tensor))
            x_batch = X_tensor[batch_start:batch_end].to(device)
            
            z = ae_system.autoencoder.encode(x_batch)
            latents.append(z.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)
    adata.obsm["X_latent"] = latents
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(output_path)
    
    print(f"\nEncoding completed!")
    print(f"Results saved to: {output_path}")
    print(f"Latent dimension: {latents.shape}")


def visualize(
    data_path: str,
    output_path: str = "visualization.png",
    n_pca: int = 50,
    color_by: str = "cell_type",
    basis: str = "umap",
):
    """
    可视化预测结果 (PCA/UMAP)
    
    Args:
        data_path: 包含预测结果的 .h5ad 文件路径
        output_path: 输出图像路径
        n_pca: PCA 组件数
        color_by: 着色依据的列名
        basis: 可视化基础 (umap/pca)
    """
    print(f"Loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)
    
    print("Preprocessing...")
    # 清洗数据：处理 NaN 和负值
    if isinstance(adata.X, np.ndarray):
        adata.X = np.nan_to_num(adata.X, nan=0.0)
        adata.X = np.maximum(adata.X, 0.0)
    
    # 基础预处理用于可视化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pca)
    
    if basis == "umap":
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
    
    print(f"Plotting {basis}...")
    
    # 设置绘图风格
    sc.set_figure_params(dpi=150, frameon=False, figsize=(6, 6))
    
    # 检查是否存在 trajectory_info，如果存在则尝试按时间着色
    if "trajectory_info" in adata.uns:
        # 这里的 adata 仅包含最后时刻的数据（因为 read_h5ad 默认读 X）
        # 如果我们想可视化轨迹，我们需要从 layers 重建
        print("Detected trajectory data. Reconstructing full trajectory for visualization...")
        
        traj_info = adata.uns["trajectory_info"]
        time_points = traj_info["time_points"]
        
        # 收集所有时间点的数据
        adatas = []
        # 注意：adata.X 实际上是 t_last (如 2.0)
        # layers 中存储了其他时间点
        
        # 为了统一，我们从 layers 提取（包括最后时刻，如果 infer 脚本逻辑是一致的）
        # 在 predict_trajectory 中：X 是 t_last，layers 包含 t_0 到 t_last-1
        
        for t in time_points[:-1]:
            key = f"t_{t}"
            if key in adata.layers:
                # 创建新的 AnnData
                ad_t = sc.AnnData(X=adata.layers[key].copy(), obs=adata.obs.copy())
                ad_t.obs["time"] = float(t)  # 使用 float 以便显示连续颜色
                adatas.append(ad_t)
        
        # 添加最后时刻 (X)
        last_t = time_points[-1]
        ad_last = sc.AnnData(X=adata.X.copy(), obs=adata.obs.copy())
        ad_last.obs["time"] = float(last_t)
        adatas.append(ad_last)
        
        # 合并
        adata_full = sc.concat(adatas)
        print(f"Full trajectory shape: {adata_full.shape}")
        
        # 重新预处理合并后的数据
        # 清洗数据
        if isinstance(adata_full.X, np.ndarray):
            adata_full.X = np.nan_to_num(adata_full.X, nan=0.0)
            adata_full.X = np.maximum(adata_full.X, 0.0)
            
        sc.pp.normalize_total(adata_full, target_sum=1e4)
        sc.pp.log1p(adata_full)
        sc.pp.highly_variable_genes(adata_full, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata_full = adata_full[:, adata_full.var.highly_variable]
        sc.pp.scale(adata_full, max_value=10)
        sc.tl.pca(adata_full, svd_solver='arpack', n_comps=n_pca)
        
        if basis == "umap":
            # 增加 n_neighbors 以更好地保留全局轨迹结构
            sc.pp.neighbors(adata_full, n_neighbors=30, n_pcs=40)
            sc.tl.umap(adata_full, min_dist=0.3)
            
            # 尝试同时绘制时间和细胞类型（如果存在）
            color_cols = ["time"]
            if "cell_type" in adata_full.obs.columns:
                color_cols.append("cell_type")
            
            # 使用 viridis 颜色映射显示时间连续性
            sc.pl.umap(adata_full, color=color_cols, show=False, color_map="viridis")
        else:
            sc.pl.pca(adata_full, color=["time"], show=False, color_map="viridis")
            
    else:
        # 创建画布
        if basis == "umap":
            sc.pl.umap(adata, color=color_by if color_by in adata.obs.columns else None, show=False)
        else:
            sc.pl.pca(adata, color=color_by if color_by in adata.obs.columns else None, show=False)
        
    # 保存图像
    plt.title(f"Visualization of Predictions ({basis.upper()})", fontsize=12)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire({
        "predict": predict,
        "predict_trajectory": predict_trajectory,
        "encode_data": encode_data,
        "visualize": visualize,
    })
