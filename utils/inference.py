#!/usr/bin/env python
"""
cellTime 推理脚本
使用训练好的 AE + RTF 模型进行单细胞时序预测

使用示例：
    # 基础推理
    python utils/inference.py predict \
        --ae_checkpoint=output/ae/checkpoints/last.ckpt \
        --rtf_checkpoint=output/rtf_direct_dit/checkpoints/last.ckpt \
        --input_data=data/test_cells.h5ad \
        --output_path=results/predictions.h5ad \
        --target_time=1.0
    
    # 批量时序预测
    python utils/inference.py predict_trajectory \
        --ae_checkpoint=output/ae/checkpoints/last.ckpt \
        --rtf_checkpoint=output/rtf_direct_dit/checkpoints/last.ckpt \
        --input_data=data/test_cells.h5ad \
        --output_path=results/trajectory.h5ad \
        --time_points="[0.0,0.5,1.0,1.5,2.0]"
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
        print(f"使用设备: {device}")
        
        # 加载 AE 模型
        print(f"加载 AE 模型: {ae_checkpoint}")
        self.ae_system = AESystem.load_from_checkpoint(ae_checkpoint)
        self.ae_system.to(device)
        self.ae_system.eval()
        self.ae_system.freeze()
        
        # 加载 RTF 模型
        print(f"加载 RTF 模型: {rtf_checkpoint}")
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
        
        print("模型加载完成！")
    
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
        
        Args:
            x_start: 起始基因表达 [B, n_genes]
            t_start: 起始时间
            t_end: 目标时间
            sample_steps: 采样步数
            cfg_scale: CFG 强度
            return_trajectory: 是否返回完整轨迹
        
        Returns:
            x_end: 预测的基因表达 [B, n_genes]
            或 (x_end, trajectory) 如果 return_trajectory=True
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
    """
    单次时序预测
    
    Args:
        ae_checkpoint: AE 模型路径
        rtf_checkpoint: RTF 模型路径
        input_data: 输入数据路径（.h5ad）
        output_path: 输出路径（.h5ad）
        target_time: 目标时间点
        start_time: 起始时间点
        sample_steps: 采样步数
        cfg_scale: CFG 强度
        batch_size: 批次大小
        device: 设备（auto/cuda/cpu）
    
    Example:
        python utils/inference.py predict \
            --ae_checkpoint=output/ae/checkpoints/last.ckpt \
            --rtf_checkpoint=output/rtf/checkpoints/last.ckpt \
            --input_data=data/cells.h5ad \
            --output_path=results/predictions.h5ad \
            --target_time=1.0
    """
    # 设备选择
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化推理引擎
    engine = CellTimeInference(ae_checkpoint, rtf_checkpoint, device)
    
    # 加载输入数据
    print(f"\n加载输入数据: {input_data}")
    adata = load_anndata(input_data, verbose=True)
    
    # 转换为 tensor
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # 批量预测
    print(f"\n开始预测（{start_time} -> {target_time}）...")
    predictions = []
    
    n_batches = (len(X_tensor) + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc="预测进度"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(X_tensor))
        x_batch = X_tensor[batch_start:batch_end]
        
        # 预测
        x_pred = engine.predict(
            x_batch,
            t_start=start_time,
            t_end=target_time,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
        )
        
        predictions.append(x_pred.numpy())
    
    # 合并结果
    predictions = np.concatenate(predictions, axis=0)
    
    # 创建结果 AnnData
    adata_pred = sc.AnnData(
        X=predictions,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    
    # 添加元数据
    adata_pred.uns["prediction_info"] = {
        "start_time": start_time,
        "target_time": target_time,
        "sample_steps": sample_steps,
        "cfg_scale": cfg_scale,
        "ae_checkpoint": ae_checkpoint,
        "rtf_checkpoint": rtf_checkpoint,
    }
    
    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_pred.write(output_path)
    
    print(f"\n预测完成！")
    print(f"结果保存在: {output_path}")
    print(f"预测细胞数: {len(predictions)}")
    print(f"基因数: {predictions.shape[1]}")


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
    """
    多时间点轨迹预测
    
    Args:
        ae_checkpoint: AE 模型路径
        rtf_checkpoint: RTF 模型路径
        input_data: 输入数据路径
        output_path: 输出路径（.h5ad）
        time_points: 时间点列表（JSON 格式字符串）
        sample_steps: 采样步数
        cfg_scale: CFG 强度
        batch_size: 批次大小
        device: 设备
    
    Example:
        python utils/inference.py predict_trajectory \
            --ae_checkpoint=output/ae/checkpoints/last.ckpt \
            --rtf_checkpoint=output/rtf/checkpoints/last.ckpt \
            --input_data=data/cells.h5ad \
            --output_path=results/trajectory.h5ad \
            --time_points="[0.0,0.5,1.0,1.5,2.0]"
    """
    import json
    
    # 解析时间点
    time_points = json.loads(time_points)
    print(f"预测时间点: {time_points}")
    
    # 设备选择
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化推理引擎
    engine = CellTimeInference(ae_checkpoint, rtf_checkpoint, device)
    
    # 加载输入数据
    print(f"\n加载输入数据: {input_data}")
    adata = load_anndata(input_data, verbose=True)
    
    # 转换为 tensor
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # 预测每个时间点
    all_predictions = {}
    
    for t in time_points:
        print(f"\n预测时间点 t={t}...")
        predictions = []
        
        n_batches = (len(X_tensor) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc=f"t={t}"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(X_tensor))
            x_batch = X_tensor[batch_start:batch_end]
            
            # 预测
            x_pred = engine.predict(
                x_batch,
                t_start=time_points[0],
                t_end=t,
                sample_steps=sample_steps,
                cfg_scale=cfg_scale,
            )
            
            predictions.append(x_pred.numpy())
        
        all_predictions[f"t_{t}"] = np.concatenate(predictions, axis=0)
    
    # 创建结果 AnnData（使用最后一个时间点作为主数据）
    last_time = time_points[-1]
    adata_pred = sc.AnnData(
        X=all_predictions[f"t_{last_time}"],
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    
    # 将其他时间点作为 layers 保存
    for t in time_points[:-1]:
        adata_pred.layers[f"t_{t}"] = all_predictions[f"t_{t}"]
    
    # 添加元数据
    adata_pred.uns["trajectory_info"] = {
        "time_points": time_points,
        "sample_steps": sample_steps,
        "cfg_scale": cfg_scale,
        "ae_checkpoint": ae_checkpoint,
        "rtf_checkpoint": rtf_checkpoint,
    }
    
    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_pred.write(output_path)
    
    print(f"\n轨迹预测完成！")
    print(f"结果保存在: {output_path}")
    print(f"预测细胞数: {len(adata_pred)}")
    print(f"时间点数: {len(time_points)}")
    print(f"数据结构:")
    print(f"  - X: 时间点 t={last_time} 的预测")
    for t in time_points[:-1]:
        print(f"  - layers['t_{t}']: 时间点 t={t} 的预测")


def encode_data(
    ae_checkpoint: str,
    input_data: str,
    output_path: str,
    batch_size: int = 64,
    device: str = "auto",
):
    """
    将单细胞数据编码到潜空间
    
    Args:
        ae_checkpoint: AE 模型路径
        input_data: 输入数据路径
        output_path: 输出路径（.h5ad）
        batch_size: 批次大小
        device: 设备
    
    Example:
        python utils/inference.py encode_data \
            --ae_checkpoint=output/ae/checkpoints/last.ckpt \
            --input_data=data/cells.h5ad \
            --output_path=results/latent.h5ad
    """
    # 设备选择
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载 AE 模型
    print(f"加载 AE 模型: {ae_checkpoint}")
    ae_system = AESystem.load_from_checkpoint(ae_checkpoint)
    ae_system.to(device)
    ae_system.eval()
    ae_system.freeze()
    
    # 加载输入数据
    print(f"\n加载输入数据: {input_data}")
    adata = load_anndata(input_data, verbose=True)
    
    # 转换为 tensor
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # 批量编码
    print(f"\n编码到潜空间...")
    latents = []
    
    n_batches = (len(X_tensor) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="编码进度"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(X_tensor))
            x_batch = X_tensor[batch_start:batch_end].to(device)
            
            # 编码
            z = ae_system.autoencoder.encode(x_batch)
            latents.append(z.cpu().numpy())
    
    # 合并结果
    latents = np.concatenate(latents, axis=0)
    
    # 保存为 AnnData（潜空间作为 obsm）
    adata.obsm["X_latent"] = latents
    
    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(output_path)
    
    print(f"\n编码完成！")
    print(f"结果保存在: {output_path}")
    print(f"潜空间维度: {latents.shape}")


if __name__ == "__main__":
    fire.Fire({
        "predict": predict,
        "predict_trajectory": predict_trajectory,
        "encode_data": encode_data,
    })







