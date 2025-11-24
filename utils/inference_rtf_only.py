#!/usr/bin/env python
"""
RTF Only 推理脚本
使用 RTF Only 模型直接在基因空间进行单细胞时序预测
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
import pandas as pd

from models.rtf_only import RTFOnlySystem
from dataset import load_anndata

class RTFOnlyInference:
    """RTF Only 推理引擎"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        print(f"Using device: {device}")
        
        print(f"Loading RTF Only model: {checkpoint_path}")
        self.system = RTFOnlySystem.load_from_checkpoint(checkpoint_path)
        self.system.to(device)
        self.system.eval()
        self.system.freeze()
        
        print(f"Model loaded! n_genes: {self.system.cfg.model.n_genes}")

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
        端到端预测：x_start -> x_end (直接在基因空间)
        """
        x_start = x_start.to(self.device)
        batch_size = x_start.shape[0]
        
        # 准备条件
        t_start_tensor = torch.full((batch_size,), t_start, device=self.device)
        t_end_tensor = torch.full((batch_size,), t_end, device=self.device)
        
        cond = None
        null_cond = None
        
        if self.system.cfg.model.use_cond:
            cond = torch.stack([t_start_tensor, t_end_tensor], dim=-1)
            null_cond = torch.zeros_like(cond)
            
        normalize = getattr(self.system.cfg.model, 'normalize', False)
        
        # 采样
        trajectory = self.system.model.sample(
            x_start,
            sample_steps=sample_steps,
            cond=cond,
            null_cond=null_cond,
            cfg_scale=cfg_scale,
            normalize=normalize
        )
        
        x_end = trajectory[-1].cpu()
        
        if return_trajectory:
            x_traj = [t.cpu() for t in trajectory]
            return x_end, x_traj
        else:
            return x_end

def predict(
    checkpoint: str,
    input_data: str,
    output_path: str,
    target_genes_path: str = "gene_order.tsv",
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
    
    engine = RTFOnlyInference(checkpoint, device)
    
    # 加载目标基因列表
    print(f"Loading target genes from: {target_genes_path}")
    with open(target_genes_path, 'r') as f:
        target_genes = [line.strip() for line in f if line.strip()]
    
    print(f"\nLoading input data: {input_data}")
    # 强制使用 target_genes 对齐
    adata = load_anndata(input_data, target_genes=target_genes, verbose=True)
    
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # 确保维度匹配
    if X_tensor.shape[1] != engine.system.cfg.model.n_genes:
         raise ValueError(
             f"Input data genes ({X_tensor.shape[1]}) != Model genes ({engine.system.cfg.model.n_genes}).\n"
             f"Please re-train the model with the correct gene count (using gene_order.tsv).\n"
             f"Run: bash rtf_only.sh"
         )
    
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
        "checkpoint": checkpoint,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_pred.write(output_path)
    
    print(f"\nPrediction completed!")
    print(f"Results saved to: {output_path}")

def predict_trajectory(
    checkpoint: str,
    input_data: str,
    output_path: str,
    target_genes_path: str = "gene_order.tsv",
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
    
    engine = RTFOnlyInference(checkpoint, device)
    
    print(f"Loading target genes from: {target_genes_path}")
    with open(target_genes_path, 'r') as f:
        target_genes = [line.strip() for line in f if line.strip()]
    
    print(f"\nLoading input data: {input_data}")
    adata = load_anndata(input_data, target_genes=target_genes, verbose=True)
    
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # 确保维度匹配
    if X_tensor.shape[1] != engine.system.cfg.model.n_genes:
         raise ValueError(
             f"Input data genes ({X_tensor.shape[1]}) != Model genes ({engine.system.cfg.model.n_genes}).\n"
             f"Please re-train the model with the correct gene count (using gene_order.tsv).\n"
             f"Run: bash rtf_only.sh"
         )
    
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
        "checkpoint": checkpoint,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_pred.write(output_path)
    
    print(f"\nTrajectory prediction completed!")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    fire.Fire({
        "predict": predict,
        "predict_trajectory": predict_trajectory,
    })
