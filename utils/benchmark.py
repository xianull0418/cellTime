#!/usr/bin/env python
"""
cellTime Benchmark 评估脚本 (简化版)
核心功能：
1. 评估 AE 重建质量
2. 评估 RTF 时序预测质量 (Current -> Next)
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
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional, Dict
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from utils.inference import CellTimeInference
from models.ae import AESystem
from dataset import load_anndata


class CellTimeBenchmark:
    def __init__(
        self,
        ae_checkpoint: str,
        rtf_checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        print(f"Device: {device}")
        
        # 加载 AE
        print(f"Loading AE: {ae_checkpoint}")
        self.ae_system = AESystem.load_from_checkpoint(ae_checkpoint)
        self.ae_system.to(device).eval().freeze()
        
        # 加载 RTF
        self.rtf_system = None
        if rtf_checkpoint:
            print(f"Loading RTF: {rtf_checkpoint}")
            self.inference_engine = CellTimeInference(ae_checkpoint, rtf_checkpoint, device)
            self.rtf_system = self.inference_engine.rtf_system
        print("Models loaded.\n")

    @torch.no_grad()
    def evaluate_ae(self, adata: sc.AnnData, batch_size: int = 64, n_top_genes: int = 2000) -> Dict:
        """评估 AE 重建质量"""
        print(">>> Evaluating AE Reconstruction...")
        
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        
        # --- 数据预处理检查 ---
        print(f"[Data Check] Raw Input: min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
        
        # 1. 检查是否需要 log1p
        if X.max() > 20:
            print("警告：数据最大值 > 20，检测为 Raw Counts。自动执行 log1p...")
            X = np.log1p(X)
            print(f"[Data Check] After log1p: min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
            
        # 2. 检查维度匹配 (假设模型需要特定维度，这里只打印警告)
        # 注意：通常需要 subset 到模型训练时的高变基因。如果直接用全部基因，维度对不上会报错。
        # 这里假设输入数据已经只有高变基因了，或者维度正好对上。
        # -------------------
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # 批量重建
        reconstructions = []
        n_batches = (len(X_tensor) + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Reconstructing"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(X_tensor))
            x_batch = X_tensor[batch_start:batch_end].to(self.device)
            
            z = self.ae_system.autoencoder.encode(x_batch)
            x_recon = self.ae_system.autoencoder.decode(z)
            reconstructions.append(x_recon.cpu().numpy())
        
        X_recon = np.concatenate(reconstructions, axis=0)
        
        # 计算指标
        metrics = {}
        metrics['mse'] = mean_squared_error(X, X_recon)
        
        # 基因相关性
        gene_corrs = [pearsonr(X[:, i], X_recon[:, i])[0] 
                     for i in range(X.shape[1]) if np.std(X[:, i]) > 0]
        metrics['mean_gene_corr'] = np.mean(gene_corrs)
        
        # 细胞相关性
        cell_corrs = [pearsonr(X[i], X_recon[i])[0] 
                     for i in range(X.shape[0]) if np.std(X[i]) > 0]
        metrics['mean_cell_corr'] = np.mean(cell_corrs)
        
        # 打印简报
        print(f"AE MSE: {metrics['mse']:.4f}")
        print(f"Gene Corr: {metrics['mean_gene_corr']:.4f}")
        print(f"Cell Corr: {metrics['mean_cell_corr']:.4f}\n")
        
        # 保存用于绘图的数据
        metrics['gene_corrs'] = gene_corrs
        metrics['cell_corrs'] = cell_corrs
        
        return metrics

    @torch.no_grad()
    def evaluate_rtf(
        self, 
        adata: sc.AnnData, 
        batch_size: int = 64, 
        sample_steps: int = 50, 
        cfg_scale: float = 2.0
    ) -> Dict:
        """评估 RTF 时序预测 (Current -> Next)"""
        if not self.rtf_system:
            raise ValueError("RTF model not loaded.")
            
        print(">>> Evaluating RTF Temporal Prediction...")
        
        # 准备数据：Current -> Next
        next_cell_col = "next_cell_id"
        if next_cell_col not in adata.obs.columns:
            raise ValueError(f"Column '{next_cell_col}' missing.")
            
        valid_mask = ~pd.isna(adata.obs[next_cell_col])
        valid_indices = np.where(valid_mask)[0]
        
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X_curr = X[valid_indices]
        
        # 获取真实的 Next Cell
        next_indices = adata.obs.iloc[valid_indices][next_cell_col].values.astype(int)
        X_next_true = X[next_indices]
        
        print(f"Valid pairs: {len(X_curr)}")
        
        # 批量预测 (统一 t=0 -> t=1)
        predictions = []
        n_batches = (len(X_curr) + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Predicting"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(X_curr))
            x_batch = torch.tensor(X_curr[batch_start:batch_end], dtype=torch.float32)
            
            # 核心预测逻辑：t=0 -> t=1
            x_pred = self.inference_engine.predict(
                x_batch, t_start=0.0, t_end=1.0, 
                sample_steps=sample_steps, cfg_scale=cfg_scale
            )
            predictions.append(x_pred.numpy())
            
        X_next_pred = np.concatenate(predictions, axis=0)
        
        # --- 调试信息：检查数值范围 ---
        print(f"\n[Debug Statistics]")
        print(f"True Data:  min={X_next_true.min():.4f}, max={X_next_true.max():.4f}, mean={X_next_true.mean():.4f}")
        print(f"Pred Data:  min={X_next_pred.min():.4f}, max={X_next_pred.max():.4f}, mean={X_next_pred.mean():.4f}")
        print(f"Pred Std:   {X_next_pred.std():.4f}")
        if X_next_pred.std() < 1e-6:
            print("警告：预测结果几乎为常数，模型可能未正常工作！")
        # ---------------------------
        
        # 计算指标
        metrics = {}
        metrics['mse'] = mean_squared_error(X_next_true, X_next_pred)
        
        # 基因相关性 (Predicted vs Ground Truth)
        gene_corrs = []
        for i in range(X_next_true.shape[1]):
            # 只有当真实值和预测值都有波动时，才能计算相关性
            if np.std(X_next_true[:, i]) > 1e-6 and np.std(X_next_pred[:, i]) > 1e-6:
                gene_corrs.append(pearsonr(X_next_true[:, i], X_next_pred[:, i])[0])
        
        if len(gene_corrs) > 0:
            metrics['mean_gene_corr'] = np.mean(gene_corrs)
        else:
            metrics['mean_gene_corr'] = 0.0
        
        # 细胞相关性
        cell_corrs = []
        for i in range(X_next_true.shape[0]):
            if np.std(X_next_true[i]) > 1e-6 and np.std(X_next_pred[i]) > 1e-6:
                cell_corrs.append(pearsonr(X_next_true[i], X_next_pred[i])[0])
        
        if len(cell_corrs) > 0:
            metrics['mean_cell_corr'] = np.mean(cell_corrs)
        else:
            metrics['mean_cell_corr'] = 0.0
        
        # 基因级别平均表达相关性 (Squidiff 核心指标)
        gene_mean_true = np.mean(X_next_true, axis=0)
        gene_mean_pred = np.mean(X_next_pred, axis=0)
        metrics['squidiff_corr'] = pearsonr(gene_mean_true, gene_mean_pred)[0]
        
        # 高变基因 (Top 500) 的 Squidiff 相关性
        gene_vars = np.var(X_next_true, axis=0)
        top_indices = np.argsort(gene_vars)[-500:]
        metrics['squidiff_hvg_corr'] = pearsonr(
            gene_mean_true[top_indices], gene_mean_pred[top_indices]
        )[0]
        
        print(f"RTF MSE: {metrics['mse']:.4f}")
        print(f"Mean Gene Corr: {metrics['mean_gene_corr']:.4f}")
        print(f"Mean Cell Corr: {metrics['mean_cell_corr']:.4f}")
        print(f"Squidiff Score (All Genes): {metrics['squidiff_corr']:.4f}")
        print(f"Squidiff Score (HVG 500): {metrics['squidiff_hvg_corr']:.4f}\n")
        
        # 保存数据用于绘图
        metrics['gene_corrs'] = gene_corrs
        metrics['cell_corrs'] = cell_corrs
        metrics['gene_mean_true'] = gene_mean_true
        metrics['gene_mean_pred'] = gene_mean_pred
        metrics['hvg_indices'] = top_indices
        
        return metrics

    def plot_results(self, ae_metrics: Optional[Dict], rtf_metrics: Optional[Dict], output_dir: Path):
        """统一绘图"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. AE 图表
        if ae_metrics:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ae_metrics['gene_corrs'], bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f"AE Reconstruction (Gene Corr)\nMean: {ae_metrics['mean_gene_corr']:.3f}")
            ax.set_xlabel("Pearson Correlation")
            plt.tight_layout()
            plt.savefig(output_dir / "ae_eval.png", dpi=150)
            plt.close()
            
        # 2. RTF 图表 (Squidiff Style)
        if rtf_metrics:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # All Genes
            ax = axes[0]
            x = rtf_metrics['gene_mean_pred']
            y = rtf_metrics['gene_mean_true']
            ax.scatter(x, y, s=10, alpha=0.5)
            
            min_v, max_v = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'r--')
            ax.set_title(f"All Genes Prediction\nPearson R = {rtf_metrics['squidiff_corr']:.3f}")
            ax.set_xlabel("Predicted Expression")
            ax.set_ylabel("True Expression")
            
            # HVG
            ax = axes[1]
            idx = rtf_metrics['hvg_indices']
            x_hvg = rtf_metrics['gene_mean_pred'][idx]
            y_hvg = rtf_metrics['gene_mean_true'][idx]
            ax.scatter(x_hvg, y_hvg, s=10, c='orange', alpha=0.5)
            
            min_v, max_v = min(x_hvg.min(), y_hvg.min()), max(x_hvg.max(), y_hvg.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'r--')
            ax.set_title(f"Top 500 HVG Prediction\nPearson R = {rtf_metrics['squidiff_hvg_corr']:.3f}")
            ax.set_xlabel("Predicted Expression")
            ax.set_ylabel("True Expression")
            
            plt.tight_layout()
            plt.savefig(output_dir / "rtf_squidiff_eval.png", dpi=150)
            plt.close()
            print(f"Plots saved to {output_dir}")

# CLI 接口
def run(
    mode: str = "full",
    ae_checkpoint: str = "",
    rtf_checkpoint: str = "",
    input_data: str = "",
    output_dir: str = "output/benchmark",
    batch_size: int = 128,
    sample_steps: int = 50,
    cfg_scale: float = 2.0,
    device: str = "auto"
):
    """
    简化版 Benchmark 入口
    mode: full | ae | rtf
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"Mode: {mode} | Input: {input_data}")
    
    # 加载数据
    adata = load_anndata(input_data)
    bm = CellTimeBenchmark(ae_checkpoint, rtf_checkpoint if mode != "ae" else None, device)
    
    ae_res, rtf_res = None, None
    
    if mode in ["full", "ae"]:
        ae_res = bm.evaluate_ae(adata, batch_size)
        
    if mode in ["full", "rtf"]:
        rtf_res = bm.evaluate_rtf(adata, batch_size, sample_steps, cfg_scale)
        
    bm.plot_results(ae_res, rtf_res, output_dir)
    
    # 保存 JSON
    results = {}
    if ae_res:
        results['ae'] = {k: float(v) for k, v in ae_res.items() if isinstance(v, (int, float, np.float32, np.float64))}
    if rtf_res:
        results['rtf'] = {k: float(v) for k, v in rtf_res.items() if isinstance(v, (int, float, np.float32, np.float64))}
        
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    fire.Fire(run)
