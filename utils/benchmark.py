#!/usr/bin/env python
"""
cellTime Benchmark 评估脚本
评估 AE 重建质量和 RTF 时序预测质量

使用示例：
    # 完整评估（AE + RTF）
    python utils/benchmark.py full \
        --ae_checkpoint=output/ae/checkpoints/last.ckpt \
        --rtf_checkpoint=output/rtf_direct_dit/checkpoints/last.ckpt \
        --input_data=data/test.h5ad \
        --output_dir=results/benchmark
    
    # 仅评估 AE
    python utils/benchmark.py ae \
        --ae_checkpoint=output/ae/checkpoints/last.ckpt \
        --input_data=data/test.h5ad \
        --output_dir=results/benchmark
    
    # 仅评估 RTF
    python utils/benchmark.py rtf \
        --ae_checkpoint=output/ae/checkpoints/last.ckpt \
        --rtf_checkpoint=output/rtf_direct_dit/checkpoints/last.ckpt \
        --input_data=data/test.h5ad \
        --output_dir=results/benchmark
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional, Dict, List
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

from utils.inference import CellTimeInference
from models.ae import AESystem
from dataset import load_anndata


class CellTimeBenchmark:
    """cellTime 性能评估"""
    
    def __init__(
        self,
        ae_checkpoint: str,
        rtf_checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化评估器
        
        Args:
            ae_checkpoint: AE 模型路径
            rtf_checkpoint: RTF 模型路径（可选）
            device: 设备
        """
        self.device = device
        print(f"使用设备: {device}\n")
        
        # 加载 AE 模型
        print(f"加载 AE 模型: {ae_checkpoint}")
        self.ae_system = AESystem.load_from_checkpoint(ae_checkpoint)
        self.ae_system.to(device)
        self.ae_system.eval()
        self.ae_system.freeze()
        
        # 加载 RTF 模型（如果提供）
        self.rtf_system = None
        if rtf_checkpoint:
            print(f"加载 RTF 模型: {rtf_checkpoint}")
            self.inference_engine = CellTimeInference(
                ae_checkpoint, rtf_checkpoint, device
            )
            self.rtf_system = self.inference_engine.rtf_system
        
        print("模型加载完成！\n")
    
    # ========================================================================
    # AE 重建质量评估
    # ========================================================================
    
    @torch.no_grad()
    def evaluate_ae_reconstruction(
        self,
        adata: sc.AnnData,
        batch_size: int = 64,
        n_top_genes: int = 2000,
    ) -> Dict:
        """
        评估 AE 重建质量
        
        Args:
            adata: 测试数据
            batch_size: 批次大小
            n_top_genes: 高变基因数量
        
        Returns:
            评估指标字典
        """
        print("=" * 60)
        print("评估 AE 重建质量")
        print("=" * 60)
        
        # 准备数据
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # 批量重建
        print("进行重建...")
        reconstructions = []
        latents = []
        
        n_batches = (len(X_tensor) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc="重建进度"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(X_tensor))
            x_batch = X_tensor[batch_start:batch_end].to(self.device)
            
            # 编码-解码
            z = self.ae_system.autoencoder.encode(x_batch)
            x_recon = self.ae_system.autoencoder.decode(z)
            
            reconstructions.append(x_recon.cpu().numpy())
            latents.append(z.cpu().numpy())
        
        X_recon = np.concatenate(reconstructions, axis=0)
        Z = np.concatenate(latents, axis=0)
        
        # 计算评估指标
        print("\n计算评估指标...")
        metrics = {}
        
        # 1. 整体重建误差
        metrics['mse'] = mean_squared_error(X, X_recon)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(X, X_recon)
        
        # 2. 每个基因的相关性
        gene_correlations = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 0 and np.std(X_recon[:, i]) > 0:
                corr, _ = pearsonr(X[:, i], X_recon[:, i])
                gene_correlations.append(corr)
        
        metrics['mean_gene_correlation'] = np.mean(gene_correlations)
        metrics['median_gene_correlation'] = np.median(gene_correlations)
        
        # 3. 每个细胞的相关性
        cell_correlations = []
        for i in range(X.shape[0]):
            if np.std(X[i, :]) > 0 and np.std(X_recon[i, :]) > 0:
                corr, _ = pearsonr(X[i, :], X_recon[i, :])
                cell_correlations.append(corr)
        
        metrics['mean_cell_correlation'] = np.mean(cell_correlations)
        metrics['median_cell_correlation'] = np.median(cell_correlations)
        
        # 4. 高变基因重建质量
        if n_top_genes > 0:
            # 识别高变基因
            gene_vars = np.var(X, axis=0)
            top_gene_indices = np.argsort(gene_vars)[-n_top_genes:]
            
            X_hvg = X[:, top_gene_indices]
            X_recon_hvg = X_recon[:, top_gene_indices]
            
            metrics['hvg_mse'] = mean_squared_error(X_hvg, X_recon_hvg)
            metrics['hvg_rmse'] = np.sqrt(metrics['hvg_mse'])
            
            hvg_correlations = []
            for i in top_gene_indices:
                if np.std(X[:, i]) > 0 and np.std(X_recon[:, i]) > 0:
                    corr, _ = pearsonr(X[:, i], X_recon[:, i])
                    hvg_correlations.append(corr)
            
            metrics['hvg_mean_correlation'] = np.mean(hvg_correlations)
        
        # 5. 潜空间统计
        metrics['latent_dim'] = Z.shape[1]
        metrics['latent_mean'] = float(np.mean(Z))
        metrics['latent_std'] = float(np.std(Z))
        
        # 打印结果
        print("\n" + "=" * 60)
        print("AE 重建质量评估结果")
        print("=" * 60)
        print(f"整体重建误差:")
        print(f"  MSE:  {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"\n基因级别相关性:")
        print(f"  平均: {metrics['mean_gene_correlation']:.4f}")
        print(f"  中位: {metrics['median_gene_correlation']:.4f}")
        print(f"\n细胞级别相关性:")
        print(f"  平均: {metrics['mean_cell_correlation']:.4f}")
        print(f"  中位: {metrics['median_cell_correlation']:.4f}")
        
        if n_top_genes > 0:
            print(f"\n高变基因 (top {n_top_genes}) 重建质量:")
            print(f"  MSE:  {metrics['hvg_mse']:.6f}")
            print(f"  RMSE: {metrics['hvg_rmse']:.6f}")
            print(f"  平均相关性: {metrics['hvg_mean_correlation']:.4f}")
        
        print(f"\n潜空间统计:")
        print(f"  维度: {metrics['latent_dim']}")
        print(f"  均值: {metrics['latent_mean']:.4f}")
        print(f"  标准差: {metrics['latent_std']:.4f}")
        print("=" * 60 + "\n")
        
        # 保存详细数据
        metrics['gene_correlations'] = gene_correlations
        metrics['cell_correlations'] = cell_correlations
        metrics['reconstructions'] = X_recon
        metrics['latents'] = Z
        
        return metrics
    
    # ========================================================================
    # RTF 时序预测质量评估
    # ========================================================================
    
    @torch.no_grad()
    def evaluate_rtf_temporal_prediction(
        self,
        adata: sc.AnnData,
        time_col: str = "time",
        next_cell_col: str = "next_cell_id",
        batch_size: int = 64,
        sample_steps: int = 50,
        cfg_scale: float = 2.0,
    ) -> Dict:
        """
        评估 RTF 时序预测质量
        
        利用数据集中的 next_cell_id 进行真实值对比
        
        Args:
            adata: 测试数据（必须包含 next_cell_id）
            time_col: 时间列名
            next_cell_col: 下一个细胞 ID 列名
            batch_size: 批次大小
            sample_steps: 采样步数
            cfg_scale: CFG 强度
        
        Returns:
            评估指标字典
        """
        if self.rtf_system is None:
            raise ValueError("未加载 RTF 模型，无法进行时序预测评估")
        
        print("=" * 60)
        print("评估 RTF 时序预测质量")
        print("=" * 60)
        
        # 检查必要列
        if next_cell_col not in adata.obs.columns:
            raise ValueError(f"数据集缺少 {next_cell_col} 列")
        if time_col not in adata.obs.columns:
            raise ValueError(f"数据集缺少 {time_col} 列")
        
        # 筛选有效的时序对
        valid_mask = ~pd.isna(adata.obs[next_cell_col])
        valid_indices = np.where(valid_mask)[0]
        
        print(f"有效时序对: {len(valid_indices)} / {len(adata)}")
        
        if len(valid_indices) == 0:
            raise ValueError("没有有效的时序对进行评估")
        
        # 准备数据
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        
        # 提取当前细胞和下一个细胞
        X_current = X[valid_indices]
        
        # 获取下一个细胞的索引（使用 pandas 索引）
        next_cell_ids = adata.obs.iloc[valid_indices][next_cell_col].values
        
        # 将 next_cell_id 转换为整数索引
        # 假设 next_cell_id 是 pandas 的位置索引
        next_indices = next_cell_ids.astype(int)
        X_next_true = X[next_indices]
        
        # 获取时间信息
        t_current = adata.obs.iloc[valid_indices][time_col].values
        t_next = adata.obs.iloc[next_indices][time_col].values
        
        print(f"当前时间范围: [{t_current.min():.2f}, {t_current.max():.2f}]")
        print(f"目标时间范围: [{t_next.min():.2f}, {t_next.max():.2f}]")
        
        # 批量预测
        print("\n进行时序预测...")
        predictions = []
        
        n_batches = (len(X_current) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc="预测进度"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(X_current))
            
            x_batch = torch.tensor(X_current[batch_start:batch_end], dtype=torch.float32)
            t_start_batch = t_current[batch_start:batch_end]
            t_end_batch = t_next[batch_start:batch_end]
            
            # 对每个样本单独预测（因为时间点可能不同）
            batch_predictions = []
            for j in range(len(x_batch)):
                x_pred = self.inference_engine.predict(
                    x_batch[j:j+1],
                    t_start=float(t_start_batch[j]),
                    t_end=float(t_end_batch[j]),
                    sample_steps=sample_steps,
                    cfg_scale=cfg_scale,
                )
                batch_predictions.append(x_pred.numpy())
            
            predictions.extend(batch_predictions)
        
        X_next_pred = np.concatenate(predictions, axis=0)
        
        # 计算评估指标
        print("\n计算评估指标...")
        metrics = {}
        
        # 1. 预测误差
        metrics['mse'] = mean_squared_error(X_next_true, X_next_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(X_next_true, X_next_pred)
        
        # 2. 每个基因的预测相关性
        gene_correlations = []
        for i in range(X_next_true.shape[1]):
            if np.std(X_next_true[:, i]) > 0 and np.std(X_next_pred[:, i]) > 0:
                corr, _ = pearsonr(X_next_true[:, i], X_next_pred[:, i])
                gene_correlations.append(corr)
        
        metrics['mean_gene_correlation'] = np.mean(gene_correlations)
        metrics['median_gene_correlation'] = np.median(gene_correlations)
        
        # 3. 每个细胞的预测相关性
        cell_correlations = []
        for i in range(X_next_true.shape[0]):
            if np.std(X_next_true[i, :]) > 0 and np.std(X_next_pred[i, :]) > 0:
                corr, _ = pearsonr(X_next_true[i, :], X_next_pred[i, :])
                cell_correlations.append(corr)
        
        metrics['mean_cell_correlation'] = np.mean(cell_correlations)
        metrics['median_cell_correlation'] = np.median(cell_correlations)
        
        # 4. Cosine 相似度
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_sims = []
        for i in range(X_next_true.shape[0]):
            sim = cosine_similarity(
                X_next_true[i:i+1],
                X_next_pred[i:i+1]
            )[0, 0]
            cosine_sims.append(sim)
        
        metrics['mean_cosine_similarity'] = np.mean(cosine_sims)
        metrics['median_cosine_similarity'] = np.median(cosine_sims)
        
        # 5. 时间差异统计
        time_diffs = t_next - t_current
        metrics['mean_time_diff'] = float(np.mean(time_diffs))
        metrics['std_time_diff'] = float(np.std(time_diffs))
        
        # 打印结果
        print("\n" + "=" * 60)
        print("RTF 时序预测评估结果")
        print("=" * 60)
        print(f"预测误差:")
        print(f"  MSE:  {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"\n基因级别预测相关性:")
        print(f"  平均: {metrics['mean_gene_correlation']:.4f}")
        print(f"  中位: {metrics['median_gene_correlation']:.4f}")
        print(f"\n细胞级别预测相关性:")
        print(f"  平均: {metrics['mean_cell_correlation']:.4f}")
        print(f"  中位: {metrics['median_cell_correlation']:.4f}")
        print(f"\nCosine 相似度:")
        print(f"  平均: {metrics['mean_cosine_similarity']:.4f}")
        print(f"  中位: {metrics['median_cosine_similarity']:.4f}")
        print(f"\n时间差异统计:")
        print(f"  平均: {metrics['mean_time_diff']:.4f}")
        print(f"  标准差: {metrics['std_time_diff']:.4f}")
        print("=" * 60 + "\n")
        
        # 保存详细数据
        metrics['gene_correlations'] = gene_correlations
        metrics['cell_correlations'] = cell_correlations
        metrics['cosine_similarities'] = cosine_sims
        metrics['time_diffs'] = time_diffs.tolist()
        metrics['predictions'] = X_next_pred
        metrics['ground_truth'] = X_next_true
        
        return metrics
    
    # ========================================================================
    # 可视化
    # ========================================================================
    
    def plot_ae_results(self, metrics: Dict, output_dir: Path):
        """绘制 AE 评估结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("生成 AE 评估可视化...")
        
        # 1. 基因相关性分布
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(metrics['gene_correlations'], bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(metrics['mean_gene_correlation'], color='red', 
                       linestyle='--', label=f"均值: {metrics['mean_gene_correlation']:.3f}")
        axes[0].set_xlabel("Pearson Correlation")
        axes[0].set_ylabel("基因数量")
        axes[0].set_title("基因重建相关性分布")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 2. 细胞相关性分布
        axes[1].hist(metrics['cell_correlations'], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[1].axvline(metrics['mean_cell_correlation'], color='red',
                       linestyle='--', label=f"均值: {metrics['mean_cell_correlation']:.3f}")
        axes[1].set_xlabel("Pearson Correlation")
        axes[1].set_ylabel("细胞数量")
        axes[1].set_title("细胞重建相关性分布")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "ae_correlation_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  保存: {output_dir / 'ae_correlation_distributions.png'}")
    
    def plot_rtf_results(self, metrics: Dict, output_dir: Path):
        """绘制 RTF 评估结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("生成 RTF 评估可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 基因预测相关性分布
        axes[0, 0].hist(metrics['gene_correlations'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(metrics['mean_gene_correlation'], color='red',
                          linestyle='--', label=f"均值: {metrics['mean_gene_correlation']:.3f}")
        axes[0, 0].set_xlabel("Pearson Correlation")
        axes[0, 0].set_ylabel("基因数量")
        axes[0, 0].set_title("基因级别预测相关性分布")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. 细胞预测相关性分布
        axes[0, 1].hist(metrics['cell_correlations'], bins=50, alpha=0.7, 
                       edgecolor='black', color='green')
        axes[0, 1].axvline(metrics['mean_cell_correlation'], color='red',
                          linestyle='--', label=f"均值: {metrics['mean_cell_correlation']:.3f}")
        axes[0, 1].set_xlabel("Pearson Correlation")
        axes[0, 1].set_ylabel("细胞数量")
        axes[0, 1].set_title("细胞级别预测相关性分布")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Cosine 相似度分布
        axes[1, 0].hist(metrics['cosine_similarities'], bins=50, alpha=0.7,
                       edgecolor='black', color='purple')
        axes[1, 0].axvline(metrics['mean_cosine_similarity'], color='red',
                          linestyle='--', label=f"均值: {metrics['mean_cosine_similarity']:.3f}")
        axes[1, 0].set_xlabel("Cosine Similarity")
        axes[1, 0].set_ylabel("细胞数量")
        axes[1, 0].set_title("预测 Cosine 相似度分布")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. 时间差异分布
        axes[1, 1].hist(metrics['time_diffs'], bins=50, alpha=0.7,
                       edgecolor='black', color='orange')
        axes[1, 1].axvline(metrics['mean_time_diff'], color='red',
                          linestyle='--', label=f"均值: {metrics['mean_time_diff']:.3f}")
        axes[1, 1].set_xlabel("时间差异 (Δt)")
        axes[1, 1].set_ylabel("样本数量")
        axes[1, 1].set_title("时序对时间差异分布")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "rtf_prediction_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  保存: {output_dir / 'rtf_prediction_distributions.png'}")


# ============================================================================
# 命令行接口
# ============================================================================

def full(
    ae_checkpoint: str,
    rtf_checkpoint: str,
    input_data: str,
    output_dir: str = "output/benchmark",
    batch_size: int = 64,
    sample_steps: int = 50,
    cfg_scale: float = 2.0,
    device: str = "auto",
):
    """
    完整评估（AE + RTF）
    
    Example:
        python utils/benchmark.py full \
            --ae_checkpoint=output/ae/checkpoints/last.ckpt \
            --rtf_checkpoint=output/rtf_direct_dit/checkpoints/last.ckpt \
            --input_data=/path/to/test.h5ad \
            --output_dir=output/benchmark
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"加载测试数据: {input_data}")
    adata = load_anndata(input_data, verbose=True)
    
    # 初始化评估器
    benchmark = CellTimeBenchmark(ae_checkpoint, rtf_checkpoint, device)
    
    # 评估 AE
    ae_metrics = benchmark.evaluate_ae_reconstruction(adata, batch_size=batch_size)
    benchmark.plot_ae_results(ae_metrics, output_dir)
    
    # 评估 RTF
    rtf_metrics = benchmark.evaluate_rtf_temporal_prediction(
        adata,
        batch_size=batch_size,
        sample_steps=sample_steps,
        cfg_scale=cfg_scale,
    )
    benchmark.plot_rtf_results(rtf_metrics, output_dir)
    
    # 保存结果（移除 numpy 数组）
    ae_metrics_save = {k: v for k, v in ae_metrics.items() 
                       if k not in ['reconstructions', 'latents']}
    rtf_metrics_save = {k: v for k, v in rtf_metrics.items()
                        if k not in ['predictions', 'ground_truth']}
    
    results = {
        'ae_metrics': ae_metrics_save,
        'rtf_metrics': rtf_metrics_save,
    }
    
    with open(output_dir / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ 评估完成！结果保存在: {output_dir}")
    print(f"  - benchmark_results.json")
    print(f"  - ae_correlation_distributions.png")
    print(f"  - rtf_prediction_distributions.png")


def ae(
    ae_checkpoint: str,
    input_data: str,
    output_dir: str = "output/benchmark",
    batch_size: int = 64,
    device: str = "auto",
):
    """
    仅评估 AE 重建质量
    
    Example:
        python utils/benchmark.py ae \
            --ae_checkpoint=output/ae/checkpoints/last.ckpt \
            --input_data=/path/to/test.h5ad \
            --output_dir=output/benchmark
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"加载测试数据: {input_data}")
    adata = load_anndata(input_data, verbose=True)
    
    # 初始化评估器
    benchmark = CellTimeBenchmark(ae_checkpoint, rtf_checkpoint=None, device=device)
    
    # 评估 AE
    ae_metrics = benchmark.evaluate_ae_reconstruction(adata, batch_size=batch_size)
    benchmark.plot_ae_results(ae_metrics, output_dir)
    
    # 保存结果
    ae_metrics_save = {k: v for k, v in ae_metrics.items()
                       if k not in ['reconstructions', 'latents']}
    
    with open(output_dir / "ae_benchmark_results.json", 'w') as f:
        json.dump({'ae_metrics': ae_metrics_save}, f, indent=2)
    
    print(f"\n✓ AE 评估完成！结果保存在: {output_dir}")


def rtf(
    ae_checkpoint: str,
    rtf_checkpoint: str,
    input_data: str,
    output_dir: str = "output/benchmark",
    batch_size: int = 64,
    sample_steps: int = 50,
    cfg_scale: float = 2.0,
    device: str = "auto",
):
    """
    仅评估 RTF 时序预测质量
    
    Example:
        python utils/benchmark.py rtf \
            --ae_checkpoint=output/ae/checkpoints/last.ckpt \
            --rtf_checkpoint=output/rtf_direct_dit/checkpoints/last.ckpt \
            --input_data=/path/to/test.h5ad \
            --output_dir=output/benchmark
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"加载测试数据: {input_data}")
    adata = load_anndata(input_data, verbose=True)
    
    # 初始化评估器
    benchmark = CellTimeBenchmark(ae_checkpoint, rtf_checkpoint, device)
    
    # 评估 RTF
    rtf_metrics = benchmark.evaluate_rtf_temporal_prediction(
        adata,
        batch_size=batch_size,
        sample_steps=sample_steps,
        cfg_scale=cfg_scale,
    )
    benchmark.plot_rtf_results(rtf_metrics, output_dir)
    
    # 保存结果
    rtf_metrics_save = {k: v for k, v in rtf_metrics.items()
                        if k not in ['predictions', 'ground_truth']}
    
    with open(output_dir / "rtf_benchmark_results.json", 'w') as f:
        json.dump({'rtf_metrics': rtf_metrics_save}, f, indent=2)
    
    print(f"\n✓ RTF 评估完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    fire.Fire({
        "full": full,
        "ae": ae,
        "rtf": rtf,
    })

