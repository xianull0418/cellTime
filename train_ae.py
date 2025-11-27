#!/usr/bin/env python
"""
Autoencoder 训练脚本
使用 Fire + PyTorch Lightning + OmegaConf
"""

import fire
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

from models.ae import AESystem


def train(
    config_path: str = "config/ae.yaml",
    data_path: str = None,
    **kwargs
):
    """
    训练 Autoencoder
    
    Args:
        config_path: 配置文件路径
        data_path: 数据路径（快捷参数）
        **kwargs: 命令行参数覆盖配置文件
    
    Example:
        python train_ae.py \\
            --data_path=/path/to/data.h5ad \\
            --model.latent_dim=256 \\
            --training.max_epochs=100
    """
    # 加载配置
    cfg = OmegaConf.load(config_path)
    
    # 处理快捷参数
    if data_path is not None:
        cfg.data.data_path = data_path
    
    # 命令行参数覆盖配置文件
    if kwargs:
        # 处理嵌套参数（如 model__latent_dim）
        flat_kwargs = {}
        for key, value in kwargs.items():
            if '__' in key:
                # 将 model__latent_dim 转换为 model.latent_dim
                key = key.replace('__', '.')
            flat_kwargs[key] = value
        
        cli_cfg = OmegaConf.from_dotlist([f"{k}={v}" for k, v in flat_kwargs.items()])
        cfg = OmegaConf.merge(cfg, cli_cfg)
    
    # 验证必要参数
    # 验证必要参数
    dataset_type = cfg.data.get("dataset_type", "h5ad")
    if cfg.data.data_path is None and dataset_type != "parquet":
        raise ValueError(
            "data_path must be provided via config or command line for h5ad dataset\n"
            "Example: --data_path=/path/to/data.h5ad"
        )
    
    # ---------------------------------------------------------
    # 关键修复：在创建模型前，根据 vocab_path 预先计算 n_genes
    # ---------------------------------------------------------
    if hasattr(cfg.data, 'vocab_path') and cfg.data.vocab_path:
        vocab_path = Path(cfg.data.vocab_path)
        if vocab_path.exists():
            try:
                # 简单的行数统计，避免引入重型依赖
                with open(vocab_path, 'r') as f:
                    # 过滤空行
                    vocab_lines = [line.strip() for line in f if line.strip()]
                    n_vocab = len(vocab_lines)
                
                if n_vocab > 0:
                    print(f"从词汇表 {vocab_path} 检测到 {n_vocab} 个基因")
                    if cfg.model.n_genes != n_vocab:
                        print(f"自动更新 model.n_genes: {cfg.model.n_genes} -> {n_vocab}")
                        cfg.model.n_genes = n_vocab
            except Exception as e:
                print(f"Warning: 尝试读取词汇表失败: {e}")
        else:
            print(f"Warning: 指定的词汇表路径不存在: {vocab_path}")

    # 打印配置
    print("=" * 80)
    print("训练配置:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # 创建 Lightning 系统
    system = AESystem(cfg)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in system.parameters() if p.requires_grad)
    print(f"\n模型参数数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"基因数量: {cfg.model.n_genes}")
    print(f"潜空间维度: {cfg.model.latent_dim}")
    print(f"隐藏层维度: {cfg.model.hidden_dim}")
    print(f"重建损失: {cfg.training.reconstruction_loss}\n")
    
    # 创建 Trainer
    trainer = pl.Trainer(
        accelerator=cfg.accelerator.accelerator,
        devices=cfg.accelerator.devices,
        precision=cfg.accelerator.precision,
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        val_check_interval=cfg.logging.val_check_interval,
        enable_checkpointing=True,
        default_root_dir=cfg.logging.output_dir,
        gradient_clip_val=cfg.accelerator.gradient_clip_val,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=Path(cfg.logging.output_dir) / "checkpoints",
                filename="ae-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                mode="min",
            ),
            pl.callbacks.RichProgressBar(),  # 使用富进度条
        ],
    )
    
    # 开始训练
    print("开始训练 Autoencoder...")
    trainer.fit(system)
    
    # 训练完成
    print(f"\n训练完成！")
    print(f"最佳模型保存在: {trainer.checkpoint_callback.best_model_path}")
    print(f"最佳验证损失: {trainer.checkpoint_callback.best_model_score:.6f}")
    
    return system, trainer


if __name__ == "__main__":
    fire.Fire(train)

