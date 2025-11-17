#!/usr/bin/env python
"""
Rectified Flow 训练脚本
在 AE 潜空间训练 RTF，使用 Fire + PyTorch Lightning + OmegaConf
"""

import fire
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pathlib import Path

from models.ae import AESystem
from models.rtf import RTFSystem


def train(
    config_path: str = "config/rtf.yaml",
    ae_checkpoint: str = None,
    data_path: str = None,
    **kwargs
):
    """
    训练 Rectified Flow
    
    Args:
        config_path: RTF 配置文件路径
        ae_checkpoint: 预训练 AE checkpoint 路径
        data_path: 数据路径（快捷参数）
        **kwargs: 命令行参数覆盖配置文件
    
    Example:
        # Direct 模式 + DiT 骨干网络
        python train_rtf.py \\
            --ae_checkpoint=output/ae/checkpoints/last.ckpt \\
            --data_path=/path/to/temporal_data.h5ad \\
            --model__mode=direct \\
            --model__backbone=dit
        
        # Inversion 模式 + MLP 骨干网络
        python train_rtf.py \\
            --ae_checkpoint=output/ae/checkpoints/last.ckpt \\
            --data_path=/path/to/temporal_data.h5ad \\
            --model__mode=inversion \\
            --model__backbone=mlp
    """
    # 加载配置
    cfg = OmegaConf.load(config_path)
    
    # 处理快捷参数
    if ae_checkpoint is not None:
        cfg.model.ae_checkpoint = ae_checkpoint
    
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
    if cfg.data.data_path is None:
        raise ValueError(
            "data_path must be provided via config or command line\n"
            "Example: --data_path=/path/to/temporal_data.h5ad"
        )
    
    if cfg.model.ae_checkpoint is None:
        raise ValueError(
            "ae_checkpoint must be provided via config or command line\n"
            "Example: --ae_checkpoint=output/ae/checkpoints/last.ckpt"
        )
    
    # 检查 checkpoint 是否存在
    checkpoint_path = Path(cfg.model.ae_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"AE checkpoint not found: {checkpoint_path}")
    
    print("=" * 80)
    print("训练配置:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # 加载预训练的 AE
    print(f"\n加载预训练 AE: {cfg.model.ae_checkpoint}")
    ae_system = AESystem.load_from_checkpoint(cfg.model.ae_checkpoint)
    ae_encoder = ae_system.autoencoder.encoder
    
    # 冻结 AE Encoder
    if cfg.model.freeze_ae:
        ae_encoder.eval()
        for param in ae_encoder.parameters():
            param.requires_grad = False
        print("AE Encoder 已冻结")
    else:
        print("Warning: AE Encoder 未冻结，将参与训练")
    
    # 创建 RTF 系统
    system = RTFSystem(cfg, ae_encoder=ae_encoder)
    
    # 打印模型信息
    rtf_params = sum(p.numel() for p in system.model.parameters() if p.requires_grad)
    ae_params = sum(p.numel() for p in ae_encoder.parameters())
    print(f"\nRTF 模型参数数量: {rtf_params:,} ({rtf_params / 1e6:.2f}M)")
    print(f"AE Encoder 参数数量: {ae_params:,} ({ae_params / 1e6:.2f}M)")
    print(f"模式: {cfg.model.mode}")
    print(f"骨干网络: {cfg.model.backbone}")
    print(f"潜空间维度: {cfg.model.latent_dim}")
    print(f"采样步数: {cfg.training.sample_steps}")
    print(f"CFG 强度: {cfg.training.cfg_scale}\n")
    
    # 创建 Trainer
    trainer = pl.Trainer(
        accelerator=cfg.accelerator.accelerator,
        devices=cfg.accelerator.devices,
        precision=cfg.accelerator.precision,
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        enable_checkpointing=True,
        default_root_dir=cfg.logging.output_dir,
        gradient_clip_val=cfg.accelerator.gradient_clip_val,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=Path(cfg.logging.output_dir) / "checkpoints",
                filename=f"rtf-{cfg.model.mode}-{cfg.model.backbone}-{{epoch:02d}}-{{train_loss:.4f}}",
                monitor="train_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                every_n_epochs=5,  # 每 5 个 epoch 保存一次
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.RichProgressBar(),  # 使用富进度条
        ],
    )
    
    # 开始训练
    print(f"开始训练 Rectified Flow ({cfg.model.mode} 模式 + {cfg.model.backbone} 骨干网络)...")
    trainer.fit(system)
    
    # 训练完成
    print(f"\n训练完成！")
    print(f"最佳模型保存在: {trainer.checkpoint_callback.best_model_path}")
    print(f"最佳训练损失: {trainer.checkpoint_callback.best_model_score:.6f}")
    
    # 采样目录
    sample_dir = Path(cfg.logging.output_dir) / "samples"
    if sample_dir.exists():
        print(f"采样结果保存在: {sample_dir}")
    
    return system, trainer


if __name__ == "__main__":
    fire.Fire(train)

