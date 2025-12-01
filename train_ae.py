#!/usr/bin/env python
"""
Autoencoder 训练脚本
"""

import fire
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pathlib import Path
from models.ae import AESystem

def train(
    config_path: str = "config/ae.yaml",
    **kwargs
):
    """
    Train Autoencoder
    
    Args:
        config_path: Path to config file
        **kwargs: Overrides
    """
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # CLI Overrides
    if kwargs:
        flat_kwargs = {}
        for key, value in kwargs.items():
            if '__' in key:
                key = key.replace('__', '.')
            flat_kwargs[key] = value
        cli_cfg = OmegaConf.from_dotlist([f"{k}={v}" for k, v in flat_kwargs.items()])
        cfg = OmegaConf.merge(cfg, cli_cfg)
    
    print("=" * 80)
    print("Training Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Initialize System
    system = AESystem(cfg)
    
    # Initialize Trainer
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
                patience=10,
                mode="min",
            ),
            pl.callbacks.RichProgressBar(),
        ],
    )
    
    # Start Training
    if trainer.global_rank == 0:
        print("Starting training...")
    trainer.fit(system)
    
    if trainer.global_rank == 0:
        print(f"\nTraining finished!")
        if trainer.checkpoint_callback.best_model_path:
            print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
            print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score:.6f}")

if __name__ == "__main__":
    fire.Fire(train)
