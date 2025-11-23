#!/usr/bin/env python
"""
RTF Only Training Script
Train Rectified Flow directly on gene expression data without AE
"""

import fire
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pathlib import Path

from models.rtf_only import RTFOnlySystem


def train(
    config_path: str = "config/rtf_only.yaml",
    data_path: str = None,
    **kwargs
):
    """
    Train RTF Only model
    
    Args:
        config_path: Path to config file
        data_path: Path to data file (override)
        **kwargs: CLI arguments to override config
    """
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Override data path
    if data_path is not None:
        cfg.data.data_path = data_path
        
    # Override with CLI args
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
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Create system
    system = RTFOnlySystem(cfg)
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=cfg.accelerator.accelerator,
        devices=cfg.accelerator.devices,
        precision=cfg.accelerator.precision,
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        default_root_dir=cfg.logging.output_dir,
        gradient_clip_val=cfg.accelerator.gradient_clip_val,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=Path(cfg.logging.output_dir) / "checkpoints",
                filename="rtf-{epoch:02d}-{train_loss:.4f}",
                monitor="train_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.RichProgressBar(),
        ],
    )
    
    print("Starting training...")
    trainer.fit(system)
    
    print("Training completed!")


if __name__ == "__main__":
    fire.Fire(train)

