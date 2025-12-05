#!/usr/bin/env python
"""
Autoencoder 训练脚本
"""

import os
import fire
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pathlib import Path
from models.ae import AESystem
import sys
import logging
from datetime import datetime


def get_rank() -> int:
    """Get current process rank in DDP. Returns 0 for non-DDP."""
    # PyTorch DDP sets these environment variables
    rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
    return int(rank)


def setup_logging(output_dir: Path, debug: bool = False, rank: int = 0):
    """
    Configure logging to write debug info to files and minimal info to console

    Args:
        output_dir: Output directory for logs
        debug: Whether to enable debug mode
        rank: GPU rank (only rank 0 writes to console)
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped log filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log_file = log_dir / f"debug_rank{rank}_{timestamp}.log"
    train_log_file = log_dir / f"train_rank{rank}_{timestamp}.log"

    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('[%(levelname)s] %(message)s')

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove existing handlers
    root_logger.handlers.clear()

    # File handler for detailed debug logs
    debug_file_handler = logging.FileHandler(debug_log_file, mode='w')
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(debug_file_handler)

    # File handler for training logs
    train_file_handler = logging.FileHandler(train_log_file, mode='w')
    train_file_handler.setLevel(logging.INFO)
    train_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(train_file_handler)

    # Console handler (only for rank 0, minimal output)
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

    # Log the setup
    logging.info(f"Logging configured for rank {rank}")
    logging.info(f"  - Debug log: {debug_log_file}")
    logging.info(f"  - Train log: {train_log_file}")

    return debug_log_file, train_log_file


class DebugCallback(pl.Callback):
    """Debug callback to track epoch boundaries and potential deadlocks"""

    def on_train_epoch_start(self, trainer, pl_module):
        logging.debug(f"[Rank {trainer.global_rank}] ========== EPOCH {trainer.current_epoch} TRAINING START ==========")

    def on_train_epoch_end(self, trainer, pl_module):
        logging.debug(f"[Rank {trainer.global_rank}] ========== EPOCH {trainer.current_epoch} TRAINING END ==========")

    def on_validation_epoch_start(self, trainer, pl_module):
        logging.debug(f"[Rank {trainer.global_rank}] ========== EPOCH {trainer.current_epoch} VALIDATION START ==========")

    def on_validation_epoch_end(self, trainer, pl_module):
        logging.debug(f"[Rank {trainer.global_rank}] ========== EPOCH {trainer.current_epoch} VALIDATION END ==========")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log progress every 500 batches
        if batch_idx % 500 == 0:
            logging.debug(f"[Rank {trainer.global_rank}] Processed {batch_idx} training batches in epoch {trainer.current_epoch}")


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

    # Setup logging early (before any training starts)
    debug_mode = cfg.get("debug", False)
    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get actual rank for DDP (only rank 0 logs to console)
    rank = get_rank()
    debug_log, train_log = setup_logging(output_dir, debug=debug_mode, rank=rank)

    logging.info("=" * 80)
    logging.info("Training Configuration:")
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info("=" * 80)

    if debug_mode:
        logging.info("!" * 80)
        logging.info("DEBUG MODE ENABLED - Verbose logging enabled")
        logging.info("!" * 80)

    # Initialize System
    system = AESystem(cfg)

    # Build callbacks list
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="ae-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.RichProgressBar(),
    ]

    # Add debug callback if debug mode is enabled
    if debug_mode:
        callbacks.append(DebugCallback())

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
        limit_val_batches=100, # Limit validation to 100 batches for stability and speed
        callbacks=callbacks,
    )

    # Start Training
    logging.info("Starting training...")
    logging.info(f"Monitor logs with: tail -f {train_log}")
    logging.info(f"Monitor debug with: tail -f {debug_log}")

    trainer.fit(system)

    logging.info("Training finished!")
    # Check if checkpoint callback exists and has saved a best model
    checkpoint_callback = trainer.checkpoint_callback
    if checkpoint_callback and checkpoint_callback.best_model_path:
        logging.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
        logging.info(f"Best val_loss: {checkpoint_callback.best_model_score:.6f}")

if __name__ == "__main__":
    fire.Fire(train)
