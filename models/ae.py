"""
Autoencoder 模型
将单细胞基因表达数据编码到潜空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import logging

# from scimilarity.nn_models import Encoder, Decoder # Removed dependency
from models.utils import compute_correlation
from dataset import ParquetDataset, ParquetIterableDataset, ZarrIterableDataset, collate_fn_static

class MLPBlock(nn.Module):
    """
    Basic MLP Block with LayerNorm, GELU and Residual Connection (if dims match)
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.has_residual = (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        
        if self.has_residual:
            x = x + residual
        return x

class Encoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int, hidden_dim: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = n_genes
        
        # Input projection -> Hidden layers
        for h_dim in hidden_dim:
            layers.append(MLPBlock(in_dim, h_dim, dropout))
            in_dim = h_dim
            
        # Final projection to latent
        self.hidden_layers = nn.Sequential(*layers)
        self.to_latent = nn.Linear(in_dim, latent_dim)
        # Optional: LayerNorm on latent? Usually raw linear is fine for AE bottleneck.
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        z = self.to_latent(x)
        return z

class Decoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int, hidden_dim: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        # Reverse hidden dims for decoder
        hidden_dim = hidden_dim[::-1]
        
        in_dim = latent_dim
        for h_dim in hidden_dim:
            layers.append(MLPBlock(in_dim, h_dim, dropout))
            in_dim = h_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        self.to_output = nn.Linear(in_dim, n_genes)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(z)
        out = self.to_output(x)
        # Since input is log1p (non-negative), we can enforce non-negativity.
        # However, raw linear allows gradient to flow back even if prediction is negative initially.
        # Let's just return raw logits for loss function to handle, 
        # OR apply ReLU if we are sure target is non-negative.
        # Given it's scRNA-seq log1p data, it is strictly >= 0.
        return F.relu(out) 

class Autoencoder(nn.Module):
    """
    Autoencoder 模型
    Custom Implementation with Residual MLP Blocks
    """
    
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 256,
        hidden_dim: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = [1024, 512]
        
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim, # Do not copy, just pass list
            dropout=dropout_rate,
        )
        
        self.decoder = Decoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim, # Do not copy
            dropout=dropout_rate,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class AESystem(pl.LightningModule):
    """
    Autoencoder 训练系统（PyTorch Lightning）
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        if isinstance(cfg, DictConfig):
            self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
            self.cfg = cfg
        else:
            self.save_hyperparameters(cfg)
            self.cfg = OmegaConf.create(cfg)
        
        cfg = self.cfg
        
        # Convert hidden_dim to list if it's OmegaConf ListConfig
        hidden_dim = list(cfg.model.hidden_dim) if hasattr(cfg.model.hidden_dim, '__iter__') else cfg.model.hidden_dim
        
        self.autoencoder = Autoencoder(
            n_genes=cfg.model.n_genes,
            latent_dim=cfg.model.latent_dim,
            hidden_dim=hidden_dim,
            dropout_rate=cfg.model.dropout_rate,
        )
        
        loss_type = cfg.training.reconstruction_loss
        if loss_type == "mse":
            self.reconstruction_loss_fn = nn.MSELoss()
        elif loss_type == "poisson":
            self.reconstruction_loss_fn = nn.PoissonNLLLoss(log_input=False)
        elif loss_type == "smooth_l1":
            self.reconstruction_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported reconstruction loss: {loss_type}")
        
        # Output dir
        self.output_dir = Path(cfg.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.val_dataset = None
        self.ood_dataset = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.autoencoder(x)
    
    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.cfg.training.scheduler.mode,
            factor=self.cfg.training.scheduler.factor,
            patience=self.cfg.training.scheduler.patience,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets.
        Now supports 'parquet' and 'zarr' type natively.
        """
        if stage == "fit" or stage is None:
            dataset_type = self.cfg.data.get("dataset_type", "parquet")
            # Renamed from parquet_path to processed_path for generality
            p_cfg = self.cfg.data.get("processed_path", self.cfg.data.get("parquet_path", None))

            # Get debug flag from config
            debug_mode = self.cfg.get("debug", False)

            if not p_cfg:
                 raise ValueError(f"dataset_type='{dataset_type}' but data.processed_path is missing in config.")

            train_path = Path(p_cfg.train)
            val_path = Path(p_cfg.val)
            ood_path = Path(p_cfg.ood)

            if dataset_type == "zarr":
                logging.info("Setting up Zarr datasets...")

                # Train
                if train_path.exists():
                    logging.info(f"Loading Training Data from {train_path} (Zarr Iterable Mode)...")
                    self.train_dataset = ZarrIterableDataset(train_path, verbose=True, shuffle_shards=True, shuffle_rows=True, debug=debug_mode)

                    if self.cfg.model.n_genes != self.train_dataset.n_genes:
                        logging.info(f"Auto-updating n_genes: {self.cfg.model.n_genes} -> {self.train_dataset.n_genes}")
                        self.cfg.model.n_genes = self.train_dataset.n_genes
                else:
                    raise FileNotFoundError(f"Train data not found: {train_path}")

                # Val
                if val_path.exists():
                    self.val_dataset = ZarrIterableDataset(val_path, verbose=True, shuffle_shards=False, shuffle_rows=False, debug=debug_mode)
                else:
                    logging.warning(f"Val data not found: {val_path}")

                # OOD
                if ood_path.exists():
                    self.ood_dataset = ZarrIterableDataset(ood_path, verbose=True, shuffle_shards=False, shuffle_rows=False, debug=debug_mode)

            elif dataset_type == "parquet":
                logging.info("Setting up Parquet datasets...")

                # Train
                if train_path.exists():
                    # Use IterableDataset for training to handle large scale data efficiently
                    logging.info(f"Loading Training Data from {train_path} (Iterable Mode)...")
                    self.train_dataset = ParquetIterableDataset(train_path, verbose=True, shuffle_shards=True, shuffle_rows=True, debug=debug_mode)

                    # Update n_genes automatically
                    if self.cfg.model.n_genes != self.train_dataset.n_genes:
                        logging.info(f"Auto-updating n_genes: {self.cfg.model.n_genes} -> {self.train_dataset.n_genes}")
                        self.cfg.model.n_genes = self.train_dataset.n_genes
                else:
                    raise FileNotFoundError(f"Train data not found: {train_path}")

                # Val
                if val_path.exists():
                    # Use IterableDataset for val as well to avoid massive RAM usage, but no shuffle
                    self.val_dataset = ParquetIterableDataset(val_path, verbose=True, shuffle_shards=False, shuffle_rows=False, debug=debug_mode)
                else:
                    logging.warning(f"Val data not found: {val_path}")

                # OOD
                if ood_path.exists():
                    self.ood_dataset = ParquetIterableDataset(ood_path, verbose=True, shuffle_shards=False, shuffle_rows=False, debug=debug_mode)

            else:
                # Legacy support if needed
                logging.warning(f"Unknown dataset_type '{dataset_type}', falling back to manual setup or error.")

        # Re-initialize autoencoder if n_genes changed during setup
        if self.autoencoder.n_genes != self.cfg.model.n_genes:
            logging.info(f"Re-initializing Autoencoder with n_genes={self.cfg.model.n_genes}")
            # Convert hidden_dim to list again for safety
            hidden_dim = list(self.cfg.model.hidden_dim) if hasattr(self.cfg.model.hidden_dim, '__iter__') else self.cfg.model.hidden_dim
            
            self.autoencoder = Autoencoder(
                n_genes=self.cfg.model.n_genes,
                latent_dim=self.cfg.model.latent_dim,
                hidden_dim=hidden_dim,
                dropout_rate=self.cfg.model.dropout_rate,
            ).to(self.device)
            
            # IMPORTANT: Sync reconstruction_loss_fn with model device
            if hasattr(self, 'reconstruction_loss_fn'):
                self.reconstruction_loss_fn = self.reconstruction_loss_fn.to(self.device)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False, # Shuffle handled internally by IterableDataset
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            collate_fn=collate_fn_static,
            drop_last=True, # Drop incomplete batches to avoid DDP deadlocks
        )
    
    def val_dataloader(self):
        if not self.val_dataset: 
            return [] 
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            collate_fn=collate_fn_static,
            drop_last=True, # Consistent behavior
        )
    
    def test_dataloader(self):
        """OOD Validation"""
        if not self.ood_dataset: return []
        return DataLoader(
            self.ood_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            collate_fn=collate_fn_static,
        )
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        x_reconstructed, latent = self.forward(x)
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)

        # Use sync_dist=True for multi-GPU training to properly aggregate metrics
        self.log("train_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        # DEBUG: Log progress periodically
        if self.cfg.get("debug", False) and batch_idx % 100 == 0:
            rank = self.trainer.global_rank if self.trainer else 0
            logging.debug(f"[Rank {rank}] Training step {batch_idx}, loss={recon_loss.item():.4f}")

        # Return loss for backprop - PyTorch Lightning handles this correctly
        return recon_loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = batch
        x_reconstructed, latent = self.forward(x)
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)

        # Detach metrics to avoid memory leaks
        # These metrics don't need gradients and shouldn't keep computation graph
        with torch.no_grad():
            mse = F.mse_loss(x_reconstructed, x)
            correlation = compute_correlation(x, x_reconstructed)

        # Log with sync_dist for multi-GPU training
        self.log("val_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_correlation", correlation, on_step=False, on_epoch=True, sync_dist=True)

        # DEBUG: Log validation progress
        if self.cfg.get("debug", False) and batch_idx % 20 == 0:
            rank = self.trainer.global_rank if self.trainer else 0
            logging.debug(f"[Rank {rank}] Validation step {batch_idx}, loss={recon_loss.item():.4f}")

        # Don't return anything - PyTorch Lightning will handle logging
        # Returning tensors can cause memory leaks by keeping computation graphs
        
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = batch
        x_reconstructed, latent = self.forward(x)
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)

        # Detach metrics to avoid memory leaks
        with torch.no_grad():
            mse = F.mse_loss(x_reconstructed, x)
            correlation = compute_correlation(x, x_reconstructed)

        # Log with sync_dist for multi-GPU training
        self.log("ood_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("ood_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("ood_correlation", correlation, on_step=False, on_epoch=True, sync_dist=True)

        # Don't return anything to avoid memory leaks
