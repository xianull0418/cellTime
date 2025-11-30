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

from scimilarity.nn_models import Encoder, Decoder
from models.utils import compute_correlation
from dataset import ParquetDataset, ParquetIterableDataset, ZarrIterableDataset, collate_fn_static

class Autoencoder(nn.Module):
    """
    Autoencoder 模型
    基于 scimilarity 的 Encoder/Decoder
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
            hidden_dim=hidden_dim.copy(),
            dropout=dropout_rate,
        )
        
        self.decoder = Decoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim.copy(),
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
        
        self.autoencoder = Autoencoder(
            n_genes=cfg.model.n_genes,
            latent_dim=cfg.model.latent_dim,
            hidden_dim=cfg.model.hidden_dim,
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
            
            if not p_cfg:
                 raise ValueError(f"dataset_type='{dataset_type}' but data.processed_path is missing in config.")

            train_path = Path(p_cfg.train)
            val_path = Path(p_cfg.val)
            ood_path = Path(p_cfg.ood)
            
            if dataset_type == "zarr":
                print("Setting up Zarr datasets...")
                
                # Train
                if train_path.exists():
                    print(f"Loading Training Data from {train_path} (Zarr Iterable Mode)...")
                    self.train_dataset = ZarrIterableDataset(train_path, verbose=True, shuffle_shards=True, shuffle_rows=True)
                    
                    if self.cfg.model.n_genes != self.train_dataset.n_genes:
                        print(f"Auto-updating n_genes: {self.cfg.model.n_genes} -> {self.train_dataset.n_genes}")
                        self.cfg.model.n_genes = self.train_dataset.n_genes
                else:
                    raise FileNotFoundError(f"Train data not found: {train_path}")
                
                # Val
                if val_path.exists():
                    self.val_dataset = ZarrIterableDataset(val_path, verbose=True, shuffle_shards=False, shuffle_rows=False)
                else:
                    print(f"Warning: Val data not found: {val_path}")
                    
                # OOD
                if ood_path.exists():
                    self.ood_dataset = ZarrIterableDataset(ood_path, verbose=True, shuffle_shards=False, shuffle_rows=False)

            elif dataset_type == "parquet":
                print("Setting up Parquet datasets...")
                
                # Train
                if train_path.exists():
                    # Use IterableDataset for training to handle large scale data efficiently
                    print(f"Loading Training Data from {train_path} (Iterable Mode)...")
                    self.train_dataset = ParquetIterableDataset(train_path, verbose=True, shuffle_shards=True, shuffle_rows=True)
                    
                    # Update n_genes automatically
                    if self.cfg.model.n_genes != self.train_dataset.n_genes:
                        print(f"Auto-updating n_genes: {self.cfg.model.n_genes} -> {self.train_dataset.n_genes}")
                        self.cfg.model.n_genes = self.train_dataset.n_genes
                else:
                    raise FileNotFoundError(f"Train data not found: {train_path}")
                
                # Val
                if val_path.exists():
                    # Use IterableDataset for val as well to avoid massive RAM usage, but no shuffle
                    self.val_dataset = ParquetIterableDataset(val_path, verbose=True, shuffle_shards=False, shuffle_rows=False)
                else:
                    print(f"Warning: Val data not found: {val_path}")
                    
                # OOD
                if ood_path.exists():
                    self.ood_dataset = ParquetIterableDataset(ood_path, verbose=True, shuffle_shards=False, shuffle_rows=False)
            
            else:
                # Legacy support if needed
                print(f"Warning: Unknown dataset_type '{dataset_type}', falling back to manual setup or error.")

        # Re-initialize autoencoder if n_genes changed during setup
        if self.autoencoder.n_genes != self.cfg.model.n_genes:
            print(f"Re-initializing Autoencoder with n_genes={self.cfg.model.n_genes}")
            self.autoencoder = Autoencoder(
                n_genes=self.cfg.model.n_genes,
                latent_dim=self.cfg.model.latent_dim,
                hidden_dim=self.cfg.model.hidden_dim,
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
        )
    
    def val_dataloader(self):
        if not self.val_dataset: 
            # Return an empty list or None is tricky in PL
            # Better to return a dummy dataloader or handle this case
            # If we return None, PL complains.
            # Let's try returning an empty DataLoader if dataset is missing, but ideally we should have val data.
            # If truly no val data, maybe we should skip val? 
            # For now, let's assume if it's None, we return an empty list which acts as an empty iterator?
            # Actually, the error says "TypeError: 'NoneType' object is not iterable" which implies PL tried to iterate over None.
            # Correct fix: if no val dataset, don't return None if PL expects it. 
            # However, PL supports None for val_dataloader IF limit_val_batches=0. 
            
            # Let's just return a dummy empty list which is iterable, although PL might expect DataLoader.
            # Safer: create a dummy dataset.
            return [] 
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            collate_fn=collate_fn_static,
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
        
        self.log("train_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        return recon_loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        x = batch
        x_reconstructed, latent = self.forward(x)
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)
        mse = F.mse_loss(x_reconstructed, x)
        correlation = compute_correlation(x, x_reconstructed)
        
        self.log("val_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True)
        self.log("val_correlation", correlation, on_step=False, on_epoch=True)
        
        return {"val_loss": recon_loss}
        
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        x = batch
        x_reconstructed, latent = self.forward(x)
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)
        mse = F.mse_loss(x_reconstructed, x)
        correlation = compute_correlation(x, x_reconstructed)
        
        self.log("ood_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("ood_mse", mse, on_step=False, on_epoch=True)
        self.log("ood_correlation", correlation, on_step=False, on_epoch=True)
        
        return {"ood_loss": recon_loss}
