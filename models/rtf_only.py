"""
Rectified Flow Only 模型
不在潜空间训练，直接在原始基因空间训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from models.utils import create_backbone


class RFDirectOnly(nn.Module):
    """
    Direct 模式：x1 -> x2
    直接从起点到终点的线性插值
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        ln_noise: bool = True,
        normalize: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.ln_noise = ln_noise
        self.normalize = normalize
    
    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """采样时间步 t ∈ [0, 1]"""
        if self.ln_noise:
            nt = torch.randn(batch_size, device=device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand(batch_size, device=device)
        return t
        
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        计算损失
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # 采样时间步
        t = self.sample_timestep(batch_size, device)
        t_exp = t.view(batch_size, *([1] * (x1.ndim - 1)))  # [B, 1, ...]
        
        # 线性插值：x_t = (1-t) * x1 + t * x2
        x_t = (1 - t_exp) * x1 + t_exp * x2
        
        if self.normalize:
            x_t = F.normalize(x_t, p=2, dim=1)
        
        # 预测速度场
        v_pred = self.backbone(x_t, t, cond)
        
        # 目标速度场：v = x2 - x1
        v_target = x2 - x1
        
        # 计算损失（每个样本）
        # 确保维度匹配
        if v_pred.shape != v_target.shape:
             # 如果 backbone 输出不匹配（例如 UNet 可能有维度问题），尝试调整
             pass

        batchwise_loss = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=list(range(1, v_pred.ndim)))
        
        # 总损失
        loss = batchwise_loss.mean()
        
        # 用于记录的损失字典
        loss_dict = [(t[i].item(), batchwise_loss[i].item()) for i in range(batch_size)]
        
        return loss, loss_dict
    
    @torch.no_grad()
    def sample(
        self,
        x_start: torch.Tensor,
        sample_steps: int = 50,
        cond: Optional[torch.Tensor] = None,
        null_cond: Optional[torch.Tensor] = None,
        cfg_scale: float = 2.0,
        normalize: bool = False,
    ) -> List[torch.Tensor]:
        """
        从 x1 采样到 x2
        """
        x = x_start.clone()
        batch_size = x.shape[0]
        device = x.device
        dt = 1.0 / sample_steps
        
        trajectory = [x.cpu()]
        
        for step in range(sample_steps):
            t_current = step / sample_steps
            t = torch.full((batch_size,), t_current, device=device)
            
            # 预测速度场
            v = self.backbone(x, t, cond)
            
            # Classifier-Free Guidance
            if null_cond is not None and cfg_scale != 1.0:
                v_uncond = self.backbone(x, t, null_cond)
                v = v_uncond + cfg_scale * (v - v_uncond)
            
            # 欧拉步
            x = x + v * dt
            
            if normalize:
                x = F.normalize(x, p=2, dim=1)
            
            trajectory.append(x.cpu())
        
        return trajectory


class RTFOnlySystem(pl.LightningModule):
    """
    Rectified Flow Only 训练系统（无 AE）
    """
    
    def __init__(
        self,
        cfg: DictConfig,
    ):
        """
        Args:
            cfg: 配置对象
        """
        super().__init__()
        
        if isinstance(cfg, DictConfig):
            self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
            self.cfg = cfg
        else:
            self.save_hyperparameters(cfg)
            self.cfg = OmegaConf.create(cfg)
        
        cfg = self.cfg
        
        # 加载骨干网络配置
        backbone_config_path = f"config/backbones/{cfg.model.backbone}.yaml"
        backbone_cfg = OmegaConf.load(backbone_config_path)
        backbone_cfg_dict = OmegaConf.to_container(backbone_cfg, resolve=True)
        
        # 调整 UNet/MLP 的输入维度为基因数
        # 注意：backbone 工厂通常用 'latent_dim' 参数作为输入维度
        # 这里我们传入 n_genes 作为 latent_dim
        input_dim = cfg.model.n_genes
        
        # 注入条件配置
        if cfg.model.use_cond and cfg.model.cond_dim is not None:
            backbone_cfg_dict['use_vector_cond'] = True
            backbone_cfg_dict['vector_cond_dim'] = cfg.model.cond_dim
        
        # 创建骨干网络
        backbone = create_backbone(
            cfg.model.backbone,
            backbone_cfg_dict,
            input_dim  # 传入基因数作为维度
        )
        
        # 创建 RTF 模型
        self.model = RFDirectOnly(
            backbone, 
            ln_noise=cfg.model.ln_noise,
            normalize=cfg.model.normalize
        )
        
        # 输出目录
        self.output_dir = Path(cfg.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._reset_loss_bins()
    
    def _reset_loss_bins(self):
        self.loss_bins = {i: 0.0 for i in range(10)}
        self.loss_counts = {i: 1e-6 for i in range(10)}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        
        if self.cfg.training.scheduler.type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.training.scheduler.T_max,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10,
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            from dataset import TemporalCellDataset
            
            target_genes = None
            if hasattr(self.cfg.data, 'target_genes_path') and self.cfg.data.target_genes_path:
                path = Path(self.cfg.data.target_genes_path)
                if path.exists():
                    with open(path, 'r') as f:
                        target_genes = [line.strip() for line in f if line.strip()]
            
            self.train_dataset = TemporalCellDataset(
                data=self.cfg.data.data_path,
                max_genes=self.cfg.model.n_genes,
                target_genes=target_genes,
                valid_pairs_only=self.cfg.data.valid_pairs_only,
                time_col=self.cfg.data.time_col,
                next_cell_col=self.cfg.data.next_cell_col,
                verbose=True,
            )
            
            # 更新 n_genes 以匹配数据集（如果未指定或不同）
            if self.cfg.model.n_genes != self.train_dataset.n_genes:
                print(f"Updating n_genes from {self.cfg.model.n_genes} to {self.train_dataset.n_genes}")
                # 注意：模型已经创建，如果维度不匹配会报错，这里仅打印警告
                # 实际应用中应该先加载数据再创建模型，或者确保配置正确
    
    def train_dataloader(self):
        from dataset import collate_fn_temporal
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            collate_fn=collate_fn_temporal,
        )
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        x_cur = batch["x_cur"]
        x_next = batch["x_next"]
        t_cur = batch["t_cur"]
        t_next = batch["t_next"]
        
        # 诊断信息
        if batch_idx % 100 == 0:
            with torch.no_grad():
                v_norm = torch.norm(x_next - x_cur, dim=-1).mean()
                print(f"\n[Batch {batch_idx}] v_target norm: {v_norm:.4f}")
        
        cond = None
        if self.cfg.model.use_cond:
            cond = torch.stack([t_cur, t_next], dim=-1)
            
        loss, loss_dict = self.model(x_cur, x_next, cond)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        for t_val, l_val in loss_dict:
            bin_idx = int(t_val * 10)
            if 0 <= bin_idx < 10:
                self.loss_bins[bin_idx] += l_val
                self.loss_counts[bin_idx] += 1.0
                
        return loss

    def on_train_epoch_end(self):
        for i in range(10):
            avg_loss = self.loss_bins[i] / self.loss_counts[i]
            self.log(f"loss_bin_{i}", avg_loss, prog_bar=False, on_epoch=True)
            
        if self.current_epoch % self.cfg.training.sample_every_n_epochs == 0:
            self._sample_and_save(self.current_epoch)
        self._reset_loss_bins()
        
    @torch.no_grad()
    def _sample_and_save(self, epoch: int):
        self.model.eval()
        batch = next(iter(self.train_dataloader()))
        x_cur = batch["x_cur"][:8].to(self.device)
        x_next = batch["x_next"][:8].to(self.device)
        t_cur = batch["t_cur"][:8].to(self.device)
        t_next = batch["t_next"][:8].to(self.device)
        
        cond = None
        null_cond = None
        if self.cfg.model.use_cond:
            cond = torch.stack([t_cur, t_next], dim=-1)
            null_cond = torch.zeros_like(cond)
            
        trajectory = self.model.sample(
            x_cur,
            sample_steps=self.cfg.training.sample_steps,
            cond=cond,
            null_cond=null_cond,
            cfg_scale=self.cfg.training.cfg_scale,
            normalize=self.cfg.model.normalize
        )
        
        x_final = trajectory[-1].to(self.device)
        
        # 计算误差
        recon_error = F.mse_loss(x_final, x_next).item()
        from models.utils import compute_correlation
        correlation = compute_correlation(x_next, x_final)
        
        self.log("sample_recon_error", recon_error, on_epoch=True)
        self.log("sample_correlation", correlation, on_epoch=True)
        
        print(f"Epoch {epoch}: Sampled")
        print(f"  Recon Error: {recon_error:.6f}")
        print(f"  Correlation: {correlation:.4f}")
        
        self.model.train()

