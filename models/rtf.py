"""
Rectified Flow 模型
支持两种模式：Direct 和 Inversion
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


class RectifiedFlow(nn.Module):
    """
    Rectified Flow 基类
    实现核心的 Rectified Flow 算法
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        ln_noise: bool = True,
        normalize_latent: bool = True,
    ):
        """
        Args:
            backbone: 速度场预测器（骨干网络）
            ln_noise: 是否使用 log-normal 噪声分布采样时间
            normalize_latent: 是否归一化潜空间向量（用于 scimilarity encoder）
        """
        super().__init__()
        self.backbone = backbone
        self.ln_noise = ln_noise
        self.normalize_latent = normalize_latent
    
    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        采样时间步 t ∈ [0, 1]
        
        Args:
            batch_size: 批次大小
            device: 设备
        
        Returns:
            时间步 [B]
        """
        if self.ln_noise:
            # Log-normal 分布（更关注中间时间步）
            nt = torch.randn(batch_size, device=device)
            t = torch.sigmoid(nt)
        else:
            # 均匀分布
            t = torch.rand(batch_size, device=device)
        return t
    
    @torch.no_grad()
    def sample(
        self,
        z_start: torch.Tensor,
        sample_steps: int = 50,
        cond: Optional[torch.Tensor] = None,
        null_cond: Optional[torch.Tensor] = None,
        cfg_scale: float = 2.0,
    ) -> List[torch.Tensor]:
        """
        从起点采样到终点（需要在子类中实现具体逻辑）
        
        Args:
            z_start: 起始潜空间 [B, latent_dim]
            sample_steps: 采样步数
            cond: 条件信息
            null_cond: 无条件信息（用于 CFG）
            cfg_scale: CFG 强度
        
        Returns:
            采样轨迹列表
        """
        raise NotImplementedError


class RFDirect(RectifiedFlow):
    """
    Direct 模式：z1 -> z2
    直接从起点到终点的线性插值
    """
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        计算 Direct 模式的损失
        
        Args:
            z1: 起点潜空间 [B, latent_dim]
            z2: 终点潜空间 [B, latent_dim]
            cond: 可选条件信息
        
        Returns:
            loss: 损失值
            loss_dict: 损失字典（用于记录）
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # 采样时间步
        t = self.sample_timestep(batch_size, device)
        t_exp = t.view(batch_size, *([1] * (z1.ndim - 1)))  # [B, 1, ...]
        
        # 线性插值：z_t = (1-t) * z1 + t * z2
        z_t = (1 - t_exp) * z1 + t_exp * z2
        
        # 🔧 关键修复：如果使用 scimilarity encoder，训练时也需要归一化
        # 保持训练-推理一致性
        if self.normalize_latent:
            # 添加小的噪声以避免零向量（数值稳定性）
            z_t = z_t + 1e-8 * torch.randn_like(z_t)
            z_t = F.normalize(z_t, p=2, dim=1)
        
        # 预测速度场
        v_pred = self.backbone(z_t, t, cond)
        
        # 目标速度场：v = z2 - z1
        v_target = z2 - z1
        
        # 计算损失（每个样本）
        batchwise_loss = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=list(range(1, v_pred.ndim)))
        
        # 总损失
        loss = batchwise_loss.mean()
        
        # 用于记录的损失字典
        loss_dict = [(t[i].item(), batchwise_loss[i].item()) for i in range(batch_size)]
        
        return loss, loss_dict
    
    @torch.no_grad()
    def sample(
        self,
        z_start: torch.Tensor,
        sample_steps: int = 50,
        cond: Optional[torch.Tensor] = None,
        null_cond: Optional[torch.Tensor] = None,
        cfg_scale: float = 2.0,
        normalize_latent: bool = True,
    ) -> List[torch.Tensor]:
        """
        从 z1 采样到 z2
        
        Args:
            z_start: 起点 z1 [B, latent_dim]
            sample_steps: 采样步数
            cond: 条件信息
            null_cond: 无条件信息（用于 CFG）
            cfg_scale: CFG 强度
            normalize_latent: 是否在每步后归一化潜空间向量（用于 scimilarity）
        
        Returns:
            采样轨迹 [z_0, z_1, ..., z_T]
        """
        z = z_start.clone()
        batch_size = z.shape[0]
        device = z.device
        dt = 1.0 / sample_steps
        
        trajectory = [z.cpu()]
        
        for step in range(sample_steps):
            t_current = step / sample_steps
            t = torch.full((batch_size,), t_current, device=device)
            
            # 预测速度场
            v = self.backbone(z, t, cond)
            
            # Classifier-Free Guidance
            if null_cond is not None and cfg_scale != 1.0:
                v_uncond = self.backbone(z, t, null_cond)
                v = v_uncond + cfg_scale * (v - v_uncond)
            
            # 欧拉步：z = z + v * dt
            z = z + v * dt
            
            # 🔧 关键修复：如果使用 scimilarity encoder，需要归一化到单位球面
            if normalize_latent:
                # 添加小的噪声以避免零向量（数值稳定性）
                z = z + 1e-8 * torch.randn_like(z)
                z = F.normalize(z, p=2, dim=1)
            
            trajectory.append(z.cpu())
        
        return trajectory


class RFInversion(RectifiedFlow):
    """
    Inversion 模式：z1 -> noise -> z2
    先反演到噪声空间，再从噪声生成目标
    """
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        cond1: Optional[torch.Tensor] = None,
        cond2: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        计算 Inversion 模式的损失
        训练两个方向：z1->noise 和 z2->noise
        
        Args:
            z1: 起点潜空间 [B, latent_dim]
            z2: 终点潜空间 [B, latent_dim]
            cond1: z1 的条件信息
            cond2: z2 的条件信息
        
        Returns:
            loss: 损失值
            loss_dict: 损失字典
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # 采样时间步
        t = self.sample_timestep(batch_size, device)
        t_exp = t.view(batch_size, *([1] * (z1.ndim - 1)))
        
        # 分别训练两个方向
        # 方向 1：z1 -> noise
        noise1 = torch.randn_like(z1)
        if self.normalize_latent:
            noise1 = F.normalize(noise1, p=2, dim=1)  # 噪声也归一化到单位球面
        z_t1 = (1 - t_exp) * z1 + t_exp * noise1
        if self.normalize_latent:
            z_t1 = z_t1 + 1e-8 * torch.randn_like(z_t1)  # 数值稳定性
            z_t1 = F.normalize(z_t1, p=2, dim=1)
        v_pred1 = self.backbone(z_t1, t, cond1)
        v_target1 = noise1 - z1
        
        # 方向 2：z2 -> noise
        noise2 = torch.randn_like(z2)
        if self.normalize_latent:
            noise2 = F.normalize(noise2, p=2, dim=1)  # 噪声也归一化到单位球面
        z_t2 = (1 - t_exp) * z2 + t_exp * noise2
        if self.normalize_latent:
            z_t2 = z_t2 + 1e-8 * torch.randn_like(z_t2)  # 数值稳定性
            z_t2 = F.normalize(z_t2, p=2, dim=1)
        v_pred2 = self.backbone(z_t2, t, cond2)
        v_target2 = noise2 - z2
        
        # 合并计算损失
        z_t = torch.cat([z_t1, z_t2], dim=0)
        v_pred = torch.cat([v_pred1, v_pred2], dim=0)
        v_target = torch.cat([v_target1, v_target2], dim=0)
        
        # 计算损失
        batchwise_loss = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=list(range(1, v_pred.ndim)))
        loss = batchwise_loss.mean()
        
        # 损失字典
        t_full = torch.cat([t, t], dim=0)
        loss_dict = [(t_full[i].item(), batchwise_loss[i].item()) for i in range(len(t_full))]
        
        return loss, loss_dict
    
    @torch.no_grad()
    def sample(
        self,
        z_start: torch.Tensor,
        sample_steps: int = 50,
        cond_start: Optional[torch.Tensor] = None,
        cond_target: Optional[torch.Tensor] = None,
        null_cond: Optional[torch.Tensor] = None,
        cfg_scale: float = 2.0,
        normalize_latent: bool = True,
    ) -> List[torch.Tensor]:
        """
        从 z1 反演到噪声，再从噪声生成 z2
        
        Args:
            z_start: 起点 z1 [B, latent_dim]
            sample_steps: 采样步数（每个阶段）
            cond_start: z1 的条件信息
            cond_target: z2 的条件信息
            null_cond: 无条件信息（用于 CFG）
            cfg_scale: CFG 强度
            normalize_latent: 是否在每步后归一化潜空间向量（用于 scimilarity）
        
        Returns:
            采样轨迹
        """
        z = z_start.clone()
        batch_size = z.shape[0]
        device = z.device
        dt = 1.0 / sample_steps
        
        trajectory = []
        
        # 阶段 1：z1 -> noise（正向）
        for step in range(sample_steps):
            t_current = step / sample_steps
            t = torch.full((batch_size,), t_current, device=device)
            
            v = self.backbone(z, t, cond_start)
            
            if null_cond is not None and cfg_scale != 1.0:
                v_uncond = self.backbone(z, t, null_cond)
                v = v_uncond + cfg_scale * (v - v_uncond)
            
            z = z + v * dt
            
            # 🔧 关键修复：如果使用 scimilarity encoder，需要归一化到单位球面
            if normalize_latent:
                z = z + 1e-8 * torch.randn_like(z)  # 数值稳定性
                z = F.normalize(z, p=2, dim=1)
            
            trajectory.append(z.cpu())
        
        # 阶段 2：noise -> z2（反向）
        for step in range(sample_steps):
            t_current = 1.0 - step / sample_steps
            t = torch.full((batch_size,), t_current, device=device)
            
            v = self.backbone(z, t, cond_target)
            
            if null_cond is not None and cfg_scale != 1.0:
                v_uncond = self.backbone(z, t, null_cond)
                v = v_uncond + cfg_scale * (v - v_uncond)
            
            z = z - v * dt
            
            # 🔧 关键修复：如果使用 scimilarity encoder，需要归一化到单位球面
            if normalize_latent:
                z = z + 1e-8 * torch.randn_like(z)  # 数值稳定性
                z = F.normalize(z, p=2, dim=1)
            
            trajectory.append(z.cpu())
        
        return trajectory


class RTFSystem(pl.LightningModule):
    """
    Rectified Flow 训练系统（PyTorch Lightning）
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        ae_encoder: nn.Module,
        ae_decoder: nn.Module,
    ):
        """
        Args:
            cfg: 配置对象（OmegaConf 或字典）
            ae_encoder: 预训练的 AE Encoder（已冻结）
            ae_decoder: 预训练的 AE Decoder（已冻结，用于计算重建误差）
        """
        super().__init__()
        
        # 如果 cfg 是 OmegaConf 对象，转换为字典；否则直接使用
        if isinstance(cfg, DictConfig):
            self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True), ignore=['ae_encoder', 'ae_decoder'])
            self.cfg = cfg
        else:
            # 从 checkpoint 加载时，cfg 已经是字典
            self.save_hyperparameters(cfg, ignore=['ae_encoder', 'ae_decoder'])
            self.cfg = OmegaConf.create(cfg)
        
        # 统一使用 self.cfg 访问配置
        cfg = self.cfg
        
        # 保存 AE Encoder 和 Decoder
        self.ae_encoder = ae_encoder
        self.ae_encoder.eval()
        for param in self.ae_encoder.parameters():
            param.requires_grad = False
        
        self.ae_decoder = ae_decoder
        self.ae_decoder.eval()
        for param in self.ae_decoder.parameters():
            param.requires_grad = False
        
        # 加载骨干网络配置
        backbone_config_path = f"config/backbones/{cfg.model.backbone}.yaml"
        backbone_cfg = OmegaConf.load(backbone_config_path)
        backbone_cfg_dict = OmegaConf.to_container(backbone_cfg, resolve=True)
        
        # 注入条件配置
        if cfg.model.use_cond and cfg.model.cond_dim is not None:
            if backbone_cfg_dict.get('use_class_cond', False):
                print("Warning: use_class_cond is enabled in backbone config but use_cond is also enabled.")
                print("Disabling use_class_cond and enabling use_vector_cond.")
                backbone_cfg_dict['use_class_cond'] = False
            
            backbone_cfg_dict['use_vector_cond'] = True
            backbone_cfg_dict['vector_cond_dim'] = cfg.model.cond_dim
            print(f"启用向量条件: dim={cfg.model.cond_dim}")
        
        # 创建骨干网络
        backbone = create_backbone(
            cfg.model.backbone,
            backbone_cfg_dict,
            cfg.model.latent_dim
        )
        
        # 创建 RTF 模型
        normalize_latent = getattr(cfg.model, 'normalize_latent', True)
        
        if cfg.model.mode == "direct":
            self.model = RFDirect(
                backbone, 
                ln_noise=cfg.model.ln_noise,
                normalize_latent=normalize_latent
            )
        elif cfg.model.mode == "inversion":
            self.model = RFInversion(
                backbone, 
                ln_noise=cfg.model.ln_noise,
                normalize_latent=normalize_latent
            )
        else:
            raise ValueError(f"Unknown mode: {cfg.model.mode}")
        
        # 输出目录
        self.output_dir = Path(cfg.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 用于统计损失分布
        self._reset_loss_bins()
    
    def _reset_loss_bins(self):
        """重置损失统计桶"""
        self.loss_bins = {i: 0.0 for i in range(10)}
        self.loss_counts = {i: 1e-6 for i in range(10)}
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        
        # 学习率调度器
        if self.cfg.training.scheduler.type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.training.scheduler.T_max,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    # test
    # 测试
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage == "fit" or stage is None:
            from dataset import TemporalCellDataset
            
            # 尝试加载目标基因列表（如果配置了）
            target_genes = None
            if hasattr(self.cfg.data, 'target_genes_path') and self.cfg.data.target_genes_path:
                path = Path(self.cfg.data.target_genes_path)
                if path.exists():
                    print(f"加载目标基因列表: {path}")
                    with open(path, 'r') as f:
                        target_genes = [line.strip() for line in f if line.strip()]
                    print(f"目标基因数量: {len(target_genes)}")
            
            # 使用 AE 的基因数作为 max_genes（如果配置了）
            max_genes = self.cfg.model.n_genes if hasattr(self.cfg.model, 'n_genes') and self.cfg.model.n_genes > 0 else None
            
            self.train_dataset = TemporalCellDataset(
                data=self.cfg.data.data_path,
                max_genes=max_genes,
                target_genes=target_genes,
                valid_pairs_only=self.cfg.data.valid_pairs_only,
                time_col=self.cfg.data.time_col,
                next_cell_col=self.cfg.data.next_cell_col,
                verbose=True,
            )
            
            print(f"训练数据集大小: {len(self.train_dataset)}")
            print(f"基因数量: {self.train_dataset.n_genes}")
            
            # 验证基因数与 AE 一致
            if max_genes is not None and self.train_dataset.n_genes != max_genes:
                print(f"Warning: 数据集基因数 ({self.train_dataset.n_genes}) 与预期 ({max_genes}) 不一致")
    
    def train_dataloader(self):
        """训练数据加载器"""
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
    
    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """使用 AE Encoder 编码到潜空间"""
        return self.ae_encoder(x)
    
    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """使用 AE Decoder 从潜空间解码"""
        return self.ae_decoder(z)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """训练步骤"""
        x_cur = batch["x_cur"]  # [B, n_genes]
        x_next = batch["x_next"]  # [B, n_genes]
        t_cur = batch["t_cur"]  # [B]
        t_next = batch["t_next"]  # [B]
        
        # 编码到潜空间
        with torch.no_grad():
            z_cur = self.encode_to_latent(x_cur)
            z_next = self.encode_to_latent(x_next)
        
        # 🔍 诊断信息（每100步打印一次）
        if batch_idx % 100 == 0:
            with torch.no_grad():
                print(f"\n[诊断 Batch {batch_idx}]")
                print(f"  原始空间 x_cur: min={x_cur.min():.4f}, max={x_cur.max():.4f}, "
                      f"mean={x_cur.mean():.4f}, std={x_cur.std():.4f}")
                print(f"  原始空间 x_next: min={x_next.min():.4f}, max={x_next.max():.4f}, "
                      f"mean={x_next.mean():.4f}, std={x_next.std():.4f}")
                print(f"  潜空间 z_cur: min={z_cur.min():.4f}, max={z_cur.max():.4f}, "
                      f"mean={z_cur.mean():.4f}, std={z_cur.std():.4f}")
                print(f"  潜空间 z_next: min={z_next.min():.4f}, max={z_next.max():.4f}, "
                      f"mean={z_next.mean():.4f}, std={z_next.std():.4f}")
                print(f"  速度场 v_target (z_next - z_cur): "
                      f"norm_mean={torch.norm(z_next - z_cur, dim=-1).mean():.6f}")
        
        # 准备条件信息（如果使用）
        cond = None
        if self.cfg.model.use_cond:
            cond = torch.stack([t_cur, t_next], dim=-1)  # [B, 2]
        
        # 计算损失
        if self.cfg.model.mode == "direct":
            loss, loss_dict = self.model(z_cur, z_next, cond)
        else:  # inversion
            cond1 = torch.stack([t_cur], dim=-1) if self.cfg.model.use_cond else None
            cond2 = torch.stack([t_next], dim=-1) if self.cfg.model.use_cond else None
            loss, loss_dict = self.model(z_cur, z_next, cond1, cond2)
        
        # 记录损失
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True)
        
        # 统计损失分布
        for t_val, l_val in loss_dict:
            bin_idx = int(t_val * 10)
            if 0 <= bin_idx < 10:
                self.loss_bins[bin_idx] += l_val
                self.loss_counts[bin_idx] += 1.0
        
        return loss
    
    def on_train_epoch_end(self):
        """训练 epoch 结束"""
        # 记录各时间段损失
        for i in range(10):
            avg_loss = self.loss_bins[i] / self.loss_counts[i]
            self.log(f"loss_bin_{i}", avg_loss, prog_bar=False, on_epoch=True)
        
        # 采样（如果需要）
        if self.current_epoch % self.cfg.training.sample_every_n_epochs == 0:
            self._sample_and_save(self.current_epoch)
        
        # 重置统计
        self._reset_loss_bins()
    
    @torch.no_grad()
    def _sample_and_save(self, epoch: int):
        """采样并保存"""
        self.model.eval()
        
        # 获取一个批次
        batch = next(iter(self.train_dataloader()))
        x_cur = batch["x_cur"][:8].to(self.device)
        x_next = batch["x_next"][:8].to(self.device)
        t_cur = batch["t_cur"][:8].to(self.device)
        t_next = batch["t_next"][:8].to(self.device)
        
        # 编码
        z_cur = self.encode_to_latent(x_cur)
        z_next = self.encode_to_latent(x_next)
        
        # 准备条件
        cond = None
        null_cond = None
        
        if self.cfg.model.use_cond:
            if self.cfg.model.mode == "direct":
                cond = torch.stack([t_cur, t_next], dim=-1)
                null_cond = torch.zeros_like(cond)
            else:
                # Inversion 模式下，条件维度是 1 (单独的时间点)
                # cond_start 和 cond_target 分别构建
                # null_cond 应该与 cond_start/target 维度一致 [B, 1]
                null_cond = torch.zeros(x_cur.shape[0], 1, device=self.device)
        
        # 采样（如果使用 scimilarity encoder，需要归一化）
        normalize_latent = getattr(self.cfg.model, 'normalize_latent', True)
        
        if self.cfg.model.mode == "direct":
            trajectory = self.model.sample(
                z_cur,
                sample_steps=self.cfg.training.sample_steps,
                cond=cond,
                null_cond=null_cond,
                cfg_scale=self.cfg.training.cfg_scale,
                normalize_latent=normalize_latent,
            )
        else:
            cond_start = torch.stack([t_cur], dim=-1) if self.cfg.model.use_cond else None
            cond_target = torch.stack([t_next], dim=-1) if self.cfg.model.use_cond else None
            trajectory = self.model.sample(
                z_cur,
                sample_steps=self.cfg.training.sample_steps,
                cond_start=cond_start,
                cond_target=cond_target,
                null_cond=null_cond,
                cfg_scale=self.cfg.training.cfg_scale,
                normalize_latent=normalize_latent,
            )
        
        # 🔧 正确计算重建误差：在原始空间而不是潜空间
        z_final = trajectory[-1].to(self.device)
        
        # 解码到原始空间
        x_reconstructed = self.decode_from_latent(z_final)
        
        # --- 🔍 调试：检查 AE 的理论上限 ---
        # 直接重建 z_next (目标潜向量)，看看 AE 自己能不能重建回去
        x_next_ae_recon = self.decode_from_latent(z_next)
        ae_recon_error = F.mse_loss(x_next_ae_recon, x_next).item()
        from models.utils import compute_correlation
        ae_correlation = compute_correlation(x_next, x_next_ae_recon)
        
        print(f"  [DEBUG] AE 直接重建误差: {ae_recon_error:.6f}")
        print(f"  [DEBUG] AE 直接重建相关性: {ae_correlation:.4f}")
        self.log("ae_oracle_correlation", ae_correlation, on_epoch=True)
        # ----------------------------------
        
        # 在原始空间计算重建误差
        recon_error_original = F.mse_loss(x_reconstructed, x_next).item()
        
        # 同时记录潜空间误差用于对比
        recon_error_latent = F.mse_loss(z_final, z_next).item()
        
        # 计算相关性（衡量重建质量）
        from models.utils import compute_correlation
        correlation = compute_correlation(x_next, x_reconstructed)
        
        self.log("sample_recon_error_original", recon_error_original, on_epoch=True)
        self.log("sample_recon_error_latent", recon_error_latent, on_epoch=True)
        self.log("sample_correlation", correlation, on_epoch=True)
        
        print(f"Epoch {epoch}: 采样完成")
        print(f"  原始空间重建误差: {recon_error_original:.6f}")
        print(f"  潜空间重建误差: {recon_error_latent:.6f}")
        print(f"  重建相关性: {correlation:.4f}")
        
        self.model.train()

