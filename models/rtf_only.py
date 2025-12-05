"""
Rectified Flow Only 模型
不在潜空间训练，直接在原始基因空间训练
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


class RFDirectOnly(nn.Module):
    """
    Direct 模式：x1 -> x2
    直接从起点到终点的线性插值
    支持可选的一致性损失
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
        use_consistency: bool = False,
        consistency_weight: float = 0.1,
    ) -> tuple:
        """
        计算损失

        Args:
            x1: 起点 [B, n_genes]
            x2: 终点 [B, n_genes]
            cond: 条件信息
            use_consistency: 是否使用一致性损失
            consistency_weight: 一致性损失权重
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

        # 计算 flow matching 损失
        batchwise_loss = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=list(range(1, v_pred.ndim)))
        flow_loss = batchwise_loss.mean()

        # 一致性损失
        consistency_loss = torch.tensor(0.0, device=device)
        if use_consistency:
            consistency_loss = self._compute_consistency_loss(x1, x2, cond)

        # 总损失
        loss = flow_loss + consistency_weight * consistency_loss

        # 用于记录的损失字典
        loss_dict = [(t[i].item(), batchwise_loss[i].item()) for i in range(batch_size)]

        return loss, loss_dict, {"flow_loss": flow_loss.item(), "consistency_loss": consistency_loss.item()}

    def _compute_consistency_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        n_steps: int = 2,
    ) -> torch.Tensor:
        """
        计算一致性损失：轨迹上不同时间点预测的终点应该一致

        思路：
        1. 采样两个时间点 t1 < t2
        2. 从 x_t1 用模型预测的速度积分到 t2，得到 x_t2_pred
        3. 直接插值得到 x_t2_true
        4. 一致性损失 = ||x_t2_pred - x_t2_true||^2
        """
        batch_size = x1.shape[0]
        device = x1.device

        # 采样两个有序时间点 t1 < t2
        t1 = torch.rand(batch_size, device=device) * 0.5  # t1 ∈ [0, 0.5]
        t2 = t1 + torch.rand(batch_size, device=device) * (1.0 - t1)  # t2 ∈ [t1, 1]

        t1_exp = t1.view(batch_size, *([1] * (x1.ndim - 1)))
        t2_exp = t2.view(batch_size, *([1] * (x1.ndim - 1)))

        # 直接插值得到 x_t1 和 x_t2
        x_t1 = (1 - t1_exp) * x1 + t1_exp * x2
        x_t2_true = (1 - t2_exp) * x1 + t2_exp * x2

        if self.normalize:
            x_t1 = F.normalize(x_t1, p=2, dim=1)
            x_t2_true = F.normalize(x_t2_true, p=2, dim=1)

        # 用模型从 x_t1 积分到 t2
        dt = (t2 - t1) / n_steps
        x_current = x_t1.clone()

        for step in range(n_steps):
            t_current = t1 + step * dt
            v = self.backbone(x_current, t_current, cond)
            x_current = x_current + v * dt.view(-1, *([1] * (x1.ndim - 1)))

            if self.normalize:
                x_current = F.normalize(x_current, p=2, dim=1)

        x_t2_pred = x_current

        # 一致性损失
        consistency_loss = F.mse_loss(x_t2_pred, x_t2_true)

        return consistency_loss

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


class RFInversionOnly(nn.Module):
    """
    Inversion 模式：x1 -> noise -> x2
    先反演到噪声空间，再从噪声生成目标
    在原始基因空间操作
    支持可选的一致性损失
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
        cond1: Optional[torch.Tensor] = None,
        cond2: Optional[torch.Tensor] = None,
        use_consistency: bool = False,
        consistency_weight: float = 0.1,
    ) -> tuple:
        """
        计算 Inversion 模式的损失
        训练两个方向：x1->noise 和 x2->noise

        Args:
            x1: 起点 [B, n_genes]
            x2: 终点 [B, n_genes]
            cond1: x1 的条件信息
            cond2: x2 的条件信息
            use_consistency: 是否使用一致性损失
            consistency_weight: 一致性损失权重
        """
        batch_size = x1.shape[0]
        device = x1.device

        # 采样时间步
        t = self.sample_timestep(batch_size, device)
        t_exp = t.view(batch_size, *([1] * (x1.ndim - 1)))

        # 方向 1：x1 -> noise
        noise1 = torch.randn_like(x1)
        if self.normalize:
            noise1 = F.normalize(noise1, p=2, dim=1)
        x_t1 = (1 - t_exp) * x1 + t_exp * noise1
        if self.normalize:
            x_t1 = x_t1 + 1e-8 * torch.randn_like(x_t1)
            x_t1 = F.normalize(x_t1, p=2, dim=1)
        v_pred1 = self.backbone(x_t1, t, cond1)
        v_target1 = noise1 - x1

        # 方向 2：x2 -> noise
        noise2 = torch.randn_like(x2)
        if self.normalize:
            noise2 = F.normalize(noise2, p=2, dim=1)
        x_t2 = (1 - t_exp) * x2 + t_exp * noise2
        if self.normalize:
            x_t2 = x_t2 + 1e-8 * torch.randn_like(x_t2)
            x_t2 = F.normalize(x_t2, p=2, dim=1)
        v_pred2 = self.backbone(x_t2, t, cond2)
        v_target2 = noise2 - x2

        # 合并计算损失
        v_pred = torch.cat([v_pred1, v_pred2], dim=0)
        v_target = torch.cat([v_target1, v_target2], dim=0)

        # 计算 flow matching 损失
        batchwise_loss = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=list(range(1, v_pred.ndim)))
        flow_loss = batchwise_loss.mean()

        # 一致性损失
        consistency_loss = torch.tensor(0.0, device=device)
        if use_consistency:
            # 对两个方向分别计算一致性损失
            consistency_loss1 = self._compute_consistency_loss(x1, noise1, cond1)
            consistency_loss2 = self._compute_consistency_loss(x2, noise2, cond2)
            consistency_loss = (consistency_loss1 + consistency_loss2) / 2

        # 总损失
        loss = flow_loss + consistency_weight * consistency_loss

        # 损失字典
        t_full = torch.cat([t, t], dim=0)
        loss_dict = [(t_full[i].item(), batchwise_loss[i].item()) for i in range(len(t_full))]

        return loss, loss_dict, {"flow_loss": flow_loss.item(), "consistency_loss": consistency_loss.item()}

    def _compute_consistency_loss(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        n_steps: int = 2,
    ) -> torch.Tensor:
        """
        计算一致性损失：轨迹上不同时间点预测的终点应该一致
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # 采样两个有序时间点 t1 < t2
        t1 = torch.rand(batch_size, device=device) * 0.5
        t2 = t1 + torch.rand(batch_size, device=device) * (1.0 - t1)

        t1_exp = t1.view(batch_size, *([1] * (x_start.ndim - 1)))
        t2_exp = t2.view(batch_size, *([1] * (x_start.ndim - 1)))

        # 直接插值得到 x_t1 和 x_t2
        x_t1 = (1 - t1_exp) * x_start + t1_exp * x_end
        x_t2_true = (1 - t2_exp) * x_start + t2_exp * x_end

        if self.normalize:
            x_t1 = F.normalize(x_t1, p=2, dim=1)
            x_t2_true = F.normalize(x_t2_true, p=2, dim=1)

        # 用模型从 x_t1 积分到 t2
        dt = (t2 - t1) / n_steps
        x_current = x_t1.clone()

        for step in range(n_steps):
            t_current = t1 + step * dt
            v = self.backbone(x_current, t_current, cond)
            x_current = x_current + v * dt.view(-1, *([1] * (x_start.ndim - 1)))

            if self.normalize:
                x_current = F.normalize(x_current, p=2, dim=1)

        x_t2_pred = x_current

        # 一致性损失
        consistency_loss = F.mse_loss(x_t2_pred, x_t2_true)

        return consistency_loss

    @torch.no_grad()
    def sample(
        self,
        x_start: torch.Tensor,
        sample_steps: int = 50,
        cond_start: Optional[torch.Tensor] = None,
        cond_target: Optional[torch.Tensor] = None,
        null_cond: Optional[torch.Tensor] = None,
        cfg_scale: float = 2.0,
        normalize: bool = False,
    ) -> List[torch.Tensor]:
        """
        从 x1 反演到噪声，再从噪声生成 x2

        Args:
            x_start: 起点 x1 [B, n_genes]
            sample_steps: 采样步数（每个阶段）
            cond_start: x1 的条件信息（源时间）
            cond_target: x2 的条件信息（目标时间）
            null_cond: 无条件信息（用于 CFG）
            cfg_scale: CFG 强度
            normalize: 是否归一化

        Returns:
            采样轨迹
        """
        x = x_start.clone()
        batch_size = x.shape[0]
        device = x.device
        dt = 1.0 / sample_steps

        trajectory = [x.cpu()]

        # 阶段 1：x1 -> noise（正向）
        for step in range(sample_steps):
            t_current = step / sample_steps
            t = torch.full((batch_size,), t_current, device=device)

            v = self.backbone(x, t, cond_start)

            if null_cond is not None and cfg_scale != 1.0:
                v_uncond = self.backbone(x, t, null_cond)
                v = v_uncond + cfg_scale * (v - v_uncond)

            x = x + v * dt

            if normalize:
                x = x + 1e-8 * torch.randn_like(x)
                x = F.normalize(x, p=2, dim=1)

            trajectory.append(x.cpu())

        # 阶段 2：noise -> x2（反向）
        for step in range(sample_steps):
            t_current = 1.0 - step / sample_steps
            t = torch.full((batch_size,), t_current, device=device)

            v = self.backbone(x, t, cond_target)

            if null_cond is not None and cfg_scale != 1.0:
                v_uncond = self.backbone(x, t, null_cond)
                v = v_uncond + cfg_scale * (v - v_uncond)

            x = x - v * dt

            if normalize:
                x = x + 1e-8 * torch.randn_like(x)
                x = F.normalize(x, p=2, dim=1)

            trajectory.append(x.cpu())

        return trajectory


class RTFOnlySystem(pl.LightningModule):
    """
    Rectified Flow Only 训练系统（无 AE）
    支持 Direct 和 Inversion 两种模式
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

        # 确定输入维度
        # 如果 n_genes 为 0 或未指定，从 target_genes_path 读取基因列表确定维度
        input_dim = cfg.model.n_genes if cfg.model.n_genes > 0 else None

        if input_dim is None or input_dim == 0:
            # 从 target_genes_path 读取基因数量
            if hasattr(cfg.data, 'target_genes_path') and cfg.data.target_genes_path:
                path = Path(cfg.data.target_genes_path)
                if path.exists():
                    print(f"n_genes=0, 从 {path} 读取基因列表...")
                    with open(path, 'r') as f:
                        target_genes = [line.strip() for line in f if line.strip()]
                    input_dim = len(target_genes)
                    print(f"基因数量: {input_dim}")
                else:
                    raise ValueError(f"target_genes_path 文件不存在: {path}")
            else:
                raise ValueError("n_genes=0 但未指定 target_genes_path")

        self.n_genes = input_dim

        # 加载骨干网络配置
        backbone_config_path = f"config/backbones/{cfg.model.backbone}.yaml"
        backbone_cfg = OmegaConf.load(backbone_config_path)
        backbone_cfg_dict = OmegaConf.to_container(backbone_cfg, resolve=True)

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

        # 获取模式（默认 direct）
        mode = getattr(cfg.model, 'mode', 'direct')
        print(f"RTF Only 模式: {mode}")

        # 创建 RTF 模型
        if mode == "direct":
            self.model = RFDirectOnly(
                backbone,
                ln_noise=cfg.model.ln_noise,
                normalize=cfg.model.normalize
            )
        elif mode == "inversion":
            self.model = RFInversionOnly(
                backbone,
                ln_noise=cfg.model.ln_noise,
                normalize=cfg.model.normalize
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'direct' or 'inversion'.")

        self.mode = mode

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

            # 读取目标基因列表
            target_genes = None
            if hasattr(self.cfg.data, 'target_genes_path') and self.cfg.data.target_genes_path:
                path = Path(self.cfg.data.target_genes_path)
                if path.exists():
                    with open(path, 'r') as f:
                        target_genes = [line.strip() for line in f if line.strip()]
                    print(f"使用目标基因列表: {len(target_genes)} 个基因")

            # 确定 max_genes 参数
            # 如果 n_genes > 0，使用它；否则使用 target_genes 的长度（不限制）
            max_genes = self.cfg.model.n_genes if self.cfg.model.n_genes > 0 else None

            self.train_dataset = TemporalCellDataset(
                data=self.cfg.data.data_path,
                max_genes=max_genes,
                target_genes=target_genes,
                valid_pairs_only=self.cfg.data.valid_pairs_only,
                time_col=self.cfg.data.time_col,
                next_cell_col=self.cfg.data.next_cell_col,
                verbose=True,
            )

            # 验证基因数与模型一致
            if self.n_genes != self.train_dataset.n_genes:
                print(f"Warning: 模型基因数 ({self.n_genes}) 与数据集 ({self.train_dataset.n_genes}) 不一致!")

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

        # 获取一致性损失配置
        use_consistency = getattr(self.cfg.training, 'use_consistency', False)
        consistency_weight = getattr(self.cfg.training, 'consistency_weight', 0.1)

        # 诊断信息
        if batch_idx % 100 == 0:
            with torch.no_grad():
                v_norm = torch.norm(x_next - x_cur, dim=-1).mean()
                print(f"\n[Batch {batch_idx}] v_target norm: {v_norm:.4f}")

        # 根据模式计算损失
        if self.mode == "direct":
            cond = None
            if self.cfg.model.use_cond:
                cond = torch.stack([t_cur, t_next], dim=-1)
            loss, loss_dict, loss_info = self.model(
                x_cur, x_next, cond,
                use_consistency=use_consistency,
                consistency_weight=consistency_weight
            )
        else:  # inversion
            cond1 = torch.stack([t_cur], dim=-1) if self.cfg.model.use_cond else None
            cond2 = torch.stack([t_next], dim=-1) if self.cfg.model.use_cond else None
            loss, loss_dict, loss_info = self.model(
                x_cur, x_next, cond1, cond2,
                use_consistency=use_consistency,
                consistency_weight=consistency_weight
            )

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("flow_loss", loss_info["flow_loss"], prog_bar=False, on_step=True, on_epoch=True)
        if use_consistency:
            self.log("consistency_loss", loss_info["consistency_loss"], prog_bar=True, on_step=True, on_epoch=True)

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

        # 根据模式采样
        if self.mode == "direct":
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
        else:  # inversion
            cond_start = torch.stack([t_cur], dim=-1) if self.cfg.model.use_cond else None
            cond_target = torch.stack([t_next], dim=-1) if self.cfg.model.use_cond else None
            null_cond = torch.zeros(x_cur.shape[0], 1, device=self.device) if self.cfg.model.use_cond else None

            trajectory = self.model.sample(
                x_cur,
                sample_steps=self.cfg.training.sample_steps,
                cond_start=cond_start,
                cond_target=cond_target,
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

        print(f"Epoch {epoch}: Sampled ({self.mode} mode)")
        print(f"  Recon Error: {recon_error:.6f}")
        print(f"  Correlation: {correlation:.4f}")

        self.model.train()

