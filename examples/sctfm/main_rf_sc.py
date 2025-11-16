
import argparse
import os
from pathlib import Path
from typing import Any, Optional, Dict, List

import fire
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models.rf import RFSet, RFTSet
from models.cae_sc import CondEncoder, CondTempEncoder
from datas.temperal_cell_data import TemporalCellDataset, collate_fn_temperal


class RFTemporalSystem(pl.LightningModule):
    def __init__(
        self,
        data_path: str,
        n_genes: int = 3000,
        latent_dim: int = 128,
        hidden_dim: List[int] = [128, 128],
        cond_dim: Optional[int] = None,
        batch_size: int = 128,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        output_dir: str = "output",
        sample_steps: int = 50,
        cfg: float = 2.0,
        num_workers: int = 4,
        valid_pairs_only: bool = True,
        time_col: str = "time",
        next_cell_col: str = "next_cell_id",
        dropout: float = 0.5,
        input_dropout: float = 0.1,
        cond_type: str = "continuous",
        use_time_embedding: bool = True,
        ln_noise: bool = True,
    ):
        """
        PyTorch Lightning模块，用于Rectified Flow在时序单细胞数据上的训练

        Args:
            data_path: 时序单细胞数据路径(.h5ad文件)
            n_genes: 基因数量
            latent_dim: 潜在空间维度
            hidden_dim: 隐藏层维度列表
            cond_dim: 条件维度，如果为None则不使用条件
            batch_size: 批次大小
            lr: 学习率
            weight_decay: 权重衰减
            output_dir: 输出目录
            sample_steps: 采样步数
            cfg: classifier-free guidance强度
            num_workers: DataLoader工作进程数
            valid_pairs_only: 是否只使用有效的时序对
            time_col: 时间列名
            next_cell_col: 下一个细胞ID列名
            dropout: 隐藏层dropout
            input_dropout: 输入层dropout
            cond_type: 条件类型 ("continuous" 或 "categorical")
            use_time_embedding: 是否使用时间嵌入
            ln_noise: 是否使用log-normal噪声分布
        """
        super().__init__()
        self.save_hyperparameters()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建CondEncoder作为骨干模型
        # self.encoder = CondEncoder(
        #     n_genes=n_genes,
        #     latent_dim=latent_dim,
        #     hidden_dim=hidden_dim,
        #     dropout=dropout,
        #     input_dropout=input_dropout,
        #     cond_dim=cond_dim,
        #     cond_type=cond_type,
        #     use_time_embedding=use_time_embedding,
        # )
        # self.model = RFSet(self.encoder, ln=ln_noise)

        # 用RFTSet包装encoder进行rectified flow训练
        self.encoder = CondTempEncoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            input_dropout=input_dropout,
            cond_dim=cond_dim,
            cond_type=cond_type,
            use_time_embedding=use_time_embedding,
        )
        self.model = RFTSet(self.encoder, ln=ln_noise)

        # 用于统计各时间段的损失
        self._reset_epoch_loss_bins()

    def _reset_epoch_loss_bins(self):
        """重置epoch损失统计"""
        self.lossbin = {i: 0.0 for i in range(10)}
        self.losscnt = {i: 1e-6 for i in range(10)}

    def prepare_data(self) -> None:
        """数据准备阶段（下载/预处理等）"""
        # 这里可以添加数据预处理逻辑
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """设置数据集"""
        if stage == "fit" or stage is None:
            self.train_dataset = TemporalCellDataset(
                data=self.hparams.data_path,
                valid_pairs_only=self.hparams.valid_pairs_only,
                time_col=self.hparams.time_col,
                next_cell_col=self.hparams.next_cell_col,
                verbose=True,
            )
            print(f"训练数据集加载完成，样本数量: {len(self.train_dataset)}")
            print(f"基因数量: {self.train_dataset.n_genes}")

    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn_temperal,
        )

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        # 可选：添加学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """训练步骤"""
        x_cur = batch["x_cur"]  # [B, n_genes]
        x_next = batch["x_next"]  # [B, n_genes]
        t_cur = batch["t_cur"]  # [B]
        t_next = batch["t_next"]  # [B]

        # 准备条件信息（如果使用）
        if self.hparams.cond_dim is not None:
            # 这里可以根据实际情况构造条件信息
            # 例如使用时间信息作为条件
            cond = torch.stack([t_cur, t_next], dim=-1)  # [B, 2]
            # 或者根据细胞类型等构造条件
        else:
            cond = None

        # Rectified Flow训练
        loss, blsct = self.model(x_cur, x_next, t_cur, t_next, cond)

        # 记录损失
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True)

        # 统计各时间段的损失
        for t, l in blsct:
            idx = int(t.item() * 10)
            if 0 <= idx < 10:
                self.lossbin[idx] += float(l)
                self.losscnt[idx] += 1.0

        return loss

    def on_train_epoch_end(self) -> None:
        """训练epoch结束回调"""
        # 记录各分段损失
        for i in range(10):
            avg_loss = self.lossbin[i] / self.losscnt[i]
            
            if self.trainer.is_global_zero:
                print(f"Epoch: {self.current_epoch}, bin {i} range loss: {avg_loss:.6f}")

            self.log(
                f"lossbin_{i}",
                avg_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # 采样和保存（仅在主进程）
        if self.trainer.is_global_zero and self.current_epoch % 1 == 0:
            self._sample_and_save(self.current_epoch)

        # 重置损失统计
        self._reset_epoch_loss_bins()

    @torch.no_grad()
    def _sample_and_save(self, epoch: int):
        """采样并保存结果"""
        self.model.eval()
        device = self.device

        # 从训练集中随机选择一些起始细胞
        n_samples = 8
        batch = next(iter(self.train_dataloader()))
        
        x1 = batch["x_cur"].to(device)  # 起始细胞
        t1 = batch["t_cur"].to(device)  # 起始时间
        t2 = batch["t_next"].to(device)  # 目标时间
        
        # 准备条件 TODO 临时处理
        if self.hparams.cond_dim is not None:
            cond = torch.stack([t1, t2], dim=-1)
            null_cond = torch.zeros_like(cond)  # 用于CFG的null条件
        else:
            cond = None
            null_cond = None

        # cond = None
        # null_cond = None
        # 采样轨迹
        trajectory = self.model.sample(
            x1=x1,
            t1=t1,
            t2=t2,
            cond=cond,
            null_cond=null_cond,
            sample_steps=self.hparams.sample_steps,
            cfg=self.hparams.cfg,
        )

        # 保存采样结果
        sample_dir = self.output_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        # # 保存起始和最终状态
        # torch.save({
        #     "epoch": epoch,
        #     "x_start": x1.cpu(),
        #     "x_end": trajectory[-1].cpu(),
        #     "x_target": batch["x_next"][:n_samples].cpu(),
        #     "t_start": t1.cpu(),
        #     "t_end": t2.cpu(),
        #     "trajectory": [x.cpu() for x in trajectory],
        # }, sample_dir / f"sample_epoch_{epoch}.pt")

        # 计算重构误差
        recon_error = nn.MSELoss()(
            trajectory[-1], 
            batch["x_next"].to(device)
        ).item()
        
        self.log("sample_recon_error", recon_error, on_epoch=True)
        
        print(f"Epoch {epoch}: 采样完成，重构误差: {recon_error:.6f}")
        
        self.model.train()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """验证步骤（可选）"""
        x_cur = batch["x_cur"]
        x_next = batch["x_next"] 
        t_cur = batch["t_cur"]
        t_next = batch["t_next"]

        # 准备条件
        if self.hparams.cond_dim is not None:
            cond = torch.stack([t_cur, t_next], dim=-1)
        else:
            cond = None

        # 计算验证损失
        val_loss, _ = self.model(x_cur, x_next, t_cur, t_next, cond)
        
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss


def train_rf_temporal(
    data_path: str,
    n_genes: int = 3000,
    latent_dim: int = 128,
    hidden_dim: List[int] = [128, 128],
    cond_dim: Optional[int] = None,  # 使用时间作为条件
    batch_size: int = 128,
    lr: float = 1e-4,
    # weight_decay: float = 1e-5,
    weight_decay: float = 0,
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: str = "auto",
    output_dir: str = "output_rf_temporal",
    sample_steps: int = 50,
    cfg: float = 2.0,
    num_workers: int = 4,
    **kwargs
):
    """
    训练Rectified Flow时序模型的主函数
    
    Args:
        data_path: 数据路径
        n_genes: 基因数量
        latent_dim: 潜在维度
        hidden_dim: 隐藏层维度列表
        cond_dim: 条件维度
        batch_size: 批次大小
        lr: 学习率
        weight_decay: 权重衰减
        max_epochs: 最大训练轮数
        accelerator: 加速器类型
        devices: 设备数量
        output_dir: 输出目录
        sample_steps: 采样步数
        cfg: CFG强度
        num_workers: 数据加载工作进程数
        **kwargs: 其他参数
    """
    
    # 创建训练系统
    system = RFTemporalSystem(
        data_path=data_path,
        n_genes=n_genes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        cond_dim=cond_dim,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        output_dir=output_dir,
        sample_steps=sample_steps,
        cfg=cfg,
        num_workers=num_workers,
        **kwargs
    )

    # 统计模型参数
    total_params = sum(p.numel() for p in system.model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 创建训练器
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        log_every_n_steps=50,
        check_val_every_n_epoch=5,
        enable_checkpointing=True,
        default_root_dir=output_dir,
        # gradient_clip_val=1.0,  # 梯度裁剪
    )

    # 开始训练
    trainer.fit(system)

    return system, trainer


if __name__ == "__main__":
    fire.Fire(train_rf_temporal)