
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Tuple, Dict, Any
# import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path

from scimilarity.nn_models import Encoder, Decoder


class Autoencoder(nn.Module):
    """
    基于 scimilarity 的 Encoder/Decoder 构建的 Autoencoder
    """
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 256,
        hidden_dim: Optional[list] = None,
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入基因表达数据 (batch_size, n_genes)
            
        Returns:
            reconstructed: 重建的基因表达数据 (batch_size, n_genes) 
            latent: 潜在表示 (batch_size, latent_dim)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到潜在空间"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码"""
        return self.decoder(z)


class AutoencoderSystem(pl.LightningModule):
    """
    PyTorch Lightning 包装的 Autoencoder 训练系统
    """
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 256,
        hidden_dim: Optional[list] = None,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        reconstruction_loss: str = "mse",  # "mse" or "poisson" or "nb"
        batch_size: int = 256,
        data_path: str = None,
        output_dir: str = "autoencoder_output",
        log_embeddings_every_n_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 模型
        self.autoencoder = Autoencoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        
        # 损失函数
        if reconstruction_loss == "mse":
            self.reconstruction_loss_fn = nn.MSELoss()
        elif reconstruction_loss == "poisson":
            self.reconstruction_loss_fn = nn.PoissonNLLLoss(log_input=False)
        else:
            raise ValueError(f"Unsupported reconstruction loss: {reconstruction_loss}")
        
        # 输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集相关
        self.data_path = data_path
        self.train_dataset = None
        self.val_dataset = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.autoencoder(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            # verbose=True,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage == "fit" or stage is None:
            from datas.temperal_cell_data import StaticCellDataset
            
            if self.data_path is None:
                raise ValueError("data_path must be provided")
            
            # 创建完整数据集
            full_dataset = StaticCellDataset(
                self.data_path,
                verbose=True,
                seed=42,
            )
            
            # 划分训练集和验证集 (80/20)
            n_total = len(full_dataset)
            n_train = int(0.8 * n_total)
            n_val = n_total - n_train
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )
            
            print(f"训练集大小: {len(self.train_dataset)}")
            print(f"验证集大小: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        from datas.temperal_cell_data import collate_fn_static
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_static,
        )
    
    def val_dataloader(self):
        from datas.temperal_cell_data import collate_fn_static
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_static,
        )
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """训练步骤"""
        x = batch  # batch shape: (batch_size, n_genes)
        
        # 前向传播
        x_reconstructed, latent = self.forward(x)
        
        # 计算重建损失
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)
        
        # 总损失
        total_loss = recon_loss
        
        # 记录指标
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=True)
        
        # 计算一些统计指标
        with torch.no_grad():
            mse = F.mse_loss(x_reconstructed, x)
            correlation = self._compute_correlation(x, x_reconstructed)
            
        self.log("train_mse", mse, on_step=True, on_epoch=True)
        self.log("train_correlation", correlation, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """验证步骤"""
        x = batch
        
        # 前向传播
        x_reconstructed, latent = self.forward(x)
        
        # 计算重建损失
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)
        total_loss = recon_loss
        
        # 记录指标
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        
        # 计算统计指标
        mse = F.mse_loss(x_reconstructed, x)
        correlation = self._compute_correlation(x, x_reconstructed)
        
        self.log("val_mse", mse, on_step=False, on_epoch=True)
        self.log("val_correlation", correlation, on_step=False, on_epoch=True)
        
        return {
            "val_loss": total_loss,
            "original": x.detach().cpu(),
            "reconstructed": x_reconstructed.detach().cpu(),
            "latent": latent.detach().cpu(),
        }
    
    def on_validation_epoch_end(self) -> None:
        """验证 epoch 结束时的操作"""
        if self.trainer.is_global_zero:
            pass
            # 每隔几个 epoch 保存一些可视化结果
            # if self.current_epoch % self.hparams.log_embeddings_every_n_epochs == 0:
                # self._save_reconstruction_examples()
                # self._save_latent_embeddings()
    
    def _compute_correlation(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """计算原始数据和重建数据的相关系数"""
        # 按样本计算相关系数，然后取平均
        batch_size = x.shape[0]
        correlations = []
        
        for i in range(batch_size):
            x_i = x[i].flatten()
            x_recon_i = x_recon[i].flatten()
            
            # 计算皮尔逊相关系数
            x_i_centered = x_i - x_i.mean()
            x_recon_i_centered = x_recon_i - x_recon_i.mean()
            
            numerator = (x_i_centered * x_recon_i_centered).sum()
            denominator = torch.sqrt((x_i_centered ** 2).sum() * (x_recon_i_centered ** 2).sum())
            
            if denominator > 0:
                corr = numerator / denominator
                correlations.append(corr)
        
        return torch.stack(correlations).mean() if correlations else torch.tensor(0.0)
    
    # @torch.no_grad()
    # def _save_reconstruction_examples(self):
    #     """保存重建示例"""
    #     self.eval()
    #
    #     # 获取一些验证样本
    #     val_loader = self.val_dataloader()
    #     batch = next(iter(val_loader))
    #     x = batch.to(self.device)
    #
    #     # 重建
    #     x_recon, latent = self.forward(x)
    #
    #     # 选择前几个样本进行可视化
    #     n_samples = min(4, x.shape[0])
    #
    #     fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    #     if n_samples == 1:
    #         axes = axes.reshape(2, 1)
    #
    #     for i in range(n_samples):
    #         # 原始数据
    #         axes[0, i].hist(x[i].cpu().numpy(), bins=50, alpha=0.7)
    #         axes[0, i].set_title(f'Original Sample {i+1}')
    #         axes[0, i].set_xlabel('Gene Expression')
    #         axes[0, i].set_ylabel('Frequency')
    #
    #         # 重建数据
    #         axes[1, i].hist(x_recon[i].cpu().numpy(), bins=50, alpha=0.7, color='orange')
    #         axes[1, i].set_title(f'Reconstructed Sample {i+1}')
    #         axes[1, i].set_xlabel('Gene Expression')
    #         axes[1, i].set_ylabel('Frequency')
    #
    #     plt.tight_layout()
    #     plt.savefig(self.output_dir / f"reconstruction_epoch_{self.current_epoch}.png", dpi=150)
    #     plt.close()
    #
    #     # 散点图比较
    #     fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 5))
    #     if n_samples == 1:
    #         axes = [axes]
    #
    #     for i in range(n_samples):
    #         x_orig = x[i].cpu().numpy()
    #         x_rec = x_recon[i].cpu().numpy()
    #
    #         axes[i].scatter(x_orig, x_rec, alpha=0.6, s=1)
    #         axes[i].plot([x_orig.min(), x_orig.max()], [x_orig.min(), x_orig.max()], 'r--', alpha=0.8)
    #         axes[i].set_xlabel('Original Expression')
    #         axes[i].set_ylabel('Reconstructed Expression')
    #         axes[i].set_title(f'Sample {i+1}')
    #
    #         # 计算相关系数
    #         corr = np.corrcoef(x_orig, x_rec)[0, 1]
    #         axes[i].text(0.05, 0.95, f'R = {corr:.3f}', transform=axes[i].transAxes,
    #                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    #
    #     plt.tight_layout()
    #     plt.savefig(self.output_dir / f"correlation_epoch_{self.current_epoch}.png", dpi=150)
    #     plt.close()
    #
    #     self.train()
    #
    # @torch.no_grad()
    # def _save_latent_embeddings(self):
    #     """保存潜在表示的可视化"""
    #     if self.hparams.latent_dim > 2:
    #         return  # 只在潜在维度较小时可视化
    #
    #     self.eval()
    #
    #     # 收集一些潜在表示
    #     latents = []
    #     val_loader = self.val_dataloader()
    #
    #     for i, batch in enumerate(val_loader):
    #         if i >= 10:  # 只取前10个batch
    #             break
    #         x = batch.to(self.device)
    #         _, latent = self.forward(x)
    #         latents.append(latent.cpu())
    #
    #     latents = torch.cat(latents, dim=0).numpy()
    #
    #     # 可视化
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(latents[:, 0], latents[:, 1], alpha=0.6, s=1)
    #     plt.xlabel('Latent Dimension 1')
    #     plt.ylabel('Latent Dimension 2')
    #     plt.title(f'Latent Space (Epoch {self.current_epoch})')
    #     plt.savefig(self.output_dir / f"latent_space_epoch_{self.current_epoch}.png", dpi=150)
    #     plt.close()
    #
    #     self.train()


def train_autoencoder(
    data_path: str,
    n_genes: int,
    latent_dim: int = 256,
    hidden_dim: Optional[list] = None,
    dropout_rate: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    reconstruction_loss: str = "mse",
    batch_size: int = 256,
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: str = "auto",
    output_dir: str = "autoencoder_output",
    log_embeddings_every_n_epochs: int = 10,
):
    """
    训练 Autoencoder 的主函数
    
    Args:
        data_path: 数据文件路径
        n_genes: 基因数量
        latent_dim: 潜在空间维度
        hidden_dim: 隐藏层维度列表
        dropout_rate: dropout 率
        learning_rate: 学习率
        weight_decay: 权重衰减
        reconstruction_loss: 重建损失函数类型
        batch_size: 批次大小
        max_epochs: 最大训练轮数
        accelerator: 加速器类型
        devices: 设备数量
        output_dir: 输出目录
        log_embeddings_every_n_epochs: 每隔几个epoch记录嵌入
    """
    
    # 创建系统
    system = AutoencoderSystem(
        n_genes=n_genes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        reconstruction_loss=reconstruction_loss,
        batch_size=batch_size,
        data_path=data_path,
        output_dir=output_dir,
        log_embeddings_every_n_epochs=log_embeddings_every_n_epochs,
    )
    
    # 打印模型信息
    model_size = sum(p.numel() for p in system.parameters() if p.requires_grad)
    print(f"模型参数数量: {model_size:,} ({model_size / 1e6:.2f}M)")
    print(f"数据路径: {data_path}")
    print(f"基因数量: {n_genes}")
    print(f"潜在维度: {latent_dim}")
    print(f"隐藏层维度: {hidden_dim}")
    
    # 创建 trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        log_every_n_steps=50,
        val_check_interval=0.5,  # 每半个epoch验证一次
        enable_checkpointing=True,
        default_root_dir=output_dir,
    )
    
    # 开始训练
    trainer.fit(system)
    
    print(f"训练完成！结果保存在: {output_dir}")
    return system, trainer


if __name__ == "__main__":
    import fire
    fire.Fire(train_autoencoder)
