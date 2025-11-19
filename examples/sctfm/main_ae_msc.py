
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from models.ae_msc import TAE


class SetAutoencoder(nn.Module):
    """
    基于 SequenceDit 的集合 Autoencoder
    将多细胞表达数据视为集合，进行编码解码
    """
    def __init__(
        self,
        n_genes: int,
        n_cells_per_sample: int = 5,
        latent_dim: int = 256,
        max_seq_len: int = 10,
        dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        class_dropout_prob: float = 0.1,
        num_classes: int = 10,
        use_pos_embedding: bool = True,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.n_cells_per_sample = n_cells_per_sample
        self.latent_dim = latent_dim
        self.dim = dim

        # 编码器：将多细胞集合映射到潜在表示
        self.encoder = TAE(
            input_dim=n_genes,
            max_seq_len=max_seq_len,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            use_pos_embedding=use_pos_embedding,
        )

        # 潜在表示投影层（将集合压缩到固定维度的潜在向量）
        # self.latent_projector = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim * n_cells_per_sample, latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim)
        # )
        #
        # # 解码器：从潜在表示恢复多细胞集合
        # self.latent_expander = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, dim * n_cells_per_sample),
        #     nn.LayerNorm(dim * n_cells_per_sample)
        # )

        self.decoder = TAE(
            input_dim=n_genes,
            max_seq_len=max_seq_len,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            use_pos_embedding=use_pos_embedding,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码多细胞集合到潜在空间

        Args:
            x: 多细胞表达集合 [batch_size, n_cells_per_sample, n_genes]
            t: 时间步 [batch_size]
            y: 条件标签 [batch_size]

        Returns:
            latent: 潜在表示 [batch_size, latent_dim]
        """
        # 通过 encoder 处理集合
        latent = self.encoder(x)  # [batch_size, n_cells_per_sample, dim]

        # 将集合展平并投影到潜在空间
        # batch_size = encoded_seq.shape[0]
        # encoded_flat = encoded_seq.reshape(batch_size, -1)  # [batch_size, n_cells_per_sample * dim]
        # latent = self.latent_projector(encoded_flat)  # [batch_size, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        从潜在空间解码到多细胞集合

        Args:
            latent: 潜在表示 [batch_size, latent_dim]
            t: 时间步 [batch_size]
            y: 条件标签 [batch_size]

        Returns:
            reconstructed: 重建的多细胞表达集合 [batch_size, n_cells_per_sample, n_genes]
        """
        # batch_size = latent.shape[0]
        #
        # # 将潜在向量扩展为集合
        # expanded = self.latent_expander(latent)  # [batch_size, n_cells_per_sample * dim]
        # expanded_seq = expanded.reshape(
        #     batch_size, self.n_cells_per_sample, self.dim
        # )  # [batch_size, n_cells_per_sample, dim]
        #
        # # 通过解码器重建集合
        # reconstructed = self.decoder(expanded_seq, t, y)  # [batch_size, n_cells_per_sample, n_genes]
        reconstructed = self.decoder(latent)  # [batch_size, n_cells_per_sample, n_genes]

        return reconstructed

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：编码-解码

        Args:
            x: 输入多细胞集合 [batch_size, n_cells_per_sample, n_genes]
            t: 时间步 [batch_size]
            y: 条件标签 [batch_size]

        Returns:
            reconstructed: 重建集合 [batch_size, n_cells_per_sample, n_genes]
            latent: 潜在表示 [batch_size, latent_dim]
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class SetAutoencoderSystem(pl.LightningModule):
    """
    PyTorch Lightning 包装的集合 Autoencoder 训练系统
    """
    def __init__(
        self,
        n_genes: int,
        n_cells_per_sample: int = 5,
        latent_dim: int = 256,
        max_seq_len: int = 10,
        dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        class_dropout_prob: float = 0.1,
        num_classes: int = 10,
        use_pos_embedding: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        reconstruction_loss: str = "mse",
        batch_size: int = 32,
        data_path: str = None,
        output_dir: str = "set_autoencoder_output",
        log_embeddings_every_n_epochs: int = 10,
        # 数据集特定参数
        sampling_with_replacement: bool = True,
        dataset_seed: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 模型
        self.autoencoder = SetAutoencoder(
            n_genes=n_genes,
            n_cells_per_sample=n_cells_per_sample,
            latent_dim=latent_dim,
            max_seq_len=max_seq_len,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            use_pos_embedding=use_pos_embedding,
        )

        # 损失函数
        if reconstruction_loss == "mse":
            self.reconstruction_loss_fn = nn.MSELoss()
        elif reconstruction_loss == "poisson":
            self.reconstruction_loss_fn = nn.PoissonNLLLoss(log_input=False)
        elif reconstruction_loss == "smooth_l1":
            self.reconstruction_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported reconstruction loss: {reconstruction_loss}")

        # 输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据集相关
        self.data_path = data_path
        self.train_dataset = None
        self.val_dataset = None

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.autoencoder(x, t, y)

    def configure_optimizers(self):
        """配置优化器"""
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
            from datas.temperal_cell_data import MultiCellDataset
            
            if self.data_path is None:
                raise ValueError("data_path must be provided")
            
            # 创建完整数据集
            full_dataset = MultiCellDataset(
                self.data_path,
                n_cells_per_sample=self.hparams.n_cells_per_sample,
                sampling_with_replacement=self.hparams.sampling_with_replacement,
                verbose=True,
                seed=self.hparams.dataset_seed,
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
        """训练数据加载器"""
        from datas.temperal_cell_data import collate_fn_multi_cell
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_multi_cell,
        )
    
    def val_dataloader(self):
        """验证数据加载器"""
        from datas.temperal_cell_data import collate_fn_multi_cell
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_multi_cell,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        # batch["expressions"]: [batch_size, n_cells_per_sample, n_genes]
        # batch["cell_indices"]: [batch_size, n_cells_per_sample]

        x = batch["expressions"]  # 多细胞表达集合
        batch_size = x.shape[0]

        # 创建虚拟的时间步和条件标签 (可以根据实际需求调整)
        t = torch.randint(0, 1000, (batch_size,), device=x.device)
        y = torch.randint(0, self.hparams.num_classes, (batch_size,), device=x.device)

        # 前向传播
        x_reconstructed, latent = self.forward(x, t, y)

        # 计算重建损失
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)

        # 计算一些统计指标
        with torch.no_grad():
            # 集合级别的MSE
            mse = F.mse_loss(x_reconstructed, x)
            # 平均相关系数
            correlation = self._compute_set_correlation(x, x_reconstructed)
            # 潜在向量的方差（衡量表示的多样性）
            latent_var = latent.var(dim=0).mean()

        # 记录指标
        self.log("train_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_mse", mse, on_step=True, on_epoch=True)
        self.log("train_correlation", correlation, on_step=True, on_epoch=True)
        self.log("train_latent_var", latent_var, on_step=True, on_epoch=True)

        return recon_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """验证步骤"""
        x = batch["expressions"]
        batch_size = x.shape[0]

        # 创建虚拟的时间步和条件标签
        t = torch.randint(0, 1000, (batch_size,), device=x.device)
        y = torch.randint(0, self.hparams.num_classes, (batch_size,), device=x.device)

        # 前向传播
        x_reconstructed, latent = self.forward(x, t, y)

        # 计算损失和指标
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)
        mse = F.mse_loss(x_reconstructed, x)
        correlation = self._compute_set_correlation(x, x_reconstructed)
        latent_var = latent.var(dim=0).mean()

        # 记录指标
        self.log("val_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True)
        self.log("val_correlation", correlation, on_step=False, on_epoch=True)
        self.log("val_latent_var", latent_var, on_step=False, on_epoch=True)

        return {
            "val_loss": recon_loss,
            "original": x.detach().cpu(),
            "reconstructed": x_reconstructed.detach().cpu(),
            "latent": latent.detach().cpu(),
        }

# ... existing code ...

    def _compute_set_correlation(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        计算集合数据的相关系数

        Args:
            x: 原始集合 [batch_size, set_size, n_genes]
            x_recon: 重建集合 [batch_size, set_size, n_genes]
        """
        batch_size, set_size, n_genes = x.shape
        correlations = []

        for b in range(batch_size):
            for s in range(set_size):
                x_flat = x[b, s].flatten()
                x_recon_flat = x_recon[b, s].flatten()

                # 计算皮尔逊相关系数
                x_centered = x_flat - x_flat.mean()
                x_recon_centered = x_recon_flat - x_recon_flat.mean()

                numerator = (x_centered * x_recon_centered).sum()
                denominator = torch.sqrt((x_centered ** 2).sum() * (x_recon_centered ** 2).sum())

                if denominator > 1e-8:
                    corr = numerator / denominator
                    correlations.append(corr)

        return torch.stack(correlations).mean() if correlations else torch.tensor(0.0, device=x.device)


def train_set_autoencoder(
    data_path: str,
    n_genes: int,
    n_cells_per_sample: int = 5,
    latent_dim: int = 256,
    max_seq_len: int = 1024,
    dim: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    multiple_of: int = 256,
    ffn_dim_multiplier: Optional[float] = None,
    norm_eps: float = 1e-5,
    class_dropout_prob: float = 0.1,
    num_classes: int = 10,
    use_pos_embedding: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    reconstruction_loss: str = "mse",
    batch_size: int = 32,
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: str = "auto",
    output_dir: str = "set_autoencoder_output",
    log_embeddings_every_n_epochs: int = 10,
    sampling_with_replacement: bool = True,
    dataset_seed: Optional[int] = None,
):
    """
    训练集合 Autoencoder 的主函数

    Args:
        data_path: 数据文件路径
        n_genes: 基因数量
        n_cells_per_sample: 每个样本的细胞数量
        latent_dim: 潜在空间维度
        max_seq_len: 最大序列长度
        dim: 模型隐藏维度
        n_layers: Transformer层数
        n_heads: 注意力头数
        multiple_of: FFN维度的倍数
        ffn_dim_multiplier: FFN维度倍数
        norm_eps: 归一化epsilon
        class_dropout_prob: 类别dropout概率
        num_classes: 类别数
        use_pos_embedding: 是否使用位置编码
        learning_rate: 学习率
        weight_decay: 权重衰减
        reconstruction_loss: 重建损失函数类型
        batch_size: 批次大小
        max_epochs: 最大训练轮数
        accelerator: 加速器类型
        devices: 设备配置
        output_dir: 输出目录
        log_embeddings_every_n_epochs: 记录嵌入的频率
        sampling_with_replacement: 数据采样是否有放回
        dataset_seed: 数据集随机种子
    """

    # 创建系统
    system = SetAutoencoderSystem(
        n_genes=n_genes,
        n_cells_per_sample=n_cells_per_sample,
        latent_dim=latent_dim,
        max_seq_len=max_seq_len,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        multiple_of=multiple_of,
        ffn_dim_multiplier=ffn_dim_multiplier,
        norm_eps=norm_eps,
        class_dropout_prob=class_dropout_prob,
        num_classes=num_classes,
        use_pos_embedding=use_pos_embedding,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        reconstruction_loss=reconstruction_loss,
        batch_size=batch_size,
        data_path=data_path,
        output_dir=output_dir,
        log_embeddings_every_n_epochs=log_embeddings_every_n_epochs,
        sampling_with_replacement=sampling_with_replacement,
        dataset_seed=dataset_seed,
    )

    # 打印模型信息
    model_size = sum(p.numel() for p in system.parameters() if p.requires_grad)
    print(f"模型参数数量: {model_size:,} ({model_size / 1e6:.2f}M)")
    print(f"数据路径: {data_path}")
    print(f"基因数量: {n_genes}")
    print(f"每样本细胞数: {n_cells_per_sample}")
    print(f"潜在维度: {latent_dim}")
    print(f"模型隐藏维度: {dim}")
    print(f"Transformer层数: {n_layers}")

    # 创建 trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        log_every_n_steps=50,
        val_check_interval=0.5,
        enable_checkpointing=True,
        default_root_dir=output_dir,
        gradient_clip_val=1.0,  # 梯度裁剪，有助于训练稳定性
    )

    # 开始训练
    trainer.fit(system)

    print(f"训练完成！结果保存在: {output_dir}")
    return system, trainer


if __name__ == "__main__":
    import fire
    fire.Fire(train_set_autoencoder)