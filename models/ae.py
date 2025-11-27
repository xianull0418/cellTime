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
        """
        Args:
            n_genes: 基因数量
            latent_dim: 潜空间维度
            hidden_dim: 隐藏层维度列表
            dropout_rate: Dropout 率
        """
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
        """编码到潜空间"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜空间解码"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入基因表达数据 [B, n_genes]
        
        Returns:
            reconstructed: 重建的基因表达数据 [B, n_genes]
            latent: 潜在表示 [B, latent_dim]
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class AESystem(pl.LightningModule):
    """
    Autoencoder 训练系统（PyTorch Lightning）
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: 配置对象（OmegaConf 或字典）
        """
        super().__init__()
        
        # 如果 cfg 是 OmegaConf 对象，转换为字典；否则直接使用
        if isinstance(cfg, DictConfig):
            self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
            self.cfg = cfg
        else:
            # 从 checkpoint 加载时，cfg 已经是字典
            self.save_hyperparameters(cfg)
            self.cfg = OmegaConf.create(cfg)
        
        # 统一使用 self.cfg 访问配置
        cfg = self.cfg
        
        # 创建模型
        self.autoencoder = Autoencoder(
            n_genes=cfg.model.n_genes,
            latent_dim=cfg.model.latent_dim,
            hidden_dim=cfg.model.hidden_dim,
            dropout_rate=cfg.model.dropout_rate,
        )
        
        # 损失函数
        loss_type = cfg.training.reconstruction_loss
        if loss_type == "mse":
            self.reconstruction_loss_fn = nn.MSELoss()
        elif loss_type == "poisson":
            self.reconstruction_loss_fn = nn.PoissonNLLLoss(log_input=False)
        elif loss_type == "smooth_l1":
            self.reconstruction_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported reconstruction loss: {loss_type}")
        
        # 加载预训练权重（如果配置了）
        if hasattr(cfg.model, 'pretrained_encoder') and cfg.model.pretrained_encoder:
            self.load_pretrained_weights(
                self.autoencoder.encoder, 
                cfg.model.pretrained_encoder, 
                "encoder"
            )
            
        if hasattr(cfg.model, 'pretrained_decoder') and cfg.model.pretrained_decoder:
            self.load_pretrained_weights(
                self.autoencoder.decoder, 
                cfg.model.pretrained_decoder, 
                "decoder"
            )

        # 输出目录
        self.output_dir = Path(cfg.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集
        self.train_dataset = None
        self.val_dataset = None

    def load_pretrained_weights(self, module: nn.Module, checkpoint_path: str, module_type: str):
        """
        加载预训练权重，自动处理维度不匹配的层（如输入/输出层）
        参考 scDiffusion 的实现策略
        """
        print(f"正在加载预训练 {module_type} 权重: {checkpoint_path}")
        try:
            # scimilarity 的权重通常保存在 state_dict 中
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            
            model_dict = module.state_dict()
            pretrained_dict = {}
            skipped_keys = []
            
            for k, v in state_dict.items():
                # 处理键名可能存在的前缀差异
                if k not in model_dict and f"network.{k}" in model_dict:
                    # 有些权重可能没有 network 前缀
                    target_k = f"network.{k}"
                else:
                    target_k = k
                
                if target_k in model_dict:
                    if model_dict[target_k].shape == v.shape:
                        pretrained_dict[target_k] = v
                    else:
                        skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[target_k].shape})")
                else:
                    # 尝试模糊匹配，scimilarity 的 key 可能是 network.0.1.weight
                    pass

            # 特殊处理 scimilarity 结构
            # 如果是微调，通常第一层(encoder input)和最后一层(decoder output)会因为 n_genes 不同而不匹配
            # 我们显式地过滤掉这些层，如果它们还没有被 shape mismatch 过滤掉
            if hasattr(self.cfg.model, 'reset_input_output_layers') and self.cfg.model.reset_input_output_layers:
                if module_type == "encoder":
                    # Encoder 第一层: network.0.1.weight/bias (linear), network.0.2 (batchnorm)
                    # 注意：scimilarity Encoder layer 0 结构: Dropout -> Linear -> BatchNorm -> PReLU
                    # Linear 是 network.0.1
                    keys_to_remove = [
                        'network.0.1.weight', 'network.0.1.bias',
                        'network.0.2.weight', 'network.0.2.bias',
                        'network.0.2.running_mean', 'network.0.2.running_var',
                        'network.0.2.num_batches_tracked'
                    ]
                    for k in keys_to_remove:
                        if k in pretrained_dict:
                            del pretrained_dict[k]
                            skipped_keys.append(f"{k} (forced reset)")
                            
                elif module_type == "decoder":
                    # Decoder 最后一层: network.{last_idx}.weight/bias
                    # 假设最后一层是 Linear(hidden -> n_genes)
                    # 查找最后一层的索引
                    last_layer_idx = len(self.cfg.model.hidden_dim)
                    keys_to_remove = [
                        f'network.{last_layer_idx}.weight', 
                        f'network.{last_layer_idx}.bias'
                    ]
                    for k in keys_to_remove:
                        if k in pretrained_dict:
                            del pretrained_dict[k]
                            skipped_keys.append(f"{k} (forced reset)")

            # 更新权重
            module.load_state_dict(pretrained_dict, strict=False)
            
            print(f"成功加载 {len(pretrained_dict)} 个参数张量")
            if skipped_keys:
                print(f"跳过了 {len(skipped_keys)} 个不匹配/重置的层:")
                for sk in skipped_keys[:5]: # 只打印前5个
                    print(f"  - {sk}")
                if len(skipped_keys) > 5:
                    print(f"  ... 等 {len(skipped_keys)-5} 个")
                    
            # 冻结中间层（如果配置）
            if hasattr(self.cfg.model, 'freeze_encoder_layers') and self.cfg.model.freeze_encoder_layers and module_type == "encoder":
                print("冻结 Encoder 部分层...")
                # 简单的策略：冻结除了第一层以外的所有层
                for name, param in module.named_parameters():
                    if name in pretrained_dict: # 如果是加载进来的权重，就冻结
                        param.requires_grad = False
                print("Encoder 中间层已冻结")

        except Exception as e:
            print(f"加载权重失败: {e}")
            import traceback
            traceback.print_exc()

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.autoencoder(x)
    
    def configure_optimizers(self):
        """配置优化器和调度器"""
        # 过滤掉 requires_grad=False 的参数
        params = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        
        # 学习率调度器
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
        """设置数据集"""
        if stage == "fit" or stage is None:
            from dataset import StaticCellDataset
            
            # 尝试加载目标基因列表（如果配置了）
            target_genes = None
            if hasattr(self.cfg.data, 'target_genes_path') and self.cfg.data.target_genes_path:
                path = Path(self.cfg.data.target_genes_path)
                if path.exists():
                    print(f"加载目标基因列表: {path}")
                    with open(path, 'r') as f:
                        target_genes = [line.strip() for line in f if line.strip()]
                    print(f"目标基因数量: {len(target_genes)}")
            
            # 创建完整数据集
            # 如果配置了 n_genes，将其作为 max_genes 传递给数据集
            max_genes = self.cfg.model.n_genes if self.cfg.model.n_genes > 0 else None
            
            # 获取词汇表路径
            vocab_path = None
            if hasattr(self.cfg.data, 'vocab_path') and self.cfg.data.vocab_path:
                vocab_path = self.cfg.data.vocab_path

            # 对大规模数据，不应在此处创建 full_dataset 然后进行 random_split
            # 而是应该利用 dataset 的能力进行 split
            # 由于 StaticCellDataset 目前不支持内置 split，我们仍然使用 random_split
            # 但要注意 scbank 的实现是懒加载的，所以初始化 full_dataset 应该很快
            
            full_dataset = StaticCellDataset(
                self.cfg.data.data_path,
                vocab_path=vocab_path,  # 传递词汇表路径
                max_genes=max_genes,
                target_genes=target_genes,
                verbose=True,
                seed=42,
                limit_cells=self.cfg.data.get("limit_cells", None),
            )
            
            # 更新 n_genes（如果需要）
            if self.cfg.model.n_genes != full_dataset.n_genes:
                if target_genes is not None:
                    print(f"数据集已映射到目标基因列表，新基因数: {full_dataset.n_genes}")
                elif max_genes is not None:
                    print(f"数据集基因数已通过高变基因选择调整为: {full_dataset.n_genes}")
                else:
                    print(f"Warning: cfg.model.n_genes ({self.cfg.model.n_genes}) "
                          f"!= dataset.n_genes ({full_dataset.n_genes})")
                print(f"Using dataset.n_genes = {full_dataset.n_genes}")
                self.cfg.model.n_genes = full_dataset.n_genes
            
            # 划分训练集和验证集
            n_total = len(full_dataset)
            n_train = int(self.cfg.data.train_split * n_total)
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
        from dataset import collate_fn_static
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            collate_fn=collate_fn_static,
        )
    
    def val_dataloader(self):
        """验证数据加载器"""
        from dataset import collate_fn_static
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            collate_fn=collate_fn_static,
        )
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        x = batch  # [B, n_genes]
        
        # 前向传播
        x_reconstructed, latent = self.forward(x)
        
        # 计算重建损失
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)
        
        # 计算统计指标
        with torch.no_grad():
            mse = F.mse_loss(x_reconstructed, x)
            correlation = compute_correlation(x, x_reconstructed)
            latent_var = latent.var(dim=0).mean()
        
        # 记录指标
        self.log("train_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_mse", mse, on_step=False, on_epoch=True)
        self.log("train_correlation", correlation, on_step=False, on_epoch=True)
        self.log("train_latent_var", latent_var, on_step=False, on_epoch=True)
        
        return recon_loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """验证步骤"""
        x = batch
        
        # 前向传播
        x_reconstructed, latent = self.forward(x)
        
        # 计算损失和指标
        recon_loss = self.reconstruction_loss_fn(x_reconstructed, x)
        mse = F.mse_loss(x_reconstructed, x)
        correlation = compute_correlation(x, x_reconstructed)
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

