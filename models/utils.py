"""
工具函数模块
包含时间编码、辅助函数等
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间编码
    将标量时间 t ∈ [0, 1] 编码为高维向量
    """
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步 [B] 或 [B, 1]，范围 [0, 1]
        
        Returns:
            编码后的时间 [B, dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        
        # 生成频率
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )  # [half_dim]
        
        # 计算编码
        args = t * freqs.unsqueeze(0)  # [B, half_dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        
        # 如果 dim 是奇数，补零
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding


class TimestepEmbedder(nn.Module):
    """
    时间步嵌入器（用于 DiT）
    将标量时间 t 编码为高维向量，然后通过 MLP
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        创建正弦位置编码
        
        Args:
            t: 时间步 [B]
            dim: 编码维度
            max_period: 最大周期
        
        Returns:
            编码 [B, dim]
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步 [B]，范围可以是任意实数
        
        Returns:
            嵌入 [B, hidden_size]
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


def compute_correlation(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """
    计算两个张量的皮尔逊相关系数（按样本）
    
    Args:
        x: 原始数据 [B, ...]
        x_recon: 重建数据 [B, ...]
    
    Returns:
        平均相关系数（标量）
    """
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
        
        if denominator > 1e-8:
            corr = numerator / denominator
            correlations.append(corr)
    
    if correlations:
        return torch.stack(correlations).mean()
    else:
        return torch.tensor(0.0, device=x.device)


def create_backbone(backbone_type: str, backbone_config: dict, latent_dim: int):
    """
    工厂函数：根据类型创建骨干网络
    
    Args:
        backbone_type: 骨干网络类型 ('dit' 或 'mlp')
        backbone_config: 骨干网络配置字典
        latent_dim: 潜空间维度
    
    Returns:
        骨干网络实例
    """
    if backbone_type == 'dit':
        from models.backbones.dit import DiTBackbone
        return DiTBackbone(latent_dim=latent_dim, **backbone_config)
    elif backbone_type == 'mlp':
        from models.backbones.mlp import MLPBackbone
        return MLPBackbone(latent_dim=latent_dim, **backbone_config)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


