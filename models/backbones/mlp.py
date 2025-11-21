"""
MLP 骨干网络
基于多层感知器的速度场预测器
"""

import torch
import torch.nn as nn
from typing import List, Optional
from models.backbones.base import BackboneBase
from models.utils import SinusoidalTimeEmbedding


class MLPBackbone(BackboneBase):
    """
    MLP 骨干网络，用于预测 Rectified Flow 的速度场
    支持时间编码和可选的条件编码
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: List[int] = [512, 512],
        dropout: float = 0.1,
        input_dropout: float = 0.1,
        use_time_embedding: bool = True,
        time_emb_dim: int = 128,
        use_cond: bool = False,
        cond_dim: Optional[int] = None,
        cond_type: str = "continuous",
    ):
        """
        Args:
            latent_dim: 潜空间维度
            hidden_dim: 隐藏层维度列表
            dropout: 隐藏层 dropout
            input_dropout: 输入层 dropout
            use_time_embedding: 是否使用时间嵌入
            time_emb_dim: 时间嵌入维度
            use_cond: 是否使用条件信息
            cond_dim: 条件维度
            cond_type: 条件类型 ('continuous' 或 'categorical')
        """
        super().__init__(latent_dim)
        
        self.use_time_embedding = use_time_embedding
        self.use_cond = use_cond
        self.cond_type = cond_type
        
        # 时间编码器
        if use_time_embedding:
            self.time_embedder = SinusoidalTimeEmbedding(time_emb_dim)
            self.time_proj = nn.Linear(time_emb_dim, latent_dim)
        
        # 条件编码器
        if use_cond and cond_dim is not None:
            if cond_type == "continuous":
                self.cond_embedder = nn.Sequential(
                    nn.Linear(cond_dim, hidden_dim[0]),
                    nn.SiLU(),
                    nn.Linear(hidden_dim[0], latent_dim)
                )
            elif cond_type == "categorical":
                self.cond_embedder = nn.Embedding(cond_dim, latent_dim)
            else:
                raise ValueError(f"Unsupported condition type: {cond_type}")
        
        # 计算输入维度（x_t 本身的维度）
        input_dim = latent_dim
        
        # 构建主网络
        layers = []
        
        # 输入 dropout
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        
        # 隐藏层
        dims = [input_dim] + hidden_dim
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(dims[-1], latent_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化输出层为接近零
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测速度场
        
        Args:
            x_t: 潜空间中的噪声状态 [B, latent_dim]
            t: 时间步 [B]，范围 [0, 1]
            cond: 可选条件信息 [B, cond_dim]
        
        Returns:
            预测的速度场 [B, latent_dim]
        """
        # 时间编码
        if self.use_time_embedding:
            t_emb = self.time_embedder(t)  # [B, time_emb_dim]
            t_emb = self.time_proj(t_emb)  # [B, latent_dim]
            x_t = x_t + t_emb  # 加入时间信息
        
        # 条件编码
        if self.use_cond and cond is not None:
            if self.cond_type == "continuous":
                cond_emb = self.cond_embedder(cond)  # [B, latent_dim]
            else:  # categorical
                cond_emb = self.cond_embedder(cond)  # [B, latent_dim]
            x_t = x_t + cond_emb  # 加入条件信息
        
        # 主网络预测速度场
        v = self.network(x_t)
        
        return v


