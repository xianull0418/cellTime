"""
骨干网络抽象基类
定义速度场预测器的统一接口
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class BackboneBase(nn.Module, ABC):
    """
    骨干网络抽象基类
    所有速度场预测器都应该继承这个类
    """
    
    def __init__(self, latent_dim: int):
        """
        Args:
            latent_dim: 潜空间维度
        """
        super().__init__()
        self.latent_dim = latent_dim
    
    @abstractmethod
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
            cond: 可选条件信息 [B, cond_dim]，如时间戳、细胞类型等
        
        Returns:
            预测的速度场 [B, latent_dim]
        """
        raise NotImplementedError
    
    def num_parameters(self) -> int:
        """返回模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

