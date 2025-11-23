"""
UNet 骨干网络
基于 1D 卷积的 U-Net 架构，适配单细胞潜空间数据
"""

import torch
import torch.nn as nn
from typing import Optional, List
from models.backbones.base import BackboneBase
from models.utils import SinusoidalTimeEmbedding


class ResidualBlock(nn.Module):
    """
    1D 残差块，包含卷积、BN/GN、激活和 dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 如果输入输出通道数不同，需要投影
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            
        # 第一个卷积块
        self.norm1 = nn.GroupNorm(8, in_channels)  # GroupNorm 通常比 BatchNorm 更适合小批量
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 时间嵌入投影
        self.time_proj = None
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, out_channels)
            
        # 第二个卷积块
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t_emb=None):
        """
        x: [B, C, L]
        t_emb: [B, time_emb_dim]
        """
        h = x
        
        # 1st block
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Add time embedding
        if self.time_proj is not None and t_emb is not None:
            # 投影并扩展维度以匹配特征图 [B, C, 1]
            t = self.time_proj(t_emb)[:, :, None]
            h = h + t
            
        # 2nd block
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class Downsample(nn.Module):
    """下采样层 (Conv1d stride=2)"""
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    """上采样层 (Interpolate + Conv)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNetBackbone(BackboneBase):
    """
    1D U-Net 骨干网络
    
    对于单细胞潜变量数据，我们将其视为 1D 信号 [B, 1, latent_dim] 进行卷积处理。
    或者视为 [B, latent_dim, 1] 进行特征通道卷积？
    
    通常 UNet 用于图像 [B, C, H, W]。
    这里有两种策略：
    1. 将 latent_dim 视为序列长度 L，通道数 C=1 -> [B, 1, L]
       优点：利用局部相关性（如果潜变量有拓扑结构）
       缺点：VAE 的潜变量通常各维度是独立的，没有空间局部性
       
    2. 将 latent_dim 映射到高维特征，再用 MLP (UNet 需要空间维度进行上下采样)
    
    更合理的做法是：
    将 latent_dim 重塑为 [B, C, L] 结构，例如 latent_dim=128 -> [B, 1, 128]。
    但是 U-Net 的核心是多尺度特征提取。如果潜变量各维度没有顺序关系，
    用卷积可能不如 Transformer (DiT) 或 MLP 有效。
    
    不过为了满足需求，我们实现一个处理 [B, C, L] 数据的 1D U-Net。
    我们将 latent_dim 视为序列长度 (L)，并设置初始通道数为 1。
    注意：这要求 latent_dim 能被 2^downsample_times 整除。
    """
    
    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4, 8],
        dropout: float = 0.1,
        use_time_embedding: bool = True,
        time_emb_dim: int = 256,
        use_cond: bool = False,
        cond_dim: int = 0,
    ):
        super().__init__(latent_dim)
        
        self.use_time_embedding = use_time_embedding
        self.use_cond = use_cond
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        
        # 检查 latent_dim 是否适合下采样
        num_resolutions = len(channel_mults)
        assert latent_dim % (2 ** (num_resolutions - 1)) == 0, \
            f"latent_dim {latent_dim} must be divisible by {2**(num_resolutions-1)} for {num_resolutions} levels"
        
        # 时间/条件 嵌入
        emb_dim = time_emb_dim
        if use_time_embedding:
            self.time_embedder = SinusoidalTimeEmbedding(time_emb_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        
        if use_cond and cond_dim > 0:
            self.cond_embedder = nn.Sequential(
                nn.Linear(cond_dim, time_emb_dim),
                nn.SiLU(),
            )
        
        # 初始卷积
        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Down 路径
        self.down_blocks = nn.ModuleList()
        curr_channels = base_channels
        
        # 记录每层的通道数，用于跳跃连接
        self.skip_connections = [curr_channels] 
        
        for level, mult in enumerate(channel_mults[:-1]):
            out_channels = base_channels * mult
            
            # 两个残差块
            block = nn.ModuleList([
                ResidualBlock(curr_channels, out_channels, emb_dim, dropout),
                ResidualBlock(out_channels, out_channels, emb_dim, dropout),
                Downsample(out_channels)
            ])
            self.down_blocks.append(block)
            
            curr_channels = out_channels
            self.skip_connections.append(curr_channels) # Block 1 output
            self.skip_connections.append(curr_channels) # Block 2 output
            # self.skip_connections.append(curr_channels) # Downsample output (不存了)
            
        # Middle 路径
        mid_channels = base_channels * channel_mults[-1]
        self.mid_block1 = ResidualBlock(curr_channels, mid_channels, emb_dim, dropout)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, emb_dim, dropout)
        curr_channels = mid_channels
        
        # Up 路径
        self.up_blocks = nn.ModuleList()
        reversed_mults = list(reversed(channel_mults[:-1]))
        
        # 模拟 Middle path 的输入被 pop 掉 (对应 forward 中的 skips.pop() 扔掉 downsample 前的特征)
        # 但是我们在 forward 里，middle 的输入是 downsample 的输出，而 downsample 的输出我们没存进栈
        # 所以 forward 里 skips 栈顶直接就是 最后一层 Down Block 的 Res2 输出
        
        # 让我们重新理清 forward 的逻辑：
        # 1. Down Loop:
        #    - Res1 -> append
        #    - Res2 -> append
        #    - Down -> (不 append) -> 传给下一层或 Middle
        
        # 2. Middle:
        #    - 接收 Down 的输出
        
        # 3. Up Loop:
        #    - 栈顶是 Res2 输出。
        #    - 我们需要 pop Res2 做 concat
        #    - 还需要 pop Res1 扔掉
        
        # 所以 __init__ 这里的逻辑应该是：
        
        for level, mult in enumerate(reversed_mults):
            out_channels = base_channels * mult
            
            # 需要加上 skip connection 的通道数
            skip_ch = self.skip_connections.pop() # Res2
            _ = self.skip_connections.pop()       # Res1 (扔掉)
            
            block = nn.ModuleList([
                # 上采样 + 卷积
                Upsample(curr_channels), 
                # 接一个残差块处理融合后的特征
                nn.Conv1d(curr_channels + skip_ch, out_channels, kernel_size=3, padding=1),
                ResidualBlock(out_channels, out_channels, emb_dim, dropout),
            ])
            self.up_blocks.append(block)
            curr_channels = out_channels
            
        # 输出卷积
        self.norm_out = nn.GroupNorm(8, curr_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv1d(curr_channels, in_channels, kernel_size=3, padding=1)
        
        # 初始化
        nn.init.zeros_(self.conv_out.weight)
        
    def forward(self, x_t, t, cond=None):
        """
        x_t: [B, latent_dim]
        t: [B]
        cond: [B, cond_dim]
        """
        # 1. 预处理
        # 将 [B, latent_dim] 重塑为 [B, 1, latent_dim] 以进行 1D 卷积
        x = x_t.unsqueeze(1) 
        
        # 2. 嵌入
        emb = None
        if self.use_time_embedding:
            t_emb = self.time_embedder(t)
            emb = self.time_mlp(t_emb)
            
        if self.use_cond and cond is not None:
            c_emb = self.cond_embedder(cond)
            emb = emb + c_emb if emb is not None else c_emb
            
        # 3. Down Path
        h = self.conv_in(x)
        skips = [h]
        
        for block in self.down_blocks:
            res_block1, res_block2, downsample = block
            
            h = res_block1(h, emb)
            skips.append(h)
            
            h = res_block2(h, emb)
            skips.append(h)
            
            h = downsample(h)
            # skips.append(h)  <-- 不存储 downsample 输出
            
        # 4. Middle Path
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)
        
        # 5. Up Path
        # skips 栈现在是: [In, B1_out, B2_out, B3_out, B4_out, ...]
        # 栈顶是最后一层的 Res2_out，正好是 Up path 第一层需要的 skip
        
        # _ = skips.pop()  <-- 不需要 pop，因为 downsample 输出没存
        
        for block in self.up_blocks:
            upsample, conv_fuse, res_block = block
            
            # 上采样
            h = upsample(h)
            
            # 获取对应的 Skip connection
            # skips 存入顺序: Res1_out, Res2_out
            # 我们需要 Res2_out
            
            skip = skips.pop() 
            
            # 还需要 pop 掉 Res1_out，因为它没被用到
            _ = skips.pop()
            
            # 如果维度因为 padding 问题不匹配，需要 crop 或 pad
            if h.shape[-1] != skip.shape[-1]:
                h = torch.nn.functional.interpolate(h, size=skip.shape[-1], mode="nearest")
                
            h = torch.cat([h, skip], dim=1)
            h = conv_fuse(h)
            h = res_block(h, emb)
            
        # 6. Output
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        
        # 移除通道维度 [B, 1, L] -> [B, L]
        return h.squeeze(1)

