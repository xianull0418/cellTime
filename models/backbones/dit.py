"""
DiT 骨干网络
适配单细胞潜空间的 Diffusion Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from models.backbones.base import BackboneBase
from models.utils import TimestepEmbedder


def modulate(x, shift, scale):
    """AdaLN 调制函数"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """多头自注意力层（带 RoPE 位置编码）"""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        self.q_norm = nn.LayerNorm(n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(n_heads * self.head_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, dim]
        Returns:
            [B, seq_len, dim]
        """
        bsz, seqlen, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        # 使用 scaled_dot_product_attention
        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),  # [B, n_heads, seq_len, head_dim]
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)  # [B, seq_len, n_heads, head_dim]
        
        output = output.flatten(-2)  # [B, seq_len, dim]
        return self.wo(output)


class FeedForward(nn.Module):
    """前馈网络（SwiGLU 激活）"""
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Transformer 块（带 AdaLN 调制）"""
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
    ):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)
        
        # AdaLN 调制（用于时间和条件信息）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )
    
    def forward(self, x, adaln_input=None):
        """
        Args:
            x: [B, seq_len, dim]
            adaln_input: [B, adaln_dim] AdaLN 条件输入
        """
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )
            
            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa)
            )
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
            )
        else:
            x = x + self.attention(self.attention_norm(x))
            x = x + self.feed_forward(self.ffn_norm(x))
        
        return x


class FinalLayer(nn.Module):
    """最终输出层（带 AdaLN）"""
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        
        # 初始化输出层为接近零
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x, c):
        """
        Args:
            x: [B, seq_len, dim]
            c: [B, adaln_dim]
        Returns:
            [B, seq_len, output_dim]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTBackbone(BackboneBase):
    """
    DiT 骨干网络，适配单细胞潜空间
    将潜空间向量视为序列长度为 1 的序列
    """
    
    def __init__(
        self,
        latent_dim: int,
        dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        use_time_embedding: bool = True,
        time_emb_dim: int = 256,
        use_class_cond: bool = False,
        num_classes: int = 10,
        class_dropout_prob: float = 0.1,
        use_vector_cond: bool = False,  # 新增：支持连续向量条件
        vector_cond_dim: int = 0,       # 新增：向量条件维度
    ):
        """
        Args:
            latent_dim: 潜空间维度
            dim: Transformer 隐藏维度
            n_layers: Transformer 层数
            n_heads: 注意力头数
            multiple_of: FFN 维度的倍数
            ffn_dim_multiplier: FFN 维度倍数
            norm_eps: 归一化 epsilon
            use_time_embedding: 是否使用时间嵌入
            time_emb_dim: 时间嵌入维度
            use_class_cond: 是否使用类别条件
            num_classes: 类别数
            class_dropout_prob: 类别 dropout 概率（CFG）
            use_vector_cond: 是否使用连续向量条件
            vector_cond_dim: 向量条件维度
        """
        super().__init__(latent_dim)
        
        self.dim = dim
        self.use_time_embedding = use_time_embedding
        self.use_class_cond = use_class_cond
        self.use_vector_cond = use_vector_cond
        
        # 输入投影：潜空间 -> Transformer 隐藏空间
        self.x_embedder = nn.Linear(latent_dim, dim, bias=True)
        
        # 时间编码器
        if use_time_embedding:
            self.t_embedder = TimestepEmbedder(min(dim, 1024), time_emb_dim)
        
        # 类别编码器（离散）
        if use_class_cond:
            use_cfg_embedding = int(class_dropout_prob > 0)
            self.y_embedder = nn.Embedding(
                num_classes + use_cfg_embedding, min(dim, 1024)
            )
            self.class_dropout_prob = class_dropout_prob
            self.num_classes = num_classes
            
        # 向量编码器（连续）
        if use_vector_cond:
            self.vector_embedder = nn.Sequential(
                nn.Linear(vector_cond_dim, min(dim, 1024)),
                nn.SiLU(),
                nn.Linear(min(dim, 1024), min(dim, 1024)),
            )
        
        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(
                layer_id,
                dim,
                n_heads,
                multiple_of,
                ffn_dim_multiplier,
                norm_eps,
            )
            for layer_id in range(n_layers)
        ])
        
        # 输出层：Transformer 隐藏空间 -> 潜空间速度场
        self.final_layer = FinalLayer(dim, latent_dim)
    
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
            cond: 可选条件信息 [B] (类别标签，整数)
        
        Returns:
            预测的速度场 [B, latent_dim]
        """
        # 投影到 Transformer 空间，添加序列维度
        x = self.x_embedder(x_t)  # [B, dim]
        x = x.unsqueeze(1)  # [B, 1, dim] - 序列长度为 1
        
        # 准备 AdaLN 条件输入
        adaln_input = None
        if self.use_time_embedding:
            t_emb = self.t_embedder(t)  # [B, min(dim, 1024)]
            adaln_input = t_emb
        
        # 可选：添加类别条件（离散）
        if self.use_class_cond and cond is not None and cond.dtype in (torch.long, torch.int):
            # Classifier-Free Guidance：训练时随机 dropout
            if self.training and self.class_dropout_prob > 0:
                drop_ids = torch.rand(cond.shape[0], device=cond.device) < self.class_dropout_prob
                cond = torch.where(drop_ids, self.num_classes, cond)
            
            y_emb = self.y_embedder(cond)  # [B, min(dim, 1024)]
            adaln_input = adaln_input + y_emb if adaln_input is not None else y_emb
            
        # 可选：添加向量条件（连续）
        if self.use_vector_cond and cond is not None and cond.dtype in (torch.float, torch.float16, torch.float32, torch.float64):
            v_emb = self.vector_embedder(cond)  # [B, min(dim, 1024)]
            adaln_input = adaln_input + v_emb if adaln_input is not None else v_emb
        
        # 通过 Transformer 层
        for layer in self.layers:
            x = layer(x, adaln_input=adaln_input)
        
        # 输出层
        x = self.final_layer(x, adaln_input)  # [B, 1, latent_dim]
        
        # 移除序列维度
        v = x.squeeze(1)  # [B, latent_dim]
        
        return v

