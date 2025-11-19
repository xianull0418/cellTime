
# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
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
    def __init__(self, layer_id, dim, n_heads, multiple_of, ffn_dim_multiplier, norm_eps):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, x, freqs_cis):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TAE(nn.Module):
    """
    支持序列输入 [batch, len, dim] 的 DiT 模型
    适用于细胞表达序列等1D序列数据
    """

    def __init__(
        self,
        input_dim,
        max_seq_len=1024,
        dim=512,
        n_layers=6,
        n_heads=8,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,
        use_pos_embedding=True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.use_pos_embedding = use_pos_embedding

        # 输入线性投影层
        self.x_embedder = nn.Linear(input_dim, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        # 位置编码（可选）
        if use_pos_embedding:
            self.pos_embedder = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

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

        # 输出层
        self.final_layer = FinalLayer(dim, input_dim)

        # 旋转位置编码
        self.freqs_cis = TAE.precompute_freqs_cis(dim // n_heads, max_seq_len)

    def forward(self, x):
        """
        Args:
            x: 输入序列 [batch, seq_len, input_dim]

        Returns:
            输出序列 [batch, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 确保 freqs_cis 在正确设备上
        self.freqs_cis = self.freqs_cis.to(x.device)

        # 输入嵌入
        x = self.x_embedder(x)

        # 添加位置编码
        if self.use_pos_embedding:
            if seq_len > self.max_seq_len:
                raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_seq_len}")
            x = x + self.pos_embedder[:, :seq_len]

        # Transformer层
        for layer in self.layers:
            x = layer(x, self.freqs_cis[:seq_len])

        # 输出层
        x = self.final_layer(x)

        return x

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        """预计算旋转位置编码"""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class FinalLayer(nn.Module):
    """序列版本的最终输出层"""

    def __init__(self, hidden_size, output_dim):
        super().__init__()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)

        # 初始化
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            [batch, seq_len, output_dim]
        """
        # x = self.norm_final(x)
        x = self.linear(x)
        return x