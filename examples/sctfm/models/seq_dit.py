# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.n_rep = 1
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
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
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

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
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

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

    def forward(self, x, freqs_cis, adaln_input=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
            )
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
            )
        else:
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            x = x + self.feed_forward(self.ffn_norm(x))

        return x

class SequenceDit(nn.Module):
    """
    支持序列输入 [batch, len, dim] 的 DiT 模型
    适用于细胞表达序列等1D序列数据
    """

    def __init__(
        self,
        input_dim,                    # 输入特征维度
        max_seq_len=1024,            # 最大序列长度
        dim=512,                     # 模型隐藏维度
        n_layers=6,                  # Transformer层数
        n_heads=8,                   # 注意力头数
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,              # 类别数
        use_pos_embedding=True,      # 是否使用位置编码
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim  # 输出维度与输入相同
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.use_pos_embedding = use_pos_embedding

        # 输入线性投影层
        self.x_embedder = nn.Linear(input_dim, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        # 位置编码（可选）
        if use_pos_embedding:
            self.pos_embedder = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        # 时间步嵌入
        self.t_embedder = TimestepEmbedder(min(dim, 1024))

        # 标签嵌入
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

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
        self.final_layer = SequenceFinalLayer(dim, input_dim)

        # 旋转位置编码
        self.freqs_cis = SequenceDit.precompute_freqs_cis(dim // n_heads, max_seq_len)

    def forward(self, x, t, y):
        """
        Args:
            x: 输入序列 [batch, seq_len, input_dim]
            t: 时间步 [batch]
            y: 条件标签 [batch]

        Returns:
            输出序列 [batch, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 确保 freqs_cis 在正确设备上
        self.freqs_cis = self.freqs_cis.to(x.device)

        # 输入嵌入: [batch, seq_len, input_dim] -> [batch, seq_len, dim]
        x = self.x_embedder(x)

        # 添加位置编码
        if self.use_pos_embedding:
            if seq_len > self.max_seq_len:
                raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_seq_len}")
            x = x + self.pos_embedder[:, :seq_len]

        # 时间步和标签嵌入
        t_emb = self.t_embedder(t)  # [batch, dim]
        y_emb = self.y_embedder(y, self.training)  # [batch, dim]
        adaln_input = t_emb.to(x.dtype) + y_emb.to(x.dtype)

        # Transformer层
        for layer in self.layers:
            x = layer(x, self.freqs_cis[:seq_len], adaln_input=adaln_input)

        # 输出层: [batch, seq_len, dim] -> [batch, seq_len, input_dim]
        x = self.final_layer(x, adaln_input)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        使用 Classifier-Free Guidance 的前向传播
        """
        half = x[:len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        cond_out, uncond_out = torch.split(model_out, len(model_out) // 2, dim=0)
        half_out = uncond_out + cfg_scale * (cond_out - uncond_out)

        return torch.cat([half_out, half_out], dim=0)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        """预计算旋转位置编码"""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class SequenceFinalLayer(nn.Module):
    """
    序列版本的最终输出层
    """

    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        # 初始化
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        """
        Args:
            x: [batch, seq_len, hidden_size]
            c: [batch, hidden_size] adaln conditioning

        Returns:
            [batch, seq_len, output_dim]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)  # [batch, hidden_size]

        # 扩展维度以匹配序列
        shift = shift.unsqueeze(1)  # [batch, 1, hidden_size]
        scale = scale.unsqueeze(1)  # [batch, 1, hidden_size]

        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


def DiT_Llama_600M_patch2(**kwargs):
    return CifarDit(patch_size=2, dim=256, n_layers=16, n_heads=32, **kwargs)


def DiT_Llama_3B_patch2(**kwargs):
    return CifarDit(patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs)


if __name__ == "__main__":
    model = DiT_Llama_600M_patch2()
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 100, (2,))
    y = torch.randint(0, 10, (2,))

    with torch.no_grad():
        out = model(x, t, y)
        print(out.shape)
        out = model.forward_with_cfg(x, t, y, 0.5)
        print(out.shape)
