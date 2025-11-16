"""
This file contains the neural network architectures.
These are all you need for inference.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Union
import math


def modulate(x, shift, scale):
    """AdaLN modulation function"""
    return x * (1 + scale) + shift


class ResidualMLP(nn.Module):
    """Generic MLP with residual connections and optional modulation"""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: Optional[int] = None,
        activation: str = "prelu",
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        final_activation: Optional[str] = None,
        embed_dim: int = 0  # For AdaLN modulation
    ):
        super().__init__()

        if output_dim is None:
            output_dim = hidden_dims[-1]

        self.use_residual = use_residual
        self.use_modulation = embed_dim > 0

        # Activation functions
        self.act_dict = {
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "prelu": nn.PReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU()
        }

        self.layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList() if use_residual else None
        self.modulation_layers = nn.ModuleList() if self.use_modulation else None

        dims = [input_dim] + list(hidden_dims) + ([output_dim] if output_dim != hidden_dims[-1] else [])

        for i in range(len(dims) - 1):
            # Build main layer components
            layer_components = []

            # Add dropout
            if i == 0 and input_dropout > 0:
                layer_components.append(nn.Dropout(input_dropout))
            elif i > 0 and dropout > 0:
                layer_components.append(nn.Dropout(dropout))

            # Linear layer
            layer_components.append(nn.Linear(dims[i], dims[i + 1]))

            # Layer normalization
            if use_layer_norm and i < len(dims) - 2:
                layer_components.append(nn.LayerNorm(dims[i + 1]))
            elif use_layer_norm and i == len(dims) - 2 and final_activation is None:
                # Don't add norm before final layer if no final activation
                pass
            elif use_layer_norm:
                layer_components.append(nn.LayerNorm(dims[i + 1]))

            # Activation
            if i < len(dims) - 2:
                layer_components.append(self.act_dict[activation])
            elif final_activation:
                layer_components.append(self.act_dict[final_activation])

            self.layers.append(nn.Sequential(*layer_components))

            # Residual projection
            if use_residual:
                if dims[i] != dims[i + 1]:
                    self.residual_projs.append(nn.Linear(dims[i], dims[i + 1]))
                else:
                    self.residual_projs.append(nn.Identity())

            # AdaLN modulation
            if self.use_modulation and i < len(dims) - 1:
                self.modulation_layers.append(
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(embed_dim, 2 * dims[i + 1], bias=True),
                    )
                )

    def forward(self, x, cond=None):
        current_x = x

        for i, layer in enumerate(self.layers):
            # Store input for residual
            layer_input = current_x

            # Forward through main layer
            current_x = layer(current_x)

            # Apply modulation if available
            if self.use_modulation and cond is not None and i < len(self.modulation_layers):
                shift, scale = self.modulation_layers[i](cond).chunk(2, dim=1)
                current_x = modulate(current_x, shift, scale)

            # Apply residual connection
            if self.use_residual and i < len(self.layers):
                residual = self.residual_projs[i](layer_input)
                current_x = current_x + residual

        return current_x

class TimestepEmbedder(nn.Module):
    """Timestep embedding for diffusion/flow models"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = ResidualMLP(
            input_dim=frequency_embedding_size,
            hidden_dims=[hidden_size],
            output_dim=hidden_size,
            activation="silu",
            use_residual=False  # Simple MLP for embedding
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Sinusoidal timestep embeddings"""
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
        return self.mlp(t_freq)


class ConditionEmbedder(nn.Module):
    """Unified condition embedder for different types of conditions"""
    def __init__(self, cond_dim: int, hidden_size: int, cond_type: str = "continuous"):
        super().__init__()
        self.cond_type = cond_type

        if cond_type == "continuous":
            self.mlp = ResidualMLP(
                input_dim=cond_dim,
                hidden_dims=[hidden_size],
                output_dim=hidden_size,
                activation="silu",
                use_residual=False  # Simple MLP for embedding
            )
        elif cond_type == "categorical":
            self.embedding = nn.Embedding(cond_dim, hidden_size)
        else:
            raise ValueError(f"Unsupported condition type: {cond_type}")

    def forward(self, cond):
        if self.cond_type == "continuous":
            return self.mlp(cond)
        else:
            return self.embedding(cond)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.linear = ResidualMLP(
            input_dim=hidden_size,
            hidden_dims=[hidden_size],
            output_dim=output_dim,
            # activation="tanh",
            activation="prelu",
            final_activation=None,
            use_residual=False  # Final layer usually doesn't need residual
        )

    def forward(self, x):
        return self.linear(x)


class UnifiedCondEncoder(nn.Module):
    """
    Unified Rectified Flow Encoder for single-cell data

    Handles both regular conditioning and temporal conditioning in one class.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [128, 128],
        dropout: float = 0.5,
        input_dropout: float = 0.,
        cond_dim: Optional[int] = None,
        cond_type: str = "continuous",
        use_time_embedding: bool = True,
        use_time_cond_embedding: bool = False,
        use_residual: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_time_embedding = use_time_embedding
        self.use_conditioning = cond_dim is not None
        self.use_time_cond_embedding = use_time_cond_embedding
        self.use_residual = use_residual

        # Calculate embedding dimensions
        embed_dims = []
        if use_time_embedding:
            self.time_embedder = TimestepEmbedder(latent_dim, latent_dim)
            embed_dims.append(latent_dim)

        if self.use_conditioning:
            embed_hidden = min(hidden_dim[0], 512)
            self.cond_embedder = ConditionEmbedder(cond_dim, embed_hidden, cond_type)
            embed_dims.append(embed_hidden)

        if use_time_cond_embedding:
            self.time_cond_embedder = TimestepEmbedder(latent_dim, latent_dim)
            embed_dims.append(latent_dim)

        total_embed_dim = sum(embed_dims)

        # Build unified network using ResidualMLP
        self.network = ResidualMLP(
            input_dim=n_genes,
            hidden_dims=hidden_dim,
            output_dim=None,
            activation="prelu",
            dropout=dropout,
            input_dropout=input_dropout,
            use_layer_norm=True,
            use_residual=use_residual,
            final_activation="tanh",
            embed_dim=total_embed_dim if total_embed_dim > 0 else 0
        )

        self.output_layer = FinalLayer(hidden_dim[-1], n_genes)

    def _prepare_embeddings(self, t=None, cond=None, cond_t=None):
        """Prepare and combine all embeddings"""
        embeddings = []

        if self.use_time_embedding and t is not None:
            if t.dim() == 2 and t.shape[1] == 1:
                t = t.squeeze(-1)
            embeddings.append(self.time_embedder(t))

        if self.use_conditioning and cond is not None:
            embeddings.append(self.cond_embedder(cond))

        if self.use_time_cond_embedding and cond_t is not None:
            if cond_t.dim() == 2 and cond_t.shape[1] == 1:
                cond_t = cond_t.squeeze(-1)
            embeddings.append(self.time_cond_embedder(cond_t))

        if embeddings:
            return torch.cat(embeddings, dim=-1)
        return None

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None,
                cond_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with unified embedding handling"""

        # Prepare combined embeddings
        combined_emb = self._prepare_embeddings(t, cond, cond_t)

        # Forward through network blocks
        x = self.network(x, combined_emb)

        return self.output_layer(x)


# Legacy aliases for backward compatibility
CondEncoder = UnifiedCondEncoder
CondTempEncoder = lambda *args, use_time_cond_embedding=True, **kwargs: \
    UnifiedCondEncoder(*args, use_time_cond_embedding=use_time_cond_embedding, **kwargs)


if __name__ == '__main__':
    # Test regular conditioning
    encoder = UnifiedCondEncoder(
        n_genes=2000,
        latent_dim=128,
        hidden_dim=[128, 128],
        cond_dim=10,
        cond_type="continuous",
        use_time_embedding=True
    )

    x = torch.randn(8, 2000)
    t = torch.rand(8)
    cond = torch.randn(8, 10)
    
    output = encoder(x, t, cond)
    print(f"Regular conditioning output shape: {output.shape}")

    # Test temporal conditioning
    temp_encoder = UnifiedCondEncoder(
        n_genes=2000,
        latent_dim=128,
        hidden_dim=[128, 128],
        cond_dim=10,
        use_time_embedding=True,
        use_time_cond_embedding=True
    )
    
    cond_t = torch.rand(8)
    output = temp_encoder(x, t, cond, cond_t)
    print(f"Temporal conditioning output shape: {output.shape}")

    # Test without conditioning
    simple_encoder = UnifiedCondEncoder(
        n_genes=2000,
        latent_dim=128,
        hidden_dim=[128, 128],
        use_time_embedding=True
    )
    
    output = simple_encoder(x, t)
    print(f"Simple output shape: {output.shape}")