import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """3D convolutional block mirroring the TensorFlow implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        downsample: bool = False,
        upsample: bool = False,
        flatten: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.downsample = nn.MaxPool3d((1, 2, 2)) if downsample else None
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest") if upsample else None
        self.flatten = flatten
        self.neg_slope = 0.3
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = F.leaky_relu(x, negative_slope=self.neg_slope)
        if self.flatten:
            b, d, h, w, c = x.shape
            return x.reshape(b, d, h * w * c)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class PosEmbedding(nn.Module):
    """Adds learnable positional embeddings to token sequences."""

    def __init__(self, num_tokens: int, token_size: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(num_tokens, token_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.num_tokens, device=tokens.device)
        pos_embed = self.embedding(positions).unsqueeze(0)
        return tokens + pos_embed


class TransformerEncoder(nn.Module):
    """Pre-norm transformer encoder block with optional causal masking."""

    def __init__(
        self,
        num_heads: int,
        num_tokens: int,
        projection_dim: int,
        causal: bool = True,
        dropout_rate: float = 0.2,
        num_forward: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(projection_dim)
        self.embed_dim = projection_dim
        self.num_heads = num_heads
        self.key_dim = projection_dim
        self.inner_dim = self.num_heads * self.key_dim
        self.q_proj = nn.Linear(self.embed_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.inner_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(self.inner_dim, self.embed_dim, bias=False)
        self.norm2 = nn.LayerNorm(projection_dim)
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        if causal:
            mask = torch.tril(
                torch.ones(num_tokens, num_tokens, dtype=torch.bool), diagonal=num_forward
            )
        else:
            mask = torch.ones(num_tokens, num_tokens, dtype=torch.bool)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm1(tokens)
        q = self.q_proj(x).view(x.shape[0], x.shape[1], self.num_heads, self.key_dim)
        k = self.k_proj(x).view(x.shape[0], x.shape[1], self.num_heads, self.key_dim)
        v = self.v_proj(x).view(x.shape[0], x.shape[1], self.num_heads, self.key_dim)

        q = q.permute(0, 2, 1, 3)  # (B, heads, tokens, key_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.key_dim)
        mask = self.mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], self.inner_dim)
        attn_out = self.out_proj(attn_out)
        tokens = tokens + attn_out
        y = self.norm2(tokens)
        y = self.mlp(y)
        return tokens + y
