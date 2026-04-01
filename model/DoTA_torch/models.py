"""
DoTA (torch) adapted from DoTA (tensorflow): 
https://github.com/opaserr/dota/tree/dev_photons
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn

try:
    from .blocks import ConvBlock, PosEmbedding, TransformerEncoder
except ImportError:  
    from blocks import ConvBlock, PosEmbedding, TransformerEncoder


@dataclass
class DotaResidualConfig:
    inshape: Iterable[int]
    steps: int
    enc_feats: int
    num_heads: int
    num_transformers: int
    kernel_size: int
    dropout_rate: float = 0.2
    causal: bool = True


class DotaResidual(nn.Module):
    def __init__(self, cfg: DotaResidualConfig) -> None:
        super().__init__()
        inshape = tuple(cfg.inshape)
        if len(inshape) != 4:
            raise ValueError(f"inshape must be length 4 (tokens, height, width, channels), got {inshape}")

        self.steps = cfg.steps
        self.kernel_size = cfg.kernel_size
        self.enc_feats = cfg.enc_feats
        self.num_tokens = inshape[0]
        slice_dim = inshape[1:]
        if any(dim % (2 ** cfg.steps) != 0 for dim in slice_dim[:-1]):
            raise ValueError("Height and width must be divisible by 2**steps")
        token_spatial = tuple(int(dim / (2 ** cfg.steps)) for dim in slice_dim[:-1])
        self.token_spatial: Tuple[int, int] = token_spatial  # (height, width) at token scale
        self.token_size = token_spatial[0] * token_spatial[1] * cfg.enc_feats

        initial_channels = inshape[-1] * 2
        down_blocks = []
        in_channels = initial_channels
        for _ in range(cfg.steps):
            down_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=cfg.kernel_size,
                    downsample=True,
                )
            )
            in_channels = 64
        self.down_blocks = nn.ModuleList(down_blocks)
        self.token_conv = ConvBlock(
            in_channels=64,
            out_channels=cfg.enc_feats,
            kernel_size=cfg.kernel_size,
            flatten=True,
        )
        self.pos_embedding = PosEmbedding(self.num_tokens, self.token_size)
        self.transformers = nn.ModuleList(
            [
                TransformerEncoder(
                    num_heads=cfg.num_heads,
                    num_tokens=self.num_tokens,
                    projection_dim=self.token_size,
                    causal=cfg.causal,
                    dropout_rate=cfg.dropout_rate,
                )
                for _ in range(cfg.num_transformers)
            ]
        )
        up_blocks = []
        in_channels = cfg.enc_feats
        for _ in range(cfg.steps):
            up_blocks.append(
                ConvBlock(
                    in_channels=in_channels + 64,
                    out_channels=64,
                    kernel_size=cfg.kernel_size,
                    upsample=True,
                )
            )
            in_channels = 64
        self.up_blocks = nn.ModuleList(up_blocks)
        padding = cfg.kernel_size // 2
        self.out_conv = nn.Conv3d(64, 1, cfg.kernel_size, padding=padding)

    def forward(self, ct_vol: torch.Tensor, ray_tr: torch.Tensor) -> torch.Tensor:
        if ct_vol.shape != ray_tr.shape:
            raise ValueError("ct_vol and ray_tr must have the same shape")
        if ct_vol.dim() != 5:
            raise ValueError("Inputs must have shape (batch, tokens, height, width, channels)")

        x = torch.cat((ct_vol, ray_tr), dim=-1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)
        x_history = [x]

        for block in self.down_blocks:
            x = block(x)
            x_history.append(x)

        tokens = self.token_conv(x)  # (B, D, token_size)
        tokens = self.pos_embedding(tokens)

        for encoder in self.transformers:
            tokens = encoder(tokens)

        b, num_tokens, _ = tokens.shape
        tokens = tokens.reshape(
            b,
            num_tokens,
            self.token_spatial[0],
            self.token_spatial[1],
            self.enc_feats,
        )
        x = tokens.permute(0, 4, 1, 2, 3).contiguous()

        for block in self.up_blocks:
            skip = x_history.pop()
            x = torch.cat((x, skip), dim=1)
            x = block(x)

        dose = self.out_conv(x)
        dose = dose.permute(0, 2, 3, 4, 1).contiguous()
        return dose


def dota_residual(
    inshape,
    steps,
    enc_feats,
    num_heads,
    num_transformers,
    kernel_size,
    dropout_rate: float = 0.2,
    causal: bool = True,
) -> DotaResidual:
    cfg = DotaResidualConfig(
        inshape=inshape,
        steps=steps,
        enc_feats=enc_feats,
        num_heads=num_heads,
        num_transformers=num_transformers,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        causal=causal,
    )
    return DotaResidual(cfg)
