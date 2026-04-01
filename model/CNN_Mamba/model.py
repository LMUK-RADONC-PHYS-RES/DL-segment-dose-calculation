import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2  # mamba-ssm>=2.x


class Mamba2BlockDRes(nn.Module):
    """Pre-Norm + Mamba2 + Dropout + Res"""
    def __init__(self, d: int, dropout: float = 0.05, res_scale: float = 1.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.core = Mamba2(d_model=d)
        self.drop = nn.Dropout(dropout)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, T, d)
        y = self.core(self.norm(x))
        return x + self.res_scale * self.drop(y)



class CNN_Mamba2_L2_SpatialMix(nn.Module):
    def __init__(self, input_h: int = 200, input_w: int = 200,
                 d: int = 32, layers: int = 2, dropout: float = 0.05) -> None:
        super().__init__()
        self.input_h, self.input_w = input_h, input_w
        self.d = d

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
        )

        # depthwise 7×7 + pointwise 1×1 
        self.spatial_pre = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, groups=64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.in_proj  = nn.Linear(64, d) if d != 64 else nn.Identity()
        self.blocks   = nn.ModuleList([Mamba2BlockDRes(d=d, dropout=dropout) for _ in range(layers)])
        self.out_proj = nn.Linear(d, 64) if d != 64 else nn.Identity()

        # depthwise 7×7 + pointwise 1×1
        self.spatial_post = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, groups=64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 16, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 1, 1, 1, 0),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1, x2: (B, T, H, W)
        B, T, H, W = x1.shape

        # (B*T, 2, H, W)
        x = torch.stack((x1, x2), dim=2).reshape(B * T, 2, H, W)

        # (B*T, 64, 25, 25)
        x = self.encoder(x)
        _, C, Hc, Wc = x.shape  # C=64

        x = x + self.spatial_pre(x)

        seq = x.view(B, T, C, Hc, Wc).permute(0, 3, 4, 1, 2).reshape(B * Hc * Wc, T, C)  # (N, T, 64)

        z = self.in_proj(seq)  # (N, T, d)
        for blk in self.blocks:
            z = blk(z)         # (N, T, d)
        seq = self.out_proj(z) # (N, T, 64)

        # (B*T, 64, 25, 25)
        x = seq.view(B, Hc, Wc, T, C).permute(0, 3, 4, 1, 2).reshape(B * T, C, Hc, Wc)

        x = x + self.spatial_post(x)

        x = self.decoder(x).view(B, T, self.input_h, self.input_w)
        return x
