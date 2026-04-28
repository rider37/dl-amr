from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttnDeltaFullRes(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 32):
        super().__init__()
        feats = base

        # Encoder (4 levels)
        self.enc1 = DoubleConv(in_ch, feats)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(feats, feats * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(feats * 2, feats * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(feats * 4, feats * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(feats * 8, feats * 16)

        # Decoder (4 levels) with skip concatenation
        self.up4 = nn.ConvTranspose2d(feats * 16, feats * 8, 2, 2)
        self.dec4 = DoubleConv(feats * 8 + feats * 8, feats * 8)
        self.up3 = nn.ConvTranspose2d(feats * 8, feats * 4, 2, 2)
        self.dec3 = DoubleConv(feats * 4 + feats * 4, feats * 4)
        self.up2 = nn.ConvTranspose2d(feats * 4, feats * 2, 2, 2)
        self.dec2 = DoubleConv(feats * 2 + feats * 2, feats * 2)
        self.up1 = nn.ConvTranspose2d(feats * 2, feats, 2, 2)
        self.dec1 = DoubleConv(feats + feats, feats)

        # g_scale(레거시 의미 유지): 학습되지 않는 상수.
        self.register_buffer("g_scale", torch.tensor(1.0))

        # Heads at full resolution
        mid = max(8, feats // 2)
        self.attn_head = nn.Sequential(
            nn.Conv2d(feats, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.out_head = nn.Conv2d(feats, out_ch, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        feat = self.dec1(torch.cat([d1, e1], dim=1))

        attn = self.attn_head(feat)
        scale = torch.exp(self.g_scale * (attn - 0.5) * 2.0)
        feat = feat * scale
        out = self.out_head(feat)
        return out, attn

    @torch.jit.export
    def attn_only(self, x: torch.Tensor) -> torch.Tensor:
        _, a = self.forward(x)
        return a
