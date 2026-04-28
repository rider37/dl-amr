from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, depth: int = 4, base_channels: int = 32):
        super().__init__()
        self.depth = depth

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.downs.append(conv_block(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        self.bottleneck = conv_block(ch, ch * 2)
        ch = ch * 2

        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i in reversed(range(depth)):
            out_ch = base_channels * (2**i)
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=2, stride=2))
            self.ups.append(conv_block(ch, out_ch))
            ch = out_ch

        self.head = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, up, skip in zip(self.upconvs, self.ups, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.shape[-2] - x.shape[-2]
                diff_x = skip.shape[-1] - x.shape[-1]
                x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([skip, x], dim=1)
            x = up(x)
        return self.head(x)
