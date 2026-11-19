from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet3D(nn.Module):
    """A deliberately small 3D U-Net suitable for a laptop demo."""
    def __init__(self, base=8):
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.pool = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.up = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec = ConvBlock(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        skip = self.enc1(x)
        x = self.enc2(self.pool(skip))
        x = self.up(x)
        x = self.dec(torch.cat([x, skip], dim=1))
        return self.out(x)


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()

