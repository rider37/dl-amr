from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("SSIM inputs must have same shape")
    device = x.device
    kernel = _gaussian_kernel(window_size, sigma, device)
    kernel = kernel.expand(x.size(1), 1, window_size, window_size)

    mu_x = F.conv2d(x, kernel, padding=window_size // 2, groups=x.size(1))
    mu_y = F.conv2d(y, kernel, padding=window_size // 2, groups=y.size(1))

    sigma_x = F.conv2d(x * x, kernel, padding=window_size // 2, groups=x.size(1)) - mu_x**2
    sigma_y = F.conv2d(y * y, kernel, padding=window_size // 2, groups=y.size(1)) - mu_y**2
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2, groups=y.size(1)) - mu_x * mu_y

    c1 = 0.01**2
    c2 = 0.03**2

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return 1 - ssim.mean()
