from __future__ import annotations

import torch


def sliced_wasserstein_distance(x: torch.Tensor, y: torch.Tensor, num_projections: int = 16) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("Inputs must have same shape for Sliced Wasserstein")
    b = x.shape[0]
    x_flat = x.view(b, -1)
    y_flat = y.view(b, -1)
    dim = x_flat.size(1)
    device = x.device
    proj = torch.randn(num_projections, dim, device=device)
    proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)

    x_proj = x_flat @ proj.t()
    y_proj = y_flat @ proj.t()

    x_proj, _ = torch.sort(x_proj, dim=0)
    y_proj, _ = torch.sort(y_proj, dim=0)

    return torch.mean(torch.abs(x_proj - y_proj))
