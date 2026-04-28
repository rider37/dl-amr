from __future__ import annotations

import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    def __call__(self, sample):
        for k in ["x", "y", "mask"]:
            if isinstance(sample[k], torch.Tensor):
                continue
            sample[k] = torch.as_tensor(sample[k])
        return sample


class Normalize:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample["x"] = (sample["x"] - self.mean) / max(self.std, 1e-8)
        return sample


class NormalizeByStats:
    def __init__(self, mean, std, eps: float = 1e-8):
        self.mean = torch.as_tensor(mean).view(-1, 1, 1)
        self.std = torch.as_tensor(std).view(-1, 1, 1)
        self.eps = eps

    def __call__(self, sample):
        x = sample["x"]
        mean = self.mean.to(x.device, dtype=x.dtype)
        std = self.std.to(x.device, dtype=x.dtype)
        sample["x"] = (x - mean) / (std + self.eps)
        return sample


class ScaleTarget:
    def __init__(self, scale, eps: float = 1e-8):
        self.scale = torch.as_tensor(scale)
        self.eps = eps

    def __call__(self, sample):
        y = sample["y"]
        scale = self.scale.to(y.device, dtype=y.dtype)
        if scale.numel() == 1:
            sample["y"] = y * scale
        else:
            sample["y"] = y * scale.view(-1, 1, 1)
        return sample


class NormalizeTargetByStats:
    def __init__(self, mean, std, eps: float = 1e-8):
        self.mean = torch.as_tensor(mean).view(-1, 1, 1)
        self.std = torch.as_tensor(std).view(-1, 1, 1)
        self.eps = eps

    def __call__(self, sample):
        y = sample["y"]
        mean = self.mean.to(y.device, dtype=y.dtype)
        std = self.std.to(y.device, dtype=y.dtype)
        sample["y"] = (y - mean) / (std + self.eps)
        return sample
