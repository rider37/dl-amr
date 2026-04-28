from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class AMRDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        task: str = "seg",
        transform=None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.task = task
        self.transform = transform

        self.mode = None
        self.data = None
        self.meta = None
        self._shards = None
        self._cum_counts = None
        self._shard_cache = None

        def _load_pt(pt_path: Path) -> bool:
            if not pt_path.exists() or pt_path.stat().st_size == 0:
                return False
            try:
                payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            except Exception:
                return False
            self.data = {
                "X": payload["X"],
                "y": payload["y"],
                "mask": payload["mask"],
            }
            self.meta = payload.get("meta", [])
            self.mode = "pt"
            return True

        if self.dataset_path.is_file() and self.dataset_path.suffix == ".pt":
            if not _load_pt(self.dataset_path):
                raise RuntimeError(f"Failed to load dataset: {self.dataset_path}")
        elif self.dataset_path.is_dir():
            pt_path = self.dataset_path / f"{split}.pt"
            zarr_path = self.dataset_path / f"{split}.zarr"
            shards_index = self.dataset_path / f"{split}_shards.json"
            if pt_path.exists() and _load_pt(pt_path):
                pass
            elif shards_index.exists():
                with open(shards_index, "r", encoding="utf-8") as f:
                    shard_info = json.load(f)
                self._shards = shard_info.get("shards", [])
                counts = [int(s.get("count", 0)) for s in self._shards]
                cum = []
                total = 0
                for c in counts:
                    total += c
                    cum.append(total)
                self._cum_counts = cum
                self.mode = "pt_sharded"
                self.length = total
                return
            elif zarr_path.exists():
                try:
                    import zarr
                except Exception as e:
                    raise RuntimeError("zarr not installed but zarr dataset requested") from e
                root = zarr.open(zarr_path, mode="r")
                self.data = {
                    "X": root["X"],
                    "y": root["y"],
                    "mask": root["mask"],
                }
                self.meta = json.loads(root.attrs.get("meta", "[]"))
                self.mode = "zarr"
            else:
                raise FileNotFoundError(
                    f"Unsupported dataset path. Provide .pt or folder containing {split}.pt/.zarr"
                )
        else:
            raise FileNotFoundError(
                f"Unsupported dataset path. Provide .pt or folder containing {split}.pt/.zarr"
            )

        if self.mode == "zarr":
            self.length = int(self.data["X"].shape[0])
        else:
            self.length = len(self.data["X"])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.mode == "pt_sharded":
            if self._cum_counts is None or self._shards is None:
                raise RuntimeError("Shard index not initialized")
            shard_idx = bisect_right(self._cum_counts, idx)
            start = 0 if shard_idx == 0 else self._cum_counts[shard_idx - 1]
            local_idx = idx - start
            if self._shard_cache is None or self._shard_cache["idx"] != shard_idx:
                shard_path = Path(self._shards[shard_idx]["path"])
                payload = torch.load(shard_path, map_location="cpu", weights_only=False)
                self._shard_cache = {"idx": shard_idx, "payload": payload}
            payload = self._shard_cache["payload"]
            x = payload["X"][local_idx]
            y = payload["y"][local_idx]
            mask = payload["mask"][local_idx]
        else:
            x = self.data["X"][idx]
            y = self.data["y"][idx]
            mask = self.data["mask"][idx]

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if y.ndim == 2:
            y = y.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if self.task == "seg":
            target = mask.float()
        else:
            target = y.float()

        sample = {"x": x.float(), "y": target, "mask": mask.float()}
        if self.transform:
            sample = self.transform(sample)
        return sample
