from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_norm_stats(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"norm stats not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
