"""Skip-friendly reference-data presence helper.

Each figure script that needs externally distributed data calls
``require_or_skip(...)`` near its top. If any required path is missing the
script prints a clear, single-block message naming the missing files and
exits with **code 2** (reserved by ``scripts/generate_figures.sh`` for
*skipped — data not available*, distinct from real failures).
"""
import os
import sys

SKIP_EXIT_CODE = 2


def require_or_skip(figure, hint, *paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if not missing:
        return
    sys.stderr.write(
        f"\n[SKIP] {figure}: required reference data not found.\n"
    )
    for p in missing:
        sys.stderr.write(f"   - {p}\n")
    sys.stderr.write(f"   {hint}\n")
    sys.exit(SKIP_EXIT_CODE)
