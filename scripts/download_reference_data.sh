#!/bin/sh
# Download minimal reference data needed by Tier-2 figures
# (Figs 9, 10, 11, 12: test.pt and per-sample predictions).
#
# Tier-3 figures (1, 3, 4, 5, 7, 8) require full OpenFOAM case results
# under cases/<geom>/<method>/. Either run the cases (./Allrun) or pull
# the optional case bundle archive separately.
#
# Tarball layout (extracted at repo root):
#   reference_data/test.pt
#   reference_data/preds/{00000.npz, 00001.npz, ...}

set -e

ZENODO_RECORD="${ZENODO_RECORD:-XXXXXXX}"
RELEASE_TAG="${RELEASE_TAG:-v1.0.0}"
ARCHIVE="reference_data_minimal.tar.gz"

# Run from repo root (parent of scripts/).
cd "$(dirname "$0")/.."

echo "Downloading $ARCHIVE..."

if wget -q "https://zenodo.org/records/${ZENODO_RECORD}/files/${ARCHIVE}" -O "${ARCHIVE}"; then
    echo "  Source: Zenodo (record ${ZENODO_RECORD})"
elif command -v gh >/dev/null 2>&1; then
    gh release download "${RELEASE_TAG}" --pattern "${ARCHIVE}"
    echo "  Source: GitHub Release (${RELEASE_TAG})"
else
    echo "ERROR: Could not download ${ARCHIVE}." >&2
    echo "Set ZENODO_RECORD and re-run, or install 'gh' CLI." >&2
    exit 1
fi

echo "Extracting into reference_data/ ..."
tar -xzf "${ARCHIVE}"
rm "${ARCHIVE}"

echo ""
echo "Done. Verify:"
ls -lh reference_data/test.pt 2>/dev/null || echo "  (missing) reference_data/test.pt"
echo "  reference_data/preds/: $(ls reference_data/preds 2>/dev/null | wc -l) files"
