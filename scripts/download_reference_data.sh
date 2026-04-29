#!/bin/sh
# Download the minimal reference data needed by Tier-2 figures
# (Figs 9, 10, 11, 12: test.pt and per-sample predictions).
#
# Tier-3 figures (1, 3, 4, 5, 7, 8) need full OpenFOAM case results under
# cases/<geom>/<method>/, obtained by running ./Allrun or by pulling an
# optional case-bundle archive separately. They are NOT in this archive.
#
# Tarball layout (extracted at repo root):
#   reference_data/test.pt
#   reference_data/preds/{00000.npz, 00001.npz, ...}
#
# Sources, in order:
#   1. Zenodo record (primary, DOI-cited)
#   2. GitHub Release of $REPO with tag $RELEASE_TAG (mirror)
#
# Override at runtime, e.g.:
#   ZENODO_RECORD=<id>  RELEASE_TAG=v1.0.2  REPO=rider37/dl-amr \
#       bash scripts/download_reference_data.sh
#
# Pinned to the same release as scripts/download_models.sh:
#   Concept DOI (always-latest): 10.5281/zenodo.19870610
#   v1.0.2 version DOI         : 10.5281/zenodo.19873110

set -e

ZENODO_RECORD="${ZENODO_RECORD:-19873110}"
RELEASE_TAG="${RELEASE_TAG:-v1.0.2}"
REPO="${REPO:-rider37/dl-amr}"
ARCHIVE="${ARCHIVE:-reference_data_minimal.tar.gz}"

# Run from repo root (parent of scripts/).
cd "$(dirname "$0")/.."

echo "Downloading ${ARCHIVE}"
echo "  Zenodo record : ${ZENODO_RECORD}"
echo "  Release       : ${REPO}@${RELEASE_TAG}"

if wget -q "https://zenodo.org/records/${ZENODO_RECORD}/files/${ARCHIVE}" -O "${ARCHIVE}"; then
    echo "  Source: Zenodo"
elif command -v gh >/dev/null 2>&1; then
    echo "  Zenodo download failed; trying GitHub Release..."
    gh release download "${RELEASE_TAG}" --repo "${REPO}" --pattern "${ARCHIVE}"
    echo "  Source: GitHub Release (${REPO}@${RELEASE_TAG})"
else
    echo "ERROR: Could not download ${ARCHIVE}." >&2
    echo "Try one of:" >&2
    echo "  - manually: https://zenodo.org/records/${ZENODO_RECORD}/files/${ARCHIVE}" >&2
    echo "  - or set ZENODO_RECORD / RELEASE_TAG / REPO env vars and re-run" >&2
    echo "  - or install the 'gh' CLI to use the GitHub Release fallback" >&2
    exit 1
fi

echo "Extracting into reference_data/ ..."
tar -xzf "${ARCHIVE}"
rm "${ARCHIVE}"

echo ""
echo "Done. Verify:"
ls -lh reference_data/test.pt 2>/dev/null || echo "  (missing) reference_data/test.pt"
echo "  reference_data/preds/: $(ls reference_data/preds 2>/dev/null | wc -l) files"
