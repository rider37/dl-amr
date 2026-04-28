#!/bin/sh
# Download pretrained models from Zenodo (primary) or GitHub Release (mirror).
# Usage: ./scripts/download_models.sh

set -e

ZENODO_RECORD="XXXXXXX"   # update with actual Zenodo record ID
RELEASE_TAG="v1.0.0"
ARCHIVE="pretrained_models.tar.gz"
EXPECTED_SHA256=""        # update with actual checksum

DEST="ml/pretrained"
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading $ARCHIVE..."

# Try Zenodo first
if wget -q "https://zenodo.org/records/${ZENODO_RECORD}/files/${ARCHIVE}" -O "${ARCHIVE}"; then
    echo "  Source: Zenodo (DOI archive)"
elif command -v gh >/dev/null 2>&1; then
    echo "  Zenodo download failed; trying GitHub Release..."
    gh release download "${RELEASE_TAG}" --pattern "${ARCHIVE}"
    echo "  Source: GitHub Release"
else
    echo "ERROR: Could not download ${ARCHIVE}."
    echo "Install gh CLI or download manually from:"
    echo "  https://zenodo.org/records/${ZENODO_RECORD}"
    exit 1
fi

# Verify checksum if available
if [ -n "$EXPECTED_SHA256" ]; then
    echo "Verifying SHA256..."
    echo "${EXPECTED_SHA256}  ${ARCHIVE}" | sha256sum -c -
fi

echo "Extracting..."
tar -xzf "${ARCHIVE}"
rm "${ARCHIVE}"

echo "Done. Models extracted to $(pwd)/"
ls -lh *.pt 2>/dev/null || true
