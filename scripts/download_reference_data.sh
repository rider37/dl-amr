#!/bin/sh
# Download minimal reference data (time-averaged fields, force coefficients,
# metric tables) from Zenodo or GitHub Release.

set -e

ZENODO_RECORD="XXXXXXX"
RELEASE_TAG="v1.0.0"
ARCHIVE="reference_data_minimal.tar.gz"

DEST="reference_data"
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading $ARCHIVE..."

if wget -q "https://zenodo.org/records/${ZENODO_RECORD}/files/${ARCHIVE}" -O "${ARCHIVE}"; then
    echo "  Source: Zenodo"
elif command -v gh >/dev/null 2>&1; then
    gh release download "${RELEASE_TAG}" --pattern "${ARCHIVE}"
    echo "  Source: GitHub Release"
else
    echo "ERROR: Could not download ${ARCHIVE}."
    exit 1
fi

tar -xzf "${ARCHIVE}"
rm "${ARCHIVE}"

echo "Done. Reference data extracted to $(pwd)/"
