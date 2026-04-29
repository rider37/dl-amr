#!/bin/sh
# Download the pretrained TorchScript heteroscedastic U-Net.
#
# Sources, in order:
#   1. Zenodo record (primary, DOI-cited)
#   2. GitHub Release of $REPO with tag $RELEASE_TAG (mirror)
#
# Override at runtime, e.g.:
#   ZENODO_RECORD=<id>  RELEASE_TAG=v1.0.3  REPO=rider37/dl-amr \
#       bash scripts/download_models.sh
#
# Pinned to the same release as scripts/download_reference_data.sh:
#   Concept DOI (always-latest): 10.5281/zenodo.19870610
#   v1.0.3 version DOI         : 10.5281/zenodo.19874536

set -e

ZENODO_RECORD="${ZENODO_RECORD:-19874536}"
RELEASE_TAG="${RELEASE_TAG:-v1.0.3}"
REPO="${REPO:-rider37/dl-amr}"
ARCHIVE="${ARCHIVE:-pretrained_models.tar.gz}"
EXPECTED_SHA256="${EXPECTED_SHA256:-}"   # to be filled after final artifact upload

DEST="ml/pretrained"
mkdir -p "$DEST"
cd "$DEST"

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

if [ -n "$EXPECTED_SHA256" ]; then
    echo "Verifying SHA256..."
    echo "${EXPECTED_SHA256}  ${ARCHIVE}" | sha256sum -c -
else
    echo "  (no EXPECTED_SHA256 provided; skipping checksum verification)"
fi

echo "Extracting..."
tar -xzf "${ARCHIVE}"
rm "${ARCHIVE}"

echo "Done. Models extracted to $(pwd)/"
ls -lh *.pt 2>/dev/null || true

# Symlink model into each dl_amr case so OpenFOAM can find it
cd ../..   # back to repo root
MODEL_FILE="ml/pretrained/heteroscedastic_unet.pt"
if [ -f "$MODEL_FILE" ]; then
    for case_dir in cases/*/dl_amr; do
        [ -d "$case_dir" ] || continue
        ln -sf "$(pwd)/$MODEL_FILE" "$case_dir/constant/model.ts"
        echo "  Linked: $case_dir/constant/model.ts"
    done
fi
