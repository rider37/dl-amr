#!/bin/sh
# Download the pretrained TorchScript heteroscedastic U-Net.
#
# Sources, in order:
#   1. Zenodo: latest version of the concept record (10.5281/zenodo.19870610),
#      resolved at runtime via the Zenodo REST API. Stable across releases.
#   2. GitHub Release: $REPO @ $RELEASE_TAG (mirror); blank tag = latest.
#
# Override knobs (all env-var, blank-friendly):
#   ZENODO_CONCEPT  Zenodo concept record ID (default: 19870610)
#   ZENODO_RECORD   pin a specific version record ID (overrides resolver)
#   RELEASE_TAG     specific GitHub Release tag (default: latest)
#   REPO            GitHub owner/repo (default: rider37/dl-amr)
#   ARCHIVE         tarball filename (default: pretrained_models.tar.gz)
#   EXPECTED_SHA256 if set, sha256sum is verified after download

set -e

ZENODO_CONCEPT="${ZENODO_CONCEPT:-19870610}"
ZENODO_RECORD="${ZENODO_RECORD:-}"
RELEASE_TAG="${RELEASE_TAG:-}"
REPO="${REPO:-rider37/dl-amr}"
ARCHIVE="${ARCHIVE:-pretrained_models.tar.gz}"
EXPECTED_SHA256="${EXPECTED_SHA256:-}"

# Resolve concept -> latest version record ID via Zenodo REST API
# (concept-level /files/ URLs return 404, so we have to redirect ourselves).
if [ -z "$ZENODO_RECORD" ] && command -v curl >/dev/null 2>&1; then
    ZENODO_RECORD=$(curl -sL "https://zenodo.org/api/records/${ZENODO_CONCEPT}" 2>/dev/null \
        | grep -oE '"id"[[:space:]]*:[[:space:]]*[0-9]+' | head -1 \
        | grep -oE '[0-9]+')
fi
# Last-resort: try the concept ID directly (rarely useful, kept for diagnostics).
ZENODO_RECORD="${ZENODO_RECORD:-${ZENODO_CONCEPT}}"

DEST="ml/pretrained"
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading ${ARCHIVE}"
echo "  Zenodo concept : ${ZENODO_CONCEPT}"
echo "  Resolved record: ${ZENODO_RECORD}"
echo "  Release        : ${REPO}@${RELEASE_TAG:-<latest>}"

if wget -q "https://zenodo.org/records/${ZENODO_RECORD}/files/${ARCHIVE}" -O "${ARCHIVE}"; then
    echo "  Source: Zenodo"
elif command -v gh >/dev/null 2>&1; then
    echo "  Zenodo download failed; trying GitHub Release..."
    if [ -n "$RELEASE_TAG" ]; then
        gh release download "${RELEASE_TAG}" --repo "${REPO}" --pattern "${ARCHIVE}"
    else
        gh release download --repo "${REPO}" --pattern "${ARCHIVE}"
    fi
    echo "  Source: GitHub Release (${REPO}@${RELEASE_TAG:-latest})"
else
    echo "ERROR: Could not download ${ARCHIVE}." >&2
    echo "Try one of:" >&2
    echo "  - manual: https://doi.org/10.5281/zenodo.${ZENODO_CONCEPT}" >&2
    echo "  - set ZENODO_RECORD / RELEASE_TAG / REPO env vars and re-run" >&2
    echo "  - install the 'gh' CLI to use the GitHub Release fallback" >&2
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
