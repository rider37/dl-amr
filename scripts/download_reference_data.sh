#!/bin/sh
# Download the minimal reference data needed by Tier-2 figures
# (Figs 9, 10, 11, 12: test.pt and per-sample predictions).
#
# Tier-3 figures (1, 3, 4, 5, 7, 8) need full OpenFOAM case results under
# cases/<geom>/<method>/, obtained by running ./Allrun. Not in this archive.
#
# Tarball layout (extracted at repo root):
#   reference_data/test.pt
#   reference_data/preds/{00000.npz, 00001.npz, ...}
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
#   ARCHIVE         tarball filename (default: reference_data_minimal.tar.gz)

set -e

ZENODO_CONCEPT="${ZENODO_CONCEPT:-19870610}"
ZENODO_RECORD="${ZENODO_RECORD:-}"
RELEASE_TAG="${RELEASE_TAG:-}"
REPO="${REPO:-rider37/dl-amr}"
ARCHIVE="${ARCHIVE:-reference_data_minimal.tar.gz}"

# Resolve concept -> latest version record ID via Zenodo REST API.
if [ -z "$ZENODO_RECORD" ] && command -v curl >/dev/null 2>&1; then
    ZENODO_RECORD=$(curl -sL "https://zenodo.org/api/records/${ZENODO_CONCEPT}" 2>/dev/null \
        | grep -oE '"id"[[:space:]]*:[[:space:]]*[0-9]+' | head -1 \
        | grep -oE '[0-9]+')
fi
ZENODO_RECORD="${ZENODO_RECORD:-${ZENODO_CONCEPT}}"

# Run from repo root (parent of scripts/).
cd "$(dirname "$0")/.."

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

echo "Extracting into reference_data/ ..."
tar -xzf "${ARCHIVE}"
rm "${ARCHIVE}"

echo ""
echo "Done. Verify:"
ls -lh reference_data/test.pt 2>/dev/null || echo "  (missing) reference_data/test.pt"
echo "  reference_data/preds/: $(ls reference_data/preds 2>/dev/null | wc -l) files"
