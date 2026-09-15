#!/bin/sh
# Download the cached wake-field data (time-averaged fields, probe series, tables)
# used by Figs 4, 5, 7, D.1 and F.1 (see reference_data/README.md).
#
# Tarball layout (extracted at repo root):
#   reference_data/fields/nc4x_fig_cache/, c7/, tables/, *.json
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
#   ARCHIVE         tarball filename (default: reference_data_fields.tar.gz)

set -e

ZENODO_CONCEPT="${ZENODO_CONCEPT:-19870610}"
ZENODO_RECORD="${ZENODO_RECORD:-}"
RELEASE_TAG="${RELEASE_TAG:-}"
REPO="${REPO:-rider37/dl-amr}"
ARCHIVE="${ARCHIVE:-reference_data_fields.tar.gz}"

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

echo "Extracting into reference_data/fields/ ..."; mkdir -p reference_data/fields
tar -xzf "${ARCHIVE}" --strip-components=1 -C reference_data/fields
rm "${ARCHIVE}"

echo ""
echo "Done. Verify:"
ls reference_data/fields/nc4x_fig_cache 2>/dev/null | wc -l | sed 's/^/  reference_data\/fields\/nc4x_fig_cache: /; s/$/ files/'
