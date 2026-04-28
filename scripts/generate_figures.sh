#!/bin/sh
# Regenerate paper figures from analysis scripts.
# Requires reference data: run `make download-reference` first
# (see reference_data/README.md for the expected file layout).
#
# Output directory:
#   $DL_AMR_OUTDIR (default: analysis/output/)
# Python interpreter:
#   $PYTHON          (default: auto-detect 'python' then 'python3')

set -e
cd "$(dirname "$0")/.."

if [ -n "$PYTHON" ]; then
    PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
    PY=python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "ERROR: neither 'python' nor 'python3' found in PATH (set \$PYTHON to override)"
    exit 1
fi

OUTDIR="${DL_AMR_OUTDIR:-analysis/output}"
mkdir -p "$OUTDIR"
echo "Output directory: $OUTDIR"
echo "Python:           $PY ($($PY --version 2>&1))"
echo ""

# Order: lighter figures first, OpenFOAM-data-heavy ones last.
SCRIPTS="
generate_fig6_error_vs_dof.py
generate_fig9_10_11_uncertainty.py
generate_fig12_threshold_sensitivity.py
generate_fig1_7_overview.py
generate_fig3_phase_averaged.py
generate_fig4_5_instantaneous_and_umean.py
generate_fig8_anchor_variants.py
"

passed=0
skipped=0
failed=0
# Loop with set +e so individual script exits don't abort the orchestrator.
set +e
for s in $SCRIPTS; do
    echo "=== Running analysis/$s ==="
    "$PY" analysis/$s
    rc=$?
    case $rc in
        0)  passed=$((passed+1)) ;;
        2)  skipped=$((skipped+1)); echo "  --> SKIPPED (reference data not available)" ;;
        *)  failed=$((failed+1));  echo "  --> FAILED (exit $rc)" ;;
    esac
    echo ""
done

echo "==================================================="
echo "  Result: $passed produced, $skipped skipped, $failed failed"
echo "  Output: $OUTDIR"
echo "==================================================="
if [ "$skipped" -gt 0 ]; then
    echo "Note: skipped scripts need reference data — see reference_data/README.md."
fi
[ "$failed" -eq 0 ] || exit 1
