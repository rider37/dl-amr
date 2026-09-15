#!/bin/sh
# Regenerate the paper figures from the downloaded reference data.
#   Tier 2 (make download-reference):  Figs 8, 9, E.1        -> analysis/uncertainty_figures.py
#   Tier 2 (make download-fields):     Figs 4, 5, 7, D.1, F.1 -> analysis/figscripts/{make_fig4_D1,make_fig5,make_fig7_F1}.py
#   Tier 3 (raw case results):         Figs 3, 6           -> analysis/figscripts/make_fig3_6.py
# Scripts whose data are absent print [SKIP] and exit with code 2.
#
#   $DL_AMR_OUTDIR  output directory (default: analysis/output/)
#   $DLAMR_REFDATA  reference_data/ (test.pt, preds/)
#   $DLAMR_CACHE    reference_data/fields/
#   $DLAMR_CASES    cases/ (raw results)
#   $PYTHON         interpreter (default: python, then python3)
set -e
cd "$(dirname "$0")/.."
if [ -n "$PYTHON" ]; then PY="$PYTHON";
elif command -v python >/dev/null 2>&1; then PY=python;
elif command -v python3 >/dev/null 2>&1; then PY=python3;
else echo "ERROR: no python interpreter found (set \$PYTHON)"; exit 1; fi
OUTDIR="${DL_AMR_OUTDIR:-analysis/output}"; mkdir -p "$OUTDIR"; export DL_AMR_OUTDIR="$OUTDIR"
echo "Output directory: $OUTDIR"; echo "Python: $PY ($($PY --version 2>&1))"; echo ""
SCRIPTS="analysis/uncertainty_figures.py analysis/figscripts/make_fig4_D1.py analysis/figscripts/make_fig5.py analysis/figscripts/make_fig7_F1.py analysis/figscripts/make_fig3_6.py"
passed=0; skipped=0; failed=0; set +e
for s in $SCRIPTS; do
    echo "=== $s ==="; "$PY" "$s"; rc=$?
    case $rc in 0) passed=$((passed+1)) ;; 2) skipped=$((skipped+1)); echo "  --> SKIPPED (data not available)" ;; *) failed=$((failed+1)); echo "  --> FAILED (exit $rc)" ;; esac
    echo ""
done
echo "==================================================="; echo "  Result: $passed produced, $skipped skipped, $failed failed"; echo "  Output: $OUTDIR"; echo "==================================================="
[ "$failed" -eq 0 ] || exit 1
