#!/bin/sh
# Quick reproducibility smoke test (no full simulation).
# Verifies:
#   1. Python environment imports the training/analysis modules
#   2. Pretrained model loads successfully
#   3. OpenFOAM solver binary is present
#   4. Case templates have required files

set -e
cd "$(dirname "$0")/.."

# Pick interpreter: $PYTHON > python > python3.
if [ -n "$PYTHON" ]; then
    PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
    PY=python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "ERROR: neither 'python' nor 'python3' found in PATH (set \$PYTHON to override)"; exit 1
fi

PASS=0
FAIL=0
check() {
    if eval "$1" >/dev/null 2>&1; then
        echo "  ✓ $2"
        PASS=$((PASS+1))
    else
        echo "  ✗ $2"
        FAIL=$((FAIL+1))
    fi
}

echo "=== DL-AMR smoke test ==="

echo ""
echo "Python imports..."
check ""$PY" -c 'import torch; import numpy; import scipy; import matplotlib'" "PyTorch + scientific stack"
check ""$PY" -c 'from ml.src.models import HeteroDeltaFullRes, UNet'"             "ml.src.models importable"
check ""$PY" -c 'from ml.src.losses import bce_dice, gdl, ssim, sliced_wasserstein'" "ml.src.losses importable"
check ""$PY" -c 'from ml.src.dataloaders import AMRDataset, NormalizeByStats'"    "ml.src.dataloaders importable"

echo ""
echo "Analysis scripts syntax..."
syntax_ok=true
for f in analysis/*.py; do
    "$PY" -m py_compile "$f" 2>/dev/null || { echo "  ✗ Syntax error: $f"; syntax_ok=false; FAIL=$((FAIL+1)); }
done
$syntax_ok && { echo "  ✓ All analysis/*.py compile"; PASS=$((PASS+1)); }

echo ""
echo "Helper scripts executable..."
for s in scripts/*.sh; do
    if [ -x "$s" ]; then echo "  ✓ $s"; PASS=$((PASS+1)); else echo "  ✗ $s not executable"; FAIL=$((FAIL+1)); fi
done

echo ""
echo "Pretrained model..."
if [ -f ml/pretrained/heteroscedastic_unet.pt ]; then
    check ""$PY" -c 'import torch; m = torch.jit.load(\"ml/pretrained/heteroscedastic_unet.pt\"); m.eval()'" "TorchScript model loads"
else
    echo "  ! Model not downloaded (run: make download-models)"
fi

echo ""
echo "OpenFOAM solver (optional - needed only for full simulations)..."
if command -v amrPimpleFoam >/dev/null 2>&1; then
    echo "  ✓ amrPimpleFoam in PATH"
    PASS=$((PASS+1))
else
    echo "  ! amrPimpleFoam not found — source OpenFOAM env and run 'make solver':"
    echo "      source \$WM_PROJECT_DIR/etc/bashrc"
    echo "      export LIBTORCH_DIR=/path/to/libtorch && make solver"
    echo "    (smoke test does not fail on this; figures and ML training do not need it)"
fi

echo ""
echo "Case templates (paper geometries)..."
for geom in circular_Re200 square_Re150 diamond_Re150; do
    for method in fine coarse dl_amr grad_amr; do
        d="cases/$geom/$method"
        check "[ -d $d/0 ] && [ -d $d/constant ] && [ -d $d/system ] && [ -f $d/system/blockMeshDict ]" "$geom/$method"
    done
done

echo ""
echo "Training reference cases (circular Re=100,150)..."
for geom in circular_Re100 circular_Re150; do
    d="cases/$geom/fine"
    check "[ -d $d/0 ] && [ -d $d/constant ] && [ -d $d/system ] && [ -f $d/constant/transportProperties ]" "$geom/fine"
done

echo ""
echo "==========================================="
echo "Result: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
