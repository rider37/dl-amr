#!/bin/sh
# Quick reproducibility smoke test (no full simulation).
# Verifies:
#   1. Python environment imports the training/analysis modules
#   2. Pretrained model loads successfully
#   3. OpenFOAM solver binary is present
#   4. Case templates have required files

set -e
cd "$(dirname "$0")/.."

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
check "python -c 'import torch; import numpy; import scipy; import matplotlib'" "PyTorch + scientific stack"

echo ""
echo "Pretrained model..."
if [ -f ml/pretrained/heteroscedastic_unet.pt ]; then
    check "python -c 'import torch; m = torch.jit.load(\"ml/pretrained/heteroscedastic_unet.pt\"); m.eval()'" "TorchScript model loads"
else
    echo "  ! Model not downloaded (run: make download-models)"
fi

echo ""
echo "OpenFOAM solver..."
check "command -v amrPimpleFoam" "amrPimpleFoam in PATH"

echo ""
echo "Case templates..."
for geom in circular_Re200 square_Re150 diamond_Re150; do
    for method in fine coarse dl_amr grad_amr; do
        d="cases/$geom/$method"
        check "[ -d $d/0 ] && [ -d $d/constant ] && [ -d $d/system ] && [ -f $d/system/blockMeshDict ]" "$geom/$method"
    done
done

echo ""
echo "==========================================="
echo "Result: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
