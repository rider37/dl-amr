#!/bin/bash
#------------------------------------------------------------------------------
# Convert an OpenFOAM case from dynamicRefineFvMesh (hexRef8, splits the span
# direction) to dynamicRefine2DFvMesh (hexRef4, in-plane only).
#
#   migrate2D.sh [-n] <case> [<case> ...]
#
#   -n   dry run: report what would change, touch nothing
#
# Edits, per case:
#   constant/dynamicMeshDict   dynamicFvMesh type + <type>Coeffs sub-dict name
#   system/controlDict         adds  libs ("libhexRef4.so");  if missing
#
# Originals are kept as *.hexRef8.bak. Nothing is run.
#------------------------------------------------------------------------------
set -u

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DRY=0
if [ "${1:-}" = "-n" ]; then DRY=1; shift; fi

if [ $# -eq 0 ]; then
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

note() { echo "    $*"; }

for CASE in "$@"; do
    echo
    echo "=== $CASE"

    DM="$CASE/constant/dynamicMeshDict"
    CD="$CASE/system/controlDict"

    if [ ! -f "$DM" ]; then
        note "no constant/dynamicMeshDict -- skipped"
        continue
    fi

    if grep -q "dynamicRefine2DFvMesh" "$DM"; then
        note "already using dynamicRefine2DFvMesh"
    elif ! grep -q "dynamicRefineFvMesh" "$DM"; then
        note "dynamicMeshDict does not use dynamicRefineFvMesh -- skipped"
        note "  found: $(grep -E '^\s*dynamicFvMesh' "$DM" | head -1)"
        continue
    else
        if [ $DRY -eq 1 ]; then
            note "would rewrite dynamicRefineFvMesh -> dynamicRefine2DFvMesh in $DM"
        else
            cp -n "$DM" "$DM.hexRef8.bak"
            # Order matters: the Coeffs sub-dict name contains the type name.
            sed -i \
                -e 's/dynamicRefineFvMeshCoeffs/dynamicRefine2DFvMeshCoeffs/g' \
                -e 's/\(dynamicFvMesh[[:space:]]\+\)dynamicRefineFvMesh/\1dynamicRefine2DFvMesh/' \
                "$DM"
            note "rewrote $DM  (backup: $(basename "$DM").hexRef8.bak)"
        fi
    fi

    # controlDict: libs entry
    if [ -f "$CD" ]; then
        if grep -q "libhexRef4" "$CD"; then
            note "controlDict already loads libhexRef4"
        elif [ $DRY -eq 1 ]; then
            note "would add  libs (\"libhexRef4.so\");  to $CD"
        else
            cp -n "$CD" "$CD.hexRef8.bak"
            if grep -qE '^\s*libs' "$CD"; then
                note "controlDict has an existing 'libs' entry -- add"
                note "  \"libhexRef4.so\" to it by hand"
            else
                # insert after the FoamFile block
                awk '
                    /^}/ && !done { print; print ""; print "libs            (\"libhexRef4.so\");"; done=1; next }
                    { print }
                ' "$CD" > "$CD.tmp" && mv "$CD.tmp" "$CD"
                note "added libs entry to $CD"
            fi
        fi
    else
        note "no system/controlDict -- add  libs (\"libhexRef4.so\");  yourself"
    fi

    # maxCells warning
    MC=$(grep -oE 'maxCells[[:space:]]+[0-9]+' "$DM" 2>/dev/null | head -1 | grep -oE '[0-9]+')
    if [ -n "${MC:-}" ]; then
        note "maxCells is $MC -- note that each split now adds 3 cells, not 7,"
        note "  so the same number buys roughly 2.3x more refinement events."
    fi

    # audit the starting mesh
    if [ -d "$CASE/constant/polyMesh" ]; then
        note "auditing constant/polyMesh:"
        python3 "$HERE/check2D.py" "$CASE/constant/polyMesh" 2>&1 \
            | grep -E "empty faces|span|NOT 2-D|ONE CELL" | sed 's/^/      /'
    fi
done

echo
echo "Done.${DRY:+ (dry run)}"
echo "Solver code doing refCast<dynamicRefineFvMesh>(mesh) must be changed to"
echo "refCast<dynamicRefine2DFvMesh>(mesh) and include dynamicRefine2DFvMesh.H."
