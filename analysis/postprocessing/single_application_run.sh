#!/usr/bin/env bash
# 보정된 m 으로 케이스 재생성 후 K=10 본런
set -u
ROOT=${DLAMR_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; CR=${DLAMR_CASES:-$ROOT/cases}
PY=python
declare -A MM=( [sigma]=2108 [gradU]=2106 [vort]=2085 [Q]=2027 [dt]=2117 [wake]=2158 [oracle]=2132 [rand]=1862 )
for k in sigma gradU vort Q dt wake oracle rand; do
  echo "=== $k  m=${MM[$k]}  $(date +%H:%M:%S)"
  $PY -u $ROOT/analysis/postprocessing/wp1b_mask_experiment.py --stage cases --geom circular --t0 200 \
      --masks $k --m ${MM[$k]} >/dev/null 2>&1
  d=$CR/wp1b_circular_t200_${k}
  ( cd "$d" && ./Allrun )
  echo "    $(grep -E '^Time = ' $d/log.pimpleFoam | tail -1)  fatal=$(grep -c 'FOAM FATAL' $d/log.pimpleFoam)  $(grep -oE 'Refined from [0-9]+ to [0-9]+ cells' $d/log.pimpleFoam | tail -1)"
done
echo "=== 본런 완료 $(date +%H:%M:%S)"
