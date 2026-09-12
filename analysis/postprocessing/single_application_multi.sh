#!/usr/bin/env bash
# 단회 정제 효용 시험을 추가 checkpoint(t0=203, 206; 방출 주기 5.1 t.u. 대비 서로 다른 위상)에서 반복. 마스크별 m 은 t0=200 보정값 재사용.
set -u
ROOT=${DLAMR_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; CR=${DLAMR_CASES:-$ROOT/cases}; PY=python
declare -A MM=( [sigma]=2108 [gradU]=2106 [vort]=2085 [Q]=2027 [dt]=2117 [wake]=2158 [oracle]=2132 [rand]=1862 )
for T0 in 203 206; do
  echo "=== prep t0=$T0 $(date +%H:%M:%S)"
  $PY -u $ROOT/analysis/postprocessing/wp1b_mask_experiment.py --stage prep --geom circular --t0 $T0
  for k in sigma gradU vort Q dt wake oracle rand; do
    $PY -u $ROOT/analysis/postprocessing/wp1b_mask_experiment.py --stage cases --geom circular --t0 $T0 --masks $k --m ${MM[$k]}
  done
  echo "=== run t0=$T0 (4 parallel) $(date +%H:%M:%S)"
  printf "%s\n" sigma gradU vort Q dt wake oracle rand | xargs -P 4 -I{} bash -c "cd $CR/wp1b_circular_t${T0}_{} && ./Allrun && echo done {} \$(grep -oE 'Refined from [0-9]+ to [0-9]+ cells' log.pimpleFoam | tail -1) fatal=\$(grep -c 'FOAM FATAL' log.pimpleFoam)"
  echo "=== evaluate t0=$T0 $(date +%H:%M:%S)"
  $PY -u $ROOT/analysis/output/eval/wp1b_evaluate.py --geom circular --t0 $T0 2>&1 | grep -v Warn | tail -15
done
echo WP1B_MULTI_DONE
