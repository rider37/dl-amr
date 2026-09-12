#!/usr/bin/env bash
# 마스크별 m 을 보정해 실현 셀 수를 49,884 에 맞춘다.
# 짧은 파일럿(t0+0.4)으로 정제 후 셀 수를 읽고 m 을 비례 조정, 최대 4회.
set -u
ROOT=${DLAMR_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; CR=${DLAMR_CASES:-$ROOT/cases}
PY=python
GEOM=circular; T0=200; TARGET=49884; BASE=16944
declare -A MM
for m in sigma gradU vort Q dt wake oracle rand; do MM[$m]=2196; done
for it in 1 2 3 4; do
  echo "===== 보정 $it 회차"
  for k in sigma gradU vort Q dt wake oracle rand; do
    m=${MM[$k]}
    $PY -u $ROOT/analysis/postprocessing/wp1b_mask_experiment.py --stage cases --geom $GEOM --t0 $T0 \
        --masks $k --m $m --endtime 0.4 > /dev/null 2>&1
    d=$CR/wp1b_${GEOM}_t${T0}_${k}
    ( cd "$d" && ./Allrun >/dev/null 2>&1 )
    n=$(grep -oE 'Refined from [0-9]+ to [0-9]+ cells' "$d/log.pimpleFoam" 2>/dev/null | tail -1 | awk '{print $5}')
    [ -z "$n" ] && { echo "  $k: 정제 기록 없음"; continue; }
    dev=$(( (n-TARGET)*1000/TARGET ))
    printf "  %-7s m=%-6s 실현=%-7s 편차=%+d‰\n" "$k" "$m" "$n" "$dev"
    if [ ${dev#-} -le 5 ]; then continue; fi
    new=$(( m * (TARGET-BASE) / (n-BASE) ))
    MM[$k]=$new
  done
done
echo "===== 최종 m"
for k in sigma gradU vort Q dt wake oracle rand; do echo "  $k ${MM[$k]}"; done
