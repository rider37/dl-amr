#!/bin/bash
# 3단계 폐루프 ablation 평가: 표 1과 동일 파이프라인(nc2_metrics*.py, 형상별 평가창). 출력 → analysis/output/eval/stage3/*.log
set -e; cd "${DLAMR_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"; mkdir -p analysis/output/eval/stage3; P=python
W0=165   W1=245   $P analysis/output/eval/nc2_metrics.py     NC49k_sigma NC49k_mu NC49k_vortgrid NC49k_vort KLW1_circular_kelly 2>&1 | grep -v Warn | tee analysis/output/eval/stage3/circ.log
W0=237.5 W1=346.5 $P analysis/output/eval/nc2_metrics_sq.py  SQ49k_sigma SQ49k_mu  2>&1 | grep -v Warn | tee analysis/output/eval/stage3/sq.log
W0=170   W1=256   $P analysis/output/eval/nc2_metrics_dia.py DIA49k_sigma DIA49k_mu 2>&1 | grep -v Warn | tee analysis/output/eval/stage3/dia.log
echo STAGE3_EVAL_DONE
