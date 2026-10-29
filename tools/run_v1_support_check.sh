#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PROJECT_ROOT/media/v1_support_check_20261024"
support_run=$(mktemp -d "$PROJECT_ROOT/media/v1_support_check_20261024/run_$(date +%Y%m%d_%H%M%S)_XXXX")
printf '%s\n' "$support_run"
timeout --signal=TERM --kill-after=10s 600s python -u tools/check_lz_q1l_v1_support.py \
    --out "$support_run/result" --task lz_q1l_flat_v1_support --seed 1 --headless \
    --sim_device cpu --rl_device cpu --pipeline cpu --num_threads 2 > "$support_run/run.log" 2>&1
printf 'COMPLETE %s\n' "$support_run"
