#!/usr/bin/env bash
set -uo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export CUDA_VISIBLE_DEVICES="$2"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/tools:$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python"
python -u tools/train_lz_q1l_continuous_rh.py --run-dir "$1" "${@:4}" > "$3" 2>&1
run_exit=$?
printf '%s\n' "$run_exit" > "$3.exit_code"
exit "$run_exit"
