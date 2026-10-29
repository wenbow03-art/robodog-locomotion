#!/usr/bin/env bash
set -uo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/tools:$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python"
python -u tools/train_lz_q1l_v4_lateral.py --iterations 500 --gpu "$2" --run-dir "$1" > "$3" 2>&1
v4_exit_code=$?
printf '%s\n' "$v4_exit_code" > "$3.exit_code"
exit "$v4_exit_code"
