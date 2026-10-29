#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/tools:$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python"
if [[ "${1:-}" == "--gpu-sim" ]]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | awk -F, '$2+0 > best {best=$2+0; uuid=$1} END {gsub(/ /,"",uuid); if (best<2048) exit 1; print uuid}')
  export CUDA_VISIBLE_DEVICES
fi
python -u tools/check_lz_q1l_v4_lateral.py "$@"
