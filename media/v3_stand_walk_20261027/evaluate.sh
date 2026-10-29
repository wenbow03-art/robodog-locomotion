#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/tools:$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python"
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | awk -F, '$2+0 > best {best=$2+0; uuid=$1} END {gsub(/ /,"",uuid); if (best<2048) exit 1; print uuid}')
export CUDA_VISIBLE_DEVICES
python -u media/v3_stand_walk_20261027/evaluate.py --task lz_q1l_flat_stand_walk --seed 1 --headless --sim_device cuda:0 --rl_device cuda:0 > media/v3_stand_walk_20261027/evaluation.log 2>&1
