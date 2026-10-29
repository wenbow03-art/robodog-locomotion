#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
read -r RECORD_GRAPHICS_DEVICE RECORD_GPU_UUID < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits | awk -F, '$3+0 > best {best=$3+0; gpu_index=$1; uuid=$2} END {gsub(/ /,"",uuid); if (best<2048) exit 1; print gpu_index,uuid}')
# Gym camera setup uses CUDA and graphics ordinals together. Keep the physical
# GPU order visible and select the same ordinal for simulation and rendering.
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)
export RECORD_GRAPHICS_DEVICE RECORD_GPU_UUID CUDA_VISIBLE_DEVICES
record_run=$(mktemp -d "$PROJECT_ROOT/media/v3_stand_walk_20261027/run_$(date +%Y%m%d_%H%M%S)_XXXX")
printf '%s\n' "$record_run" > "$PROJECT_ROOT/media/v3_stand_walk_20261027/latest_run.txt"
printf 'RUN_DIR %s\nGPU %s\nGRAPHICS %s\n' "$record_run" "$RECORD_GPU_UUID" "$RECORD_GRAPHICS_DEVICE"
timeout --signal=TERM --kill-after=15s 900s python -u media/v3_stand_walk_20261027/record.py \
 --out "$record_run/result" --task lz_q1l_flat_stand_walk --seed 1 --num_envs 1 --headless \
 --sim_device "cuda:$RECORD_GRAPHICS_DEVICE" --rl_device "cuda:$RECORD_GRAPHICS_DEVICE" > "$record_run/record.log" 2>&1
printf 'COMPLETE %s\n' "$record_run"
python media/v3_stand_walk_20261027/publish.py
