#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-13617fc8-1298-8d94-cbe1-5afdcb1d341c
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
selected_gpu=''
candidate_gpus='GPU-13617fc8-1298-8d94-cbe1-5afdcb1d341c GPU-2e39c27b-a9a2-68f7-4cd0-0b48ee731159'
if [[ "${1:-}" == '--gpu-uuid' ]]; then
    candidate_gpus="$2"
    shift 2
fi
for candidate_gpu in $candidate_gpus; do
    gpu_free=$(nvidia-smi -i "$candidate_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    gpu_util=$(nvidia-smi -i "$candidate_gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if (( gpu_free >= 8000 && gpu_util <= 10 )); then
        selected_gpu="$candidate_gpu"
        break
    fi
done
if [[ -z "$selected_gpu" ]]; then
    echo 'Neither candidate GPU currently has spare capacity.' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$selected_gpu"
echo "Using GPU $CUDA_VISIBLE_DEVICES"
diagnostic_parent=$(mktemp -d "$PROJECT_ROOT/media/standing_support_$(date +%Y%m%d_%H%M%S)_XXXX")
timeout --signal=TERM --kill-after=15s 600s python -u tools/diagnose_standing_support.py \
    --out-dir "$diagnostic_parent/result" --task lz_q1l_flat_v8 --seed 1 \
    --headless --sim_device cuda:0 --rl_device cuda:0 "$@" 2>&1 | tee "$diagnostic_parent/run.log"
