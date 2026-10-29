#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-13617fc8-1298-8d94-cbe1-5afdcb1d341c
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
gpu_usage=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( gpu_usage >= 500 )); then
    echo 'GPU 2 is no longer idle; refusing to start.' >&2
    exit 1
fi
mkdir -p "$PROJECT_ROOT/HIMLoco/legged_gym/logs1/lz_q1l_noise_ablation"
trial_dir=$(mktemp -d "$PROJECT_ROOT/HIMLoco/legged_gym/logs1/lz_q1l_noise_ablation/pilot_$(date +%Y%m%d_%H%M%S)_XXXX")
echo "TRIAL_DIR=$trial_dir"
for arm in fixed annealed; do
    mkdir "$trial_dir/$arm"
    timeout --signal=TERM --kill-after=30s 2400s python -u tools/train_lz_q1l_noise_ablation.py \
        --arm "$arm" --run-dir "$trial_dir/$arm" 2>&1 | tee "$trial_dir/$arm/run.log"
done
python tools/summarize_lz_q1l_noise_ablation.py "$trial_dir"
