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
    echo "GPU 2 is no longer idle; refusing to start." >&2
    exit 1
fi
mkdir -p media/lz_q1l_mirror
run_dir=$(mktemp -d "media/lz_q1l_mirror/eval_$(date +%Y%m%d_%H%M%S)_XXXX")
echo "RUN_DIR=$run_dir"
python tools/test_lz_q1l_mirror.py 2>&1 | tee "$run_dir/tests.log"
timeout --signal=TERM --kill-after=30s 600s python -u tools/eval_lz_q1l_mirror.py --out-dir "$run_dir" "$@" 2>&1 | tee "$run_dir/run.log"
