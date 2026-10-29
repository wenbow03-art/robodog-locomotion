#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# User preference: select the GPU with the greatest free memory, even if busy.
if [[ "${1:-}" == '--worker' ]]; then
    export CUDA_VISIBLE_DEVICES="$3"
else
    gpu_rows=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits)
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(printf '%s\n' "$gpu_rows" | awk -F, '$2+0 > best {best=$2+0; gpu=$1} END {print gpu}')
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"

check_gpu() {
    local free_mb
    free_mb=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( free_mb < 10240 )); then
        echo 'The GPU with most free memory has less than 10 GiB free; not started.' >&2
        exit 1
    fi
}

if [[ "${1:-}" == '--preflight' ]]; then
    check_gpu
    exec python -u tools/train_lz_q1l_v1_support_500.py --preflight
elif [[ "${1:-}" == '--worker' ]]; then
    run_dir="$2"
    exec 9> "$PROJECT_ROOT/.cache/v1_support_training.lock"
    flock -n 9 || { echo 'A V1 support training job is already running.'; exit 1; }
    check_gpu
    set +e
    python -u tools/train_lz_q1l_v1_support_500.py --run-dir "$run_dir" 2>&1 | tee "$run_dir/train.log"
    training_status=("${PIPESTATUS[@]}")
    set -e
    printf '%s\n' "${training_status[0]}" > "$run_dir/exit_code.txt"
    exit "${training_status[0]}"
elif [[ $# -gt 0 ]]; then
    echo 'Usage: bash tools/train_lz_q1l_v1_support_500.sh [--preflight]' >&2
    exit 2
fi

check_gpu
run_tag="v1_support_scratch_500_$(date +%Y%m%d_%H%M%S)_$$"
run_dir="$PROJECT_ROOT/HIMLoco/legged_gym/logs2/flat_lz_q1l_v1_support/$run_tag"
mkdir -p "$run_dir"
session="lz_v1_support_$(date +%Y%m%d_%H%M%S)_$$"
tmux new-session -d -s "$session" bash "$PROJECT_ROOT/tools/train_lz_q1l_v1_support_500.sh" --worker "$run_dir" "$CUDA_VISIBLE_DEVICES"
printf '%s\n' "$session" > "$run_dir/tmux_session.txt"
printf '%s\n' "$run_dir" > "$PROJECT_ROOT/media/v1_support_check_20261024/training_run.txt"
printf 'RUN_DIR %s\nTMUX %s\nGPU %s\n' "$run_dir" "$session" "$CUDA_VISIBLE_DEVICES"
