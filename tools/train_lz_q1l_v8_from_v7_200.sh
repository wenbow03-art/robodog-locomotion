#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export PYTHONPATH="$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python:${PYTHONPATH:-}"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$MPLCONFIGDIR" /home/xiaohengli/Program/training-curves

# An explicit first argument selects a GPU index or UUID. Otherwise select the
# GPU with most free memory, requiring at least 10 GiB; existing jobs are shared.
if [[ $# -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="$1"
else
    gpu_rows="$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits)"
    selected_gpu="$(printf '%s\n' "$gpu_rows" | awk -F, '$2+0 >= 10240 && $2+0 > best {best=$2+0; gpu=$1} END {if (gpu) print gpu}')"
    if [[ -z "$selected_gpu" ]]; then
        echo 'No GPU has 10 GiB free; training has not started.' >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$selected_gpu"
fi
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable in this execution environment; training has not started.')
torch.manual_seed(1)
x = torch.randn(8, 8, device='cuda')
y = x @ x.T
assert torch.isfinite(y).all()
print('CUDA_WITNESS', torch.cuda.get_device_name(0), tuple(y.shape), flush=True)
PY

run_tag="v8_ft_v7_200_200iters_$(date +%Y%m%d_%H%M%S)_$$"
run_dir="$PROJECT_ROOT/HIMLoco/legged_gym/logs1/flat_lz_q1l_v8/$run_tag"
mkdir -p "$run_dir/provenance"
for relative in \
    tools/train_lz_q1l_v8_from_v7_200.py \
    tools/train_lz_q1l_v8_from_v7_200.sh \
    tools/export_v8_training_curves.py \
    HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py \
    HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py \
    HIMLoco/legged_gym/legged_gym/envs/__init__.py \
    HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py \
    HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py \
    HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py \
    HIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py; do
    mkdir -p "$run_dir/provenance/$(dirname "$relative")"
    cp "$relative" "$run_dir/provenance/$relative"
    sha256sum "$relative" >> "$run_dir/source_checksums.sha256"
done
printf 'RUN_DIR %s\nGPU %s\n' "$run_dir" "$CUDA_VISIBLE_DEVICES"
set +e
python -u tools/train_lz_q1l_v8_from_v7_200.py \
    --run-dir "$run_dir" \
    --curves-dir /home/xiaohengli/Program/training-curves \
    2>&1 | tee "$run_dir/train.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${pipeline_status[0]}" > "$run_dir/exit_code.txt"
if [[ "${pipeline_status[0]}" -ne 0 ]]; then
    exit "${pipeline_status[0]}"
fi
exit "${pipeline_status[1]}"
