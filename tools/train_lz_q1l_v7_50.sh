#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh

RUN_TAG="v7_ft_v6_50_50iters_$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="$PROJECT_ROOT/HIMLoco/legged_gym/logs1/train_${RUN_TAG}.log"
RECORD_DIR="$PROJECT_ROOT/HIMLoco/legged_gym/logs1/v7_run_records/$RUN_TAG"
INIT_MODEL="$PROJECT_ROOT/HIMLoco/legged_gym/logs1/flat_lz_q1l_v6/Oct13_10-17-19_v6_ft_v2_100_50iters_20261013_101617/model_50.pt"
test -s "$INIT_MODEL"
mkdir -p "$RECORD_DIR"
cp tools/train_lz_q1l.py tools/train_lz_q1l_v7_50.sh "$RECORD_DIR/"
cp HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py "$RECORD_DIR/"
cp HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py "$RECORD_DIR/"
cp HIMLoco/legged_gym/legged_gym/envs/__init__.py "$RECORD_DIR/envs_init.py"
cp HIMLoco/legged_gym/legged_gym/envs/base/base_config.py "$RECORD_DIR/"
cp HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py "$RECORD_DIR/"
sha256sum "$INIT_MODEL" > "$RECORD_DIR/init_model.sha256"
printf '%s\n' "$RUN_TAG" > "$RECORD_DIR/run_name.txt"
printf '%s\n' "$TRAIN_LOG" > "$RECORD_DIR/train_log_path.txt"
printf '%s\n' "Training v7 on GPU 0; init=$INIT_MODEL; iterations=50; log=$TRAIN_LOG"
set +e
python -u tools/train_lz_q1l.py \
  --task lz_q1l_flat_v7 \
  --init_from "$INIT_MODEL" \
  --max_iterations 50 \
  --run_name "$RUN_TAG" \
  --sim_device cuda:0 --rl_device cuda:0 \
  2>&1 | tee "$TRAIN_LOG"
PIPE_RESULTS=("${PIPESTATUS[@]}")
TRAIN_STATUS=${PIPE_RESULTS[0]}
set -e
printf '%s\n' "$TRAIN_STATUS" > "$RECORD_DIR/exit_code.txt"
if [ "$TRAIN_STATUS" -ne 0 ]; then
  exit "$TRAIN_STATUS"
fi
if [ "${PIPE_RESULTS[1]}" -ne 0 ]; then
  exit "${PIPE_RESULTS[1]}"
fi
SAVED=0
for RUN_DIR in "$PROJECT_ROOT/HIMLoco/legged_gym/logs1/flat_lz_q1l_v7/"*"_$RUN_TAG"; do
  if [ -d "$RUN_DIR" ]; then
    test -s "$RUN_DIR/model_50.pt"
    cp "$RUN_DIR/model_50.pt" "$RUN_DIR/v7_model50.pt"
    cp -r "$RECORD_DIR" "$RUN_DIR/provenance"
    cp "$TRAIN_LOG" "$RUN_DIR/train.log"
    sha256sum "$RUN_DIR/model_50.pt" "$RUN_DIR/v7_model50.pt" > "$RUN_DIR/checkpoints.sha256"
    printf 'Completed: %s\n' "$RUN_DIR/v7_model50.pt"
    printf '%s\n' "$RUN_DIR/v7_model50.pt" > "$RECORD_DIR/completed_model_path.txt"
    SAVED=1
  fi
done
test "$SAVED" -eq 1
