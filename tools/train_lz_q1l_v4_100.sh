#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh

RUN_TAG="v4_ft_v2_200_100iters_$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="$PROJECT_ROOT/HIMLoco/legged_gym/logs1/train_${RUN_TAG}.log"
RECORD_DIR="$PROJECT_ROOT/HIMLoco/legged_gym/logs1/v4_run_records/$RUN_TAG"
INIT_MODEL="$PROJECT_ROOT/HIMLoco/legged_gym/logs1/flat_lz_q1l_v2/Oct04_21-48-35_v2_ft_v1_1700/model_200.pt"
mkdir -p "$RECORD_DIR"
cp tools/train_lz_q1l.py tools/train_lz_q1l_v4_100.sh "$RECORD_DIR/"
cp HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py "$RECORD_DIR/"
cp HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py "$RECORD_DIR/"
sha256sum "$INIT_MODEL" > "$RECORD_DIR/init_model.sha256"
printf '%s\n' "$RUN_TAG" > "$RECORD_DIR/run_name.txt"
printf '%s\n' "$TRAIN_LOG" > "$RECORD_DIR/train_log_path.txt"
printf '%s\n' "Training v4 on GPU 0; init=$INIT_MODEL; iterations=100; log=$TRAIN_LOG"
set +e
python -u tools/train_lz_q1l.py \
  --task lz_q1l_flat_v4 \
  --init_from "$INIT_MODEL" \
  --max_iterations 100 \
  --run_name "$RUN_TAG" \
  --sim_device cuda:0 --rl_device cuda:0 \
  2>&1 | tee "$TRAIN_LOG"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$TRAIN_STATUS" > "$RECORD_DIR/exit_code.txt"
if [ "$TRAIN_STATUS" -eq 0 ]; then
  for RUN_DIR in "$PROJECT_ROOT/HIMLoco/legged_gym/logs1/flat_lz_q1l_v4/"*"_$RUN_TAG"; do
    if [ -d "$RUN_DIR" ]; then
      cp -r "$RECORD_DIR" "$RUN_DIR/provenance"
      cp "$TRAIN_LOG" "$RUN_DIR/train.log"
      printf 'Completed: %s\n' "$RUN_DIR/model_100.pt"
    fi
  done
fi
exit "$TRAIN_STATUS"
