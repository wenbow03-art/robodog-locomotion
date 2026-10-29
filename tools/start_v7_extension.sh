#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export CUDA_VISIBLE_DEVICES=GPU-396d9087-baa8-80e1-194c-0726ad31b9b3
V7_RUN_DIR="$PROJECT_ROOT/HIMLoco/legged_gym/logs1/flat_lz_q1l_v7/Oct13_11-20-32_v7_ft_v2_200_200iters_20261013_111930"
mkdir -p "$V7_RUN_DIR/extension_provenance"
cp tools/extend_v7_to_500.py tools/start_v7_extension.sh "$V7_RUN_DIR/extension_provenance/"
python -u "$V7_RUN_DIR/extension_provenance/extend_v7_to_500.py" --run-dir "$V7_RUN_DIR" 2>&1 | tee -a "$V7_RUN_DIR/extension_train.log"
