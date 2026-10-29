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
    echo 'GPU 2 is busy; recording not started.' >&2
    exit 1
fi
mkdir -p media/lz_q1l_lateral_videos
export LATERAL_RUN_DIR
LATERAL_RUN_DIR=$(mktemp -d "$PROJECT_ROOT/media/lz_q1l_lateral_videos/run_$(date +%Y%m%d_%H%M%S)_XXXX")
mkdir "$LATERAL_RUN_DIR/public" "$LATERAL_RUN_DIR/provenance"
python - <<'PY'
import hashlib,json,os,shutil
from pathlib import Path
root=Path(os.environ['PROJECT_ROOT']); out=Path(os.environ['LATERAL_RUN_DIR'])
sources=['tools/record_lz_q1l_lateral.py','tools/run_lz_q1l_lateral_videos.sh','tools/publish_lz_q1l_lateral_videos.py',
 'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_mirror.py',
 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py',
 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
 'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
 'HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py']
hashes={}
for name in sources:
 p=root/name; hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 shutil.copyfile(p,out/'provenance'/p.name)
checkpoint=root/'HIMLoco/legged_gym/logs1/flat_lz_q1l_v8/v8_ft_v7_200_200iters_20261018_130426_2712609/model_200.pt'
manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_sha256s=hashes,gpu=os.environ['CUDA_VISIBLE_DEVICES'],policy='Default left-mirror; right original',seed=1)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('RUN_DIR='+str(out),flush=True)
PY
for direction in left right; do
    timeout --signal=TERM --kill-after=15s 600s python -u "$LATERAL_RUN_DIR/provenance/record_lz_q1l_lateral.py" \
        --direction "$direction" --out-dir "$LATERAL_RUN_DIR/public" \
        --task lz_q1l_flat_v8 --load_run v8_ft_v7_200_200iters_20261018_130426_2712609 \
        --checkpoint 200 --num_envs 1 --seed 1 --headless \
        --sim_device cuda:0 --rl_device cuda:0 2>&1 | tee "$LATERAL_RUN_DIR/$direction.log"
    env -u LD_LIBRARY_PATH /home/xiaohengli/miniconda3/envs/multinex/bin/ffprobe \
        -v error -select_streams v:0 -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration,size \
        -of json "$LATERAL_RUN_DIR/public/$direction.mp4" > "$LATERAL_RUN_DIR/$direction.check.json"
    env -u LD_LIBRARY_PATH /home/xiaohengli/miniconda3/envs/multinex/bin/ffmpeg \
        -v error -i "$LATERAL_RUN_DIR/public/$direction.mp4" -f null -
    env -u LD_LIBRARY_PATH /home/xiaohengli/miniconda3/envs/multinex/bin/ffmpeg \
        -v error -ss 6 -i "$LATERAL_RUN_DIR/public/$direction.mp4" -frames:v 1 "$LATERAL_RUN_DIR/public/$direction.jpg"
done
python "$LATERAL_RUN_DIR/provenance/publish_lz_q1l_lateral_videos.py" "$LATERAL_RUN_DIR"
