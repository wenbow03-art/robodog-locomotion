#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == '--worker' ]]; then
    exec 9> "$PROJECT_ROOT/.cache/v2_unified_800.lock"
    flock -n 9 || { echo 'New V2 training is already running.' >&2; exit 1; }
    run_dir="$2"
    gpu_uuid="$3"
    launch_log="${run_dir}.log"
    export CUDA_VISIBLE_DEVICES="$gpu_uuid"
    set +e
    python -u tools/train_lz_q1l_v2_unified.py --iterations 800 --gpu "$gpu_uuid" --run-dir "$run_dir" 2>&1 | tee "$launch_log"
    job_status=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$job_status" > "${run_dir}.exit_code"
    exit "$job_status"
fi
if [[ $# -ne 0 ]]; then
    echo 'Usage: bash tools/start_lz_q1l_v2_800.sh' >&2
    exit 2
fi

mkdir -p "$PROJECT_ROOT/.cache" "$PROJECT_ROOT/HIMLoco/legged_gym/logs2/flat_lz_q1l_v2"
python - <<'PY'
import datetime, fcntl, hashlib, json, os, shlex, subprocess
from pathlib import Path
root=Path.cwd()
with (root/'.cache/v2_unified_launch.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # The worker holds this separate lock for the entire run.
    with (root/'.cache/v2_unified_800.lock').open('a') as worker_lock:
        fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    session='lz_q1l_new_v2_800'
    if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
        raise SystemExit('New V2 tmux session already exists; refusing duplicate launch')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices=[]
    for line in rows.strip().splitlines():
        index,uuid,free,used,util=[x.strip() for x in line.split(',')]
        devices.append(dict(index=int(index),uuid=uuid,free_mib=int(free),used_mib=int(used),utilization=int(util)))
    selected=max(devices,key=lambda row:row['free_mib'])
    if selected['free_mib']<10240:
        raise SystemExit('Greatest free GPU memory is below 10 GiB; no training launched')
    source=root/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v1_support/v1_support_scratch_500_20261024_113050_2745173/model_500.pt'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='efaf1f486d0264ec8539a8c4ef50b5a556f8fe818294ca6b65b631f3b5bcb3b1'
    tag='v2_ft_newv1_500_800_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    run=root/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2'/tag
    command=['bash',str(root/'tools/start_lz_q1l_v2_800.sh'),'--worker',str(run),selected['uuid']]
    subprocess.run(['tmux','new-session','-d','-s',session,shlex.join(command)],check=True)
    metadata=dict(session=session,run_dir=str(run),launch_log=str(run)+'.log',selected_gpu=selected,gpu_snapshot=devices,updates=800,save_interval=50)
    pointer=root/'media/v2_unified_review_20261024/latest_training.json'
    pointer.write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)
PY
