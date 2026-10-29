#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaohengli/Program/robo_competition
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PROJECT_ROOT/HIMLoco/legged_gym:$PROJECT_ROOT/HIMLoco/rsl_rl:$PROJECT_ROOT/third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == '--worker' ]]; then
    exec 9> "$PROJECT_ROOT/.cache/v2_posture_100.lock"
    flock -n 9 || { echo 'Posture training already running.' >&2; exit 1; }
    run_dir="$2"
    gpu_uuid="$3"
    baseline="$4"
    resume_checkpoint="$5"
    set +e
    python -u tools/train_lz_q1l_v2_posture.py --iterations 500 --resume "$resume_checkpoint" --gpu "$gpu_uuid" --run-dir "$run_dir" --baseline "$baseline" 2>&1 | tee "${run_dir}.log"
    job_status=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$job_status" > "${run_dir}.exit_code"
    exit "$job_status"
fi
if [[ $# -ne 0 ]]; then
    echo 'Usage: bash tools/start_lz_q1l_v2_posture_500.sh' >&2
    exit 2
fi
mkdir -p "$PROJECT_ROOT/.cache" "$PROJECT_ROOT/HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_posture"
python - <<'PY'
import datetime,fcntl,hashlib,json,os,shlex,subprocess
from pathlib import Path
root=Path.cwd()
with (root/'.cache/v2_posture_launch.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (root/'.cache/v2_posture_100.lock').open('a') as worker:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
    session='lz_q1l_v2_posture_500'
    if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
        raise SystemExit('Posture tmux session already exists')
    baseline=Path(json.loads((root/'media/v2_posture_20261024/latest_preflight.json').read_text())['report'])
    report=json.loads(baseline.read_text())
    for path,digest in report['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    resume=root/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_posture/posture_from_newv1_500_100_20261024_223220_3056008/model_20.pt'
    assert resume.is_file()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices=[]
    for line in rows.strip().splitlines():
        index,uuid,free,util=[x.strip() for x in line.split(',')]
        devices.append(dict(index=int(index),uuid=uuid,free_mib=int(free),utilization=int(util)))
    selected=max(devices,key=lambda x:x['free_mib'])
    if selected['free_mib']<9216:raise SystemExit('Less than 9 GiB free; no launch')
    tag='posture_resume20_to500_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    run=root/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_posture'/tag
    command=['bash',str(root/'tools/start_lz_q1l_v2_posture_500.sh'),'--worker',str(run),selected['uuid'],str(baseline),str(resume)]
    meta=dict(run_dir=str(run),launch_log=str(run)+'.log',session=session,selected_gpu=selected,gpu_snapshot=devices,
        baseline=str(baseline),resume_checkpoint=str(resume),resume_sha256=hashlib.sha256(resume.read_bytes()).hexdigest(),
        start_iteration=20,updates=500,additional_iterations=480,save_interval=10,posture_check_interval=20,auto_stop=False)
    # Persist the launch intent before starting the detached worker.
    Path(str(run)+'.launch.json').write_text(json.dumps(meta,indent=2)+'\n')
    subprocess.run(['tmux','new-session','-d','-s',session,shlex.join(command)],check=True)
    pointer=root/'media/v2_posture_20261024/latest_training.json'
    if pointer.exists():
        Path(str(run)+'.previous_pointer.json').write_text(pointer.read_text())
    pointer.write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)
PY
