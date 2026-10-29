"""Launch authorized direct-left/right V4 once, detached from remote connection."""
from datetime import datetime
import hashlib,json,os,shlex,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'media/v4_lateral_20261027';folder.mkdir(exist_ok=True)
check_dir=ROOT/'media/v4_lateral_review_20261027/direct_no_mirror'
checks=json.loads((check_dir/'checks.json').read_text())
for key in ['new_baseline_exact_nonlateral','direct_observations_and_actions','no_mirror_constraint','no_noise_permutation','rollout_update_lateral_means_equal','frozen_after_synthetic_optimizer_step']:
    assert checks[key],key
assert checks['simulation']['balanced_direction_sampling'] and checks['simulation']['finite']
manifest=json.loads((check_dir/'manifest.json').read_text())
for path,digest in manifest['code_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
session='lz_q1l_v4_lateral_500'
if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
    raise RuntimeError('V4 session already exists; refusing a duplicate training job')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
gpus=[(int(parts[2]),int(parts[0]),parts[1].strip()) for parts in (line.split(',') for line in rows.strip().splitlines())]
free,index,uuid=max(gpus,key=lambda x:x[0])
if free<12288:raise RuntimeError('Not enough GPU memory available to start 4096 environments')
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
run=ROOT/('HIMLoco/legged_gym/logs2/flat_lz_q1l_v4_lateral/direct_from_newbaseline_500_'+stamp+'_'+str(os.getpid()))
log=folder/(run.name+'.log')
launch=dict(state='launching',task='lz_q1l_flat_v4_lateral',target_updates=500,save_interval=50,run_dir=str(run),log=str(log),session=session,
    gpu_index=index,gpu_uuid=uuid,free_mib_at_launch=free,mirroring=False,commands=dict(stand=.2,forward=.2,left=.3,right=.3),code_sha256=manifest['code_sha256'])
(folder/'latest_training.json').write_text(json.dumps(launch,indent=2)+'\n')
cmd=shlex.join(['bash',str(ROOT/'tools/run_lz_q1l_v4_lateral_500.sh'),str(run),uuid,str(log)])
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),cmd],check=True)
launch['state']='spawned_verify_progress'
(folder/'latest_training.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch,indent=2))
