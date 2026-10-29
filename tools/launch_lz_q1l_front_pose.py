"""Start the 300-update teacher-free side adaptation after explicit user authorization once, detached."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'media/front_pose_20261029'
folder.mkdir(exist_ok=True)
review_dir=ROOT/'media/front_pose_review_20261029'
review=json.loads((review_dir/'review_manifest.json').read_text())
checks=json.loads((review_dir/'gpu_checks.json').read_text())
assert all(checks[k] for k in ['passed','teacher_absent','native_actions','estimator_unchanged','actor_updated','optimizer_restored'])
for name,sha in review['code_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
selected=dict(checkpoint=review['source'],checkpoint_sha256=review['source_sha256'])
assert hashlib.sha256(Path(selected['checkpoint']).read_bytes()).hexdigest()==selected['checkpoint_sha256']
session='lz_q1l_front_pose_300'
if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
    raise RuntimeError('Native side speed training session already exists')
data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
gpus=[(int(parts[2]),int(parts[0]),parts[1].strip()) for parts in (line.split(',') for line in data.strip().splitlines())]
free,index,uuid=max(gpus,key=lambda x:x[0])
if free<12288:raise RuntimeError('Need at least12GiB free GPU memory')
name='front_pose_300_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
run=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_front_pose'/name
log=folder/(name+'.log')
metadata=dict(state='spawned_verify_progress',task='lz_q1l_flat_front_pose',target_updates=300,
    save_interval=50,run_dir=str(run),log=str(log),session=session,
    gpu_index=index,gpu_uuid=uuid,free_mib_at_launch=free,physics_dt=.001,decimation=20,
    baseline_unchanged=True,source_checkpoint=selected['checkpoint'],source_sha256=selected['checkpoint_sha256'],teacher_enabled=False,mirroring=False,estimator_frozen=True,base_height_scale=-100.,preflight=str(review_dir/'gpu_checks.json'))
cmd=shlex.join(['bash',str(ROOT/'tools/run_lz_q1l_front_pose.sh'),str(run),uuid,str(log)])
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),cmd],check=True)
(folder/'latest_training.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
