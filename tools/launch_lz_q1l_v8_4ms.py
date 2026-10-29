"""Start the user-authorized 200-update 4ms adaptation once, detached."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'media/v8_4ms_20261028'
checks=json.loads((folder/'preflight_01/checks.json').read_text())
assert checks['passed'] and checks['finite'] and checks['original_rewards']
for name,sha in checks['code_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
session='lz_q1l_v8_4ms_200'
if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
    raise RuntimeError('Cadence training session already exists')
data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
gpus=[(int(parts[2]),int(parts[0]),parts[1].strip()) for parts in (line.split(',') for line in data.strip().splitlines())]
free,index,uuid=max(gpus,key=lambda x:x[0])
if free<12288:raise RuntimeError('Need at least12GiB free GPU memory')
name='v8_4ms_200_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
run=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v8_4ms'/name
log=folder/(name+'.log')
metadata=dict(state='spawned_verify_progress',task='lz_q1l_flat_v8_4ms',target_updates=200,
    save_interval=50,run_dir=str(run),log=str(log),session=session,
    gpu_index=index,gpu_uuid=uuid,free_mib_at_launch=free,physics_dt=.004,decimation=5,
    baseline_unchanged=True,preflight=str(folder/'preflight_01/checks.json'))
cmd=shlex.join(['bash',str(ROOT/'tools/run_lz_q1l_v8_4ms.sh'),str(run),uuid,str(log)])
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),cmd],check=True)
(folder/'latest_training.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
