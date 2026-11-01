"""Preflight explicitly, or launch only with an explicit --iterations budget."""
import argparse
from datetime import datetime
import json,os,shlex,subprocess
from pathlib import Path
from lz_q1l_acceleration_common import ROOT,TASK,digest,write

def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true')
    group.add_argument('--iterations',type=int)
    p.add_argument('--resume',type=Path)
    opt=p.parse_args()
    if opt.iterations is not None and opt.iterations<=0:p.error('--iterations must be positive')
    if opt.preflight and opt.resume:p.error('Cannot resume in preflight')
    folder=ROOT/'media/forward_acceleration_20261031';folder.mkdir(exist_ok=True)
    if not opt.preflight:
        review=Path(json.loads((folder/'latest_preflight.json').read_text())['run_dir'])
        assert json.loads((review/'checks.json').read_text())['passed']
        assert json.loads((review/'status.json').read_text())['state']=='completed'
        metadata=json.loads((review/'run_metadata.json').read_text())
        for f,h in metadata['code_sha256'].items():assert digest(ROOT/f)==h, 'Preflight stale: '+f
        assert digest(metadata['source_checkpoint'])==metadata['source_sha256']
        pointer=folder/'latest_training.json'
        if pointer.exists():
            old=json.loads(pointer.read_text());state=Path(old['run_dir'])/'status.json'
            if not state.exists() or json.loads(state.read_text())['state'] not in ('completed','failed'):
                raise RuntimeError('Existing run may be active; inspect before launching another')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    gpus=[(int(p[2]),int(p[0]),p[1].strip()) for p in (line.split(',') for line in rows.strip().splitlines())]
    free,index,uuid=sorted(gpus,key=lambda x:(-x[0],x[1]))[0]
    assert free>=(4096 if opt.preflight else 12288), 'Insufficient free memory'
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    name=('preflight_' if opt.preflight else 'acceleration_%d_'%opt.iterations)+stamp
    run=folder/name if opt.preflight else ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_forward_acceleration'/name
    log=folder/(name+'.log')
    command=['bash',str(ROOT/'tools/run_lz_q1l_forward_acceleration.sh'),str(run),uuid,str(log)]
    command+=['--preflight'] if opt.preflight else ['--iterations',str(opt.iterations)]
    if opt.resume:command+=['--resume',str(opt.resume.resolve())]
    info=dict(state='preflight_running' if opt.preflight else 'launched_pending_first_update',
              task=TASK,run_dir=str(run),log=str(log),gpu_index=index,gpu_uuid=uuid,
              requested_updates=opt.iterations,preflight=opt.preflight)
    if opt.preflight:
        write(folder/'latest_preflight.json',info)
        print(json.dumps(info,indent=2),flush=True)
        subprocess.run(command,cwd=str(ROOT),check=True)
        assert json.loads((run/'checks.json').read_text())['passed']
        info['state']='preflight_passed';write(folder/'latest_preflight.json',info)
        print('PREFLIGHT_PASSED',run,flush=True)
    else:
        session='lz_q1l_acceleration_'+stamp
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),shlex.join(command)],check=True)
        info['session']=session;write(folder/'latest_training.json',info)
        print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
