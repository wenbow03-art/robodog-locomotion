"""Detached user-authorized 300-update RH-only run, with preserved launch state."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from train_lz_q1l_v3_right_rear_only import ROOT, SOURCE, SOURCE_SHA256, REVIEW, preflight
from train_lz_q1l_v1_support_500 import write_json

SESSION='lz_q1l_v3_right_rear_only_300'
POINTER=ROOT/'media/v3_right_rear_only_20261027/latest_training.json'
LOCK=ROOT/'.cache/v3_right_rear_only_300.lock'


def devices():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True)
    result=[]
    for row in raw.strip().splitlines():
        index,uuid,free,util=[x.strip() for x in row.split(',')]
        result.append(dict(index=int(index),uuid=uuid,free_mib=int(free),utilization=int(util)))
    return result


def worker(path):
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta=json.loads(path.read_text())
        def persist(**changes):
            meta.update(changes,checked_at=datetime.datetime.now().isoformat(),worker_pid=os.getpid())
            write_json(path,meta);write_json(POINTER,meta)
        try:
            preflight()
            while True:
                snapshot=devices();gpu=max(snapshot,key=lambda d:d['free_mib'])
                persist(state='waiting_for_gpu',gpu_snapshot=snapshot)
                if gpu['free_mib']>=12288:break
                time.sleep(30)
            for source,digest in meta['code_sha256'].items():
                assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest,source
            preflight();run=Path(meta['run_dir'])
            command=[sys.executable,'-u',str(ROOT/'tools/train_lz_q1l_v3_right_rear_only.py'),
                     '--iterations','300','--gpu',gpu['uuid'],'--run-dir',str(run)]
            with Path(meta['launch_log']).open('x') as log:
                process=subprocess.Popen(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                persist(state='training_process_started',gpu=gpu,training_pid=process.pid,command=command)
                print('PROCESS_STARTED',process.pid,gpu,flush=True)
                while process.poll() is None:
                    status=run/'status.json'
                    if status.exists():
                        progress=json.loads(status.read_text())
                        persist(completed_updates=progress.get('completed_updates',0),training_state=progress.get('state'))
                    time.sleep(10)
                code=process.returncode
            Path(str(run)+'.exit_code').write_text(str(code)+'\n')
            progress=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
            persist(state='completed' if code==0 else 'failed',exit_code=code,
                    completed_updates=progress.get('completed_updates',0),training_state=progress.get('state'))
            if code==0:
                # Read-only final audit: this does not terminate training on quality.
                import torch
                original=torch.load(SOURCE,map_location='cpu')['model_state_dict']
                final_path=run/'model_300.pt';final=torch.load(final_path,map_location='cpu')
                frozen=[k for k in original if k.startswith(('actor.','estimator.')) or k=='std']
                unchanged=all(torch.equal(original[k],final['model_state_dict'][k]) for k in frozen)
                audit=dict(frozen_policy_bitwise_unchanged=unchanged,checked_tensors=len(frozen),
                           iteration=final['iter'],checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
                           baseline_sha256=SOURCE_SHA256)
                write_json(run/'frozen_policy_audit.json',audit)
                persist(frozen_policy_audit=audit)
        except BaseException as exc:
            persist(state='launcher_failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:return worker(args.worker)
    preflight()
    spec=json.dumps(json.loads((ROOT/'.aris/compute/local-env-spec.json').read_text()),sort_keys=True,separators=(',',':'))
    assert hashlib.sha256(spec.encode()).hexdigest()[:8]=='fbc8f0c4'
    snapshot=devices();LOCK.parent.mkdir(exist_ok=True);POINTER.parent.mkdir(parents=True,exist_ok=True)
    with (LOCK.parent/'v3_right_rear_only_launch.lock').open('a') as launch:
        fcntl.flock(launch,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with LOCK.open('a') as active:fcntl.flock(active,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            raise RuntimeError('RH-only training session already exists')
        tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        directory=POINTER.parent/('run_'+tag);directory.mkdir(exist_ok=False)
        run=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v3_right_rear_only'/('rh_only_from_gait150_300_'+tag)
        run.parent.mkdir(parents=True,exist_ok=True)
        sources=json.loads((REVIEW/'report.json').read_text())['source_sha256']
        for source in [Path(__file__).resolve(),ROOT/'tools/start_lz_q1l_v3_right_rear_only_300.sh']:
            sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
            shutil.copyfile(source,directory/source.name)
        meta=dict(state='queued',session=SESSION,run_dir=str(run),launch_log=str(run)+'.log',
                  manifest=str(directory/'manifest.json'),source_checkpoint=str(SOURCE),source_sha256=SOURCE_SHA256,
                  target_updates=300,completed_updates=0,save_interval=10,seed=1,num_envs=4096,
                  rollout_steps=100,rear_teacher_coef=.10,other_legs_frozen=True,
                  auto_stop=False,minimum_free_mib=12288,gpu_snapshot=snapshot,code_sha256=sources)
        path=Path(meta['manifest']);write_json(path,meta);write_json(POINTER,meta)
        command=['bash',str(ROOT/'tools/start_lz_q1l_v3_right_rear_only_300.sh'),'--worker',str(path)]
        subprocess.run(['tmux','new-session','-d','-s',SESSION,
                        shlex.join(command)+' > '+shlex.quote(str(directory/'launcher.log'))+' 2>&1'],check=True)
        print(json.dumps(dict(session=SESSION,pointer=str(POINTER),run_dir=str(run)),indent=2),flush=True)


if __name__=='__main__':main()
