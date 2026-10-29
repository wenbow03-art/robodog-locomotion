"""Five independent v3 runs; one own training job per GPU, persistent status."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from train_lz_q1l_v3_rear_teacher import ROOT, SOURCE, SOURCE_SHA256, REVIEW, preflight
from train_lz_q1l_v1_support_500 import write_json

COEFFICIENTS = [0., .01, .02, .03, .04]
SESSION = 'lz_q1l_v3_sweep_300'
POINTER = ROOT/'media/v3_sweep_20261026/latest_training.json'
LOCK = ROOT/'.cache/v3_sweep_worker.lock'


def gpu_snapshot():
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True)
    devices=[]
    for row in output.strip().splitlines():
        index,uuid,free,util=[v.strip() for v in row.split(',')]
        devices.append(dict(index=int(index),uuid=uuid,free_mib=int(free),utilization=int(util)))
    return devices


def check_sources(meta):
    for path,digest in meta['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path


def worker(manifest):
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta=json.loads(manifest.read_text()); running={}
        def persist():
            meta.update(checked_at=datetime.datetime.now().isoformat(),worker_pid=os.getpid())
            write_json(manifest,meta);write_json(POINTER,meta)
        try:
            preflight(); check_sources(meta)
            assert all(j['state']=='pending' for j in meta['jobs']), 'Worker replay is not an automatic retry'
            meta['state']='scheduling';persist()
            while any(j['state'] in ('pending','running') for j in meta['jobs']):
                for job_id,(process,log) in list(running.items()):
                    job=meta['jobs'][job_id];run=Path(job['run_dir'])
                    status_file=run/'status.json'
                    status=json.loads(status_file.read_text()) if status_file.exists() else {}
                    job['completed_updates']=status.get('completed_updates',0)
                    code=process.poll()
                    if code is None: continue
                    log.close(); del running[job_id]
                    Path(str(run)+'.exit_code').write_text(str(code)+'\n')
                    job.update(exit_code=code,finished_at=datetime.datetime.now().isoformat())
                    if code==0 and status.get('state')=='completed' and status.get('completed_updates')==meta['iterations'] and (run/('model_%d.pt'%meta['iterations'])).exists():
                        job['state']='completed'
                        job['checkpoint_sha256']=hashlib.sha256((run/('model_%d.pt'%meta['iterations'])).read_bytes()).hexdigest()
                    elif (code!=0 and status.get('completed_updates')==0 and 'CUDA out of memory' in status.get('error','')
                          and not any(run.glob('model_*.pt')) and job['attempt']<3):
                        job.setdefault('failed_attempts',[]).append(dict(run_dir=str(run),gpu=job['gpu'],error=status.get('error'),exit_code=code))
                        job.update(state='pending',attempt=job['attempt']+1,not_before=time.time()+30,
                                   minimum_free_mib=min(16384,job['minimum_free_mib']+1024))
                    else:
                        job.update(state='failed',error=status.get('error','Missing completed status/checkpoint; inspect training log'))
                devices=gpu_snapshot();meta['gpu_snapshot']=devices
                used={meta['jobs'][i]['gpu']['uuid'] for i in running}
                for job_id,job in enumerate(meta['jobs']):
                    if job['state']!='pending' or time.time()<job.get('not_before',0):continue
                    available=[d for d in devices if d['uuid'] not in used and d['free_mib']>=job['minimum_free_mib']]
                    if not available:continue
                    check_sources(meta)
                    selected=max(available,key=lambda d:d['free_mib']);used.add(selected['uuid'])
                    run=Path(job['base_run_dir']+('_retry%d'%job['attempt'] if job['attempt'] else ''))
                    command=[sys.executable,'-u',str(ROOT/'tools/train_lz_q1l_v3_rear_teacher.py'),
                             '--iterations',str(meta['iterations']),'--gpu',selected['uuid'],'--run-dir',str(run),
                             '--other-legs-coef',str(job['other_legs_coef'])]
                    log=Path(str(run)+'.log').open('x')
                    try:process=subprocess.Popen(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                    except BaseException:log.close();raise
                    running[job_id]=(process,log)
                    job.update(state='running',run_dir=str(run),launch_log=str(run)+'.log',gpu=selected,
                               training_pid=process.pid,command=command,completed_updates=0,
                               started_at=datetime.datetime.now().isoformat())
                    print('PROCESS_STARTED',job_id,job['other_legs_coef'],selected['index'],process.pid,flush=True)
                    persist()
                persist()
                if running or any(j['state']=='pending' for j in meta['jobs']):time.sleep(15)
            meta['state']='completed' if all(j['state']=='completed' for j in meta['jobs']) else 'partial_failure'
            persist()
            if any(j['state']=='completed' for j in meta['jobs']):
                check_sources(meta)
                meta['evaluation_state']='running';persist()
                with (manifest.parent/'evaluation.log').open('x') as log:
                    result=subprocess.run([sys.executable,'-u',str(ROOT/'tools/eval_lz_q1l_v3_sweep.py'),str(manifest)],
                                          cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                meta.update(evaluation_state='completed' if result.returncode==0 else 'failed',
                            evaluation_exit_code=result.returncode,evaluation_dir=str(manifest.parent/'evaluation'))
                persist()
        except BaseException as exc:
            meta.update(state='scheduler_failed',error=repr(exc));persist();raise
        finally:
            for _,log in running.values():log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--iterations',type=int,choices=[300])
    args=parser.parse_args()
    if args.worker:return worker(args.worker)
    if args.iterations is None:parser.error('Specify the user-authorized --iterations 300')
    preflight();devices=gpu_snapshot()
    LOCK.parent.mkdir(exist_ok=True);POINTER.parent.mkdir(parents=True,exist_ok=True)
    with (LOCK.parent/'v3_sweep_launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with LOCK.open('a') as active:fcntl.flock(active,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            raise RuntimeError('V3 sweep session already exists')
        tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        directory=POINTER.parent/('sweep_'+tag);directory.mkdir(exist_ok=False)
        logroot=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v3_rear_teacher';logroot.mkdir(parents=True,exist_ok=True)
        manifest=directory/'manifest.json'
        meta=dict(state='queued',session=SESSION,manifest=str(manifest),iterations=args.iterations,
                  source_checkpoint=str(SOURCE),teacher_frozen=True,
                  rear_teacher_coef=.1,coefficients=COEFFICIENTS,save_interval=10,seed=1,num_envs=4096,
                  rollout_steps=100,auto_stop=False,one_own_job_per_gpu=True,gpu_snapshot=devices,
                  source_sha256=json.loads((REVIEW/'report.json').read_text())['source_sha256'],jobs=[])
        # Keep the checkpoint digest and code digest mapping under distinct names.
        meta['checkpoint_sha256']=SOURCE_SHA256
        for i,coefficient in enumerate(COEFFICIENTS):
            run=logroot/('v3_other%02d_300_%s'%(round(coefficient*100),tag))
            meta['jobs'].append(dict(id=i,other_legs_coef=coefficient,state='pending',attempt=0,
                                     minimum_free_mib=12288,base_run_dir=str(run),completed_updates=0))
        write_json(manifest,meta);write_json(POINTER,meta)
        command=['bash',str(ROOT/'tools/start_lz_q1l_v3_sweep.sh'),'--worker',str(manifest)]
        shell=shlex.join(command)+' > '+shlex.quote(str(directory/'scheduler.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',SESSION,shell],check=True)
        print(json.dumps(dict(session=SESSION,manifest=str(manifest),pointer=str(POINTER),jobs=meta['jobs']),indent=2),flush=True)


if __name__=='__main__':main()
