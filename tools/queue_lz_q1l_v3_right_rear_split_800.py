"""Wait for the active600 run, then add200: total500 new updates from RH-only300."""
import argparse,datetime,fcntl,hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
from train_lz_q1l_v2_unified import ROOT
from train_lz_q1l_v1_support_500 import write_json
from train_lz_q1l_v3_right_rear_split import SOURCE,SOURCE_SHA256
from launch_lz_q1l_v3_right_rear_split_600 import devices
SESSION='lz_q1l_v3_right_rear_split_to800'
POINTER=ROOT/'media/v3_right_rear_split_20261027/continuation_800.json'
LOCK=ROOT/'.cache/v3_right_rear_split_to800.lock'
ACTIVE_LOCK=ROOT/'.cache/v3_right_rear_only_300.lock'
SOURCE_RUN=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v3_right_rear_split/rh_split_from300_to600_20261027_171824_1957397'
NEW_FILES=['tools/queue_lz_q1l_v3_right_rear_split_800.py','tools/start_lz_q1l_v3_right_rear_split_800.sh','tools/train_lz_q1l_v3_right_rear_split_extend.py']

def verify_sources(meta):
    for name,digest in meta['code_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name

def worker(path):
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta=json.loads(path.read_text())
        def persist(**kwargs):
            meta.update(kwargs,checked_at=datetime.datetime.now().isoformat(),worker_pid=os.getpid())
            write_json(path,meta);write_json(POINTER,meta)
        try:
            verify_sources(meta)
            while True:
                status=json.loads((SOURCE_RUN/'status.json').read_text());completed=status.get('completed_updates',300)
                persist(state='waiting_for_model600',completed_updates=completed,additional_completed=completed-300,active_run_dir=str(SOURCE_RUN),upstream_state=status['state'])
                if status['state'] in ['failed','stopped_by_user']:raise RuntimeError('Upstream did not complete; extension will not restart it')
                audit_path=SOURCE_RUN/'frozen_policy_audit.json';exit_file=Path(str(SOURCE_RUN)+'.exit_code')
                if status['state']=='completed' and completed==600 and audit_path.exists() and exit_file.exists():
                    assert exit_file.read_text().strip()=='0'
                    audit=json.loads(audit_path.read_text());assert audit['frozen_policy_bitwise_unchanged'] and audit['iteration']==600
                    resume=SOURCE_RUN/'model_600.pt';digest=hashlib.sha256(resume.read_bytes()).hexdigest();assert digest==audit['checkpoint_sha256']
                    break
                time.sleep(10)
            with ACTIVE_LOCK.open('a') as active:
                while True:
                    try:fcntl.flock(active,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:time.sleep(10)
                verify_sources(meta)
                while True:
                    snapshot=devices();gpu=max(snapshot,key=lambda d:d['free_mib'])
                    persist(state='waiting_for_gpu',gpu_snapshot=snapshot,source_checkpoint=str(resume),source_sha256=digest)
                    if gpu['free_mib']>=12288:break
                    time.sleep(30)
                run=Path(meta['run_dir'])
                command=[sys.executable,'-u',str(ROOT/'tools/train_lz_q1l_v3_right_rear_split_extend.py'),'--resume',str(resume),'--resume-sha256',digest,'--iterations','200','--gpu',gpu['uuid'],'--run-dir',str(run)]
                with Path(meta['launch_log']).open('x') as log:
                    process=subprocess.Popen(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                    persist(state='training_process_started',training_pid=process.pid,gpu=gpu,command=command,active_run_dir=str(run))
                    while process.poll() is None:
                        status_file=run/'status.json'
                        if status_file.exists():
                            status=json.loads(status_file.read_text());completed=status.get('completed_updates',600)
                            persist(completed_updates=completed,additional_completed=completed-300,training_state=status['state'])
                        time.sleep(10)
                    code=process.returncode
                Path(str(run)+'.exit_code').write_text(str(code)+'\n')
                if code!=0:raise RuntimeError('Extension training exited with code %s'%code)
                status=json.loads((run/'status.json').read_text());assert status['state']=='completed' and status['completed_updates']==800
                import torch
                base=torch.load(SOURCE,map_location='cpu')['model_state_dict'];final_path=run/'model_800.pt';final=torch.load(final_path,map_location='cpu')
                frozen=[k for k in base if k=='std' or k.startswith(('actor.','estimator.'))]
                audit=dict(frozen_policy_bitwise_unchanged=all(torch.equal(base[k],final['model_state_dict'][k]) for k in frozen),checked_tensors=len(frozen),iteration=final['iter'],checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),baseline_sha256=SOURCE_SHA256)
                write_json(run/'frozen_policy_audit.json',audit)
                persist(state='completed',completed_updates=800,additional_completed=500,training_state='completed',frozen_policy_audit=audit)
        except BaseException as exc:
            persist(state='failed',error=repr(exc));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path);args=p.parse_args()
    if args.worker:return worker(args.worker)
    parent=json.loads((SOURCE_RUN/'run_metadata.json').read_text());assert parent['distillation']['rear_teacher_coef']==.1 and parent['distillation']['rear_teacher_forward_coef']==.3
    sources={str(ROOT/name):digest for name,digest in parent['code_sha256'].items()}
    for name in NEW_FILES:sources[str(ROOT/name)]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    verify_sources(dict(code_sha256=sources))
    with (ROOT/'.cache/v3_right_rear_split800_launch.lock').open('a') as launch:
        fcntl.flock(launch,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with LOCK.open('a') as active:fcntl.flock(active,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise RuntimeError('Extension queue already running')
        if POINTER.exists() and json.loads(POINTER.read_text())['state']=='completed':raise RuntimeError('Authorized extension already completed')
        tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid());directory=POINTER.parent/('extension_'+tag);directory.mkdir(exist_ok=False)
        run=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v3_right_rear_split'/('rh_split_from600_to800_'+tag)
        status=json.loads((SOURCE_RUN/'status.json').read_text())
        meta=dict(state='queued',session=SESSION,source_run_dir=str(SOURCE_RUN),active_run_dir=str(SOURCE_RUN),run_dir=str(run),launch_log=str(run)+'.log',manifest=str(directory/'manifest.json'),target_updates=800,completed_updates=status.get('completed_updates',300),additional_target=500,initial_iteration=300,queued_additional_iterations=200,save_interval=10,stand_coefficient=.1,forward_coefficient=.3,auto_stop=False,code_sha256=sources)
        for name in NEW_FILES:shutil.copyfile(ROOT/name,directory/Path(name).name)
        path=Path(meta['manifest']);write_json(path,meta);write_json(POINTER,meta)
        command=['bash',str(ROOT/'tools/start_lz_q1l_v3_right_rear_split_800.sh'),'--worker',str(path)]
        subprocess.run(['tmux','new-session','-d','-s',SESSION,shlex.join(command)+' > '+shlex.quote(str(directory/'queue.log'))+' 2>&1'],check=True)
        print(json.dumps(dict(session=SESSION,overall_pointer=str(POINTER),target_updates=800,additional_target=500),indent=2))
if __name__=='__main__':main()
