"""User-authorized 300-update continuation of the reviewed 1 ms coordination task."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT/'media/soft_slip_review_20261029'
TASK = 'lz_q1l_flat_soft_slip'
TARGET = 300


def write(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    opts=parser.parse_args();out=opts.run_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(state='initializing',completed_updates=0,target_updates=TARGET))
    env=runner=None
    try:
        review=json.loads((REVIEW/'review_manifest.json').read_text())
        for name,sha in review['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
        checks=json.loads((REVIEW/'gpu_checks.json').read_text())
        assert all(checks[k] for k in ['passed','teacher_absent','native_actions','weights_unchanged','optimizer_unchanged','estimator_frozen','exact_restore','lifecycle_passed'])
        assert json.loads((REVIEW/'synthetic_checks.json').read_text())['passed']
        source=Path(review['source']);source_sha=hashlib.sha256(source.read_bytes()).hexdigest();assert source_sha==review['source_sha256']
        baseline=ROOT/'HIMLoco/legged_gym/logs2/baselines/stand_walk800_20261027/model_800.pt'
        baseline_sha=hashlib.sha256(baseline.read_bytes()).hexdigest();assert baseline_sha=='23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086'
        import isaacgym
        import torch
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        from legged_gym.utils.helpers import class_to_dict
        from rsl_rl.runners.him_side_coordination_runner import HIMSideCoordinationRunner
        cfg,train=task_registry.get_cfgs(TASK)
        audited=json.loads((REVIEW/'config_audit.json').read_text())
        assert class_to_dict(cfg)==audited['env'] and class_to_dict(train)==audited['train']
        # Only the user-authorized run budget overrides the reviewed default.
        train.runner.max_iterations=TARGET
        assert cfg.env.num_envs==4096 and train.runner.num_steps_per_env==100
        assert train.runner.save_interval==50 and cfg.sim.dt==.001 and cfg.control.decimation==20
        files=set(review['code_sha256'])
        previous=json.loads(source.with_name('run_metadata.json').read_text())
        for name,sha in previous['code_sha256'].items():
            if name not in files:assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
            files.add(name)
        files.update(['tools/train_lz_q1l_soft_slip.py','tools/run_lz_q1l_soft_slip.sh','tools/launch_lz_q1l_soft_slip.py'])
        metadata=dict(task=TASK,source_checkpoint=str(source),source_sha256=source_sha,source_iteration=300,
            physics_dt=.001,decimation=20,additional_iterations=TARGET,save_interval=50,
            iteration_numbering='New soft slip stage0..300 additional updates from front pose model300',
            load_optimizer=True,seed=1,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            teacher_enabled=False,mirroring=False,estimator_frozen=True,baseline_checkpoint=str(baseline),baseline_sha256=baseline_sha,
            configuration_changes=audited['changes'],review_directory=str(REVIEW),
            code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(files)})
        for name in files:
            dest=out/'provenance'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        for name in ['review_manifest.json','config_audit.json','synthetic_checks.json','trace_calibration.json','gpu_checks.json']:
            dest=out/'review'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(REVIEW/name,dest)
        write(out/'run_metadata.json',metadata);write(out/'env_config.json',class_to_dict(cfg));write(out/'train_config.json',class_to_dict(train))
        sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0']
        args=get_args();env,_=task_registry.make_env(TASK,args=args,env_cfg=cfg)
        runner=HIMSideCoordinationRunner(env,class_to_dict(train),str(out),device=env.device)
        saved=torch.load(source,map_location=env.device);assert saved['iter']==300
        def same(a,b):
            if isinstance(a,torch.Tensor):return torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
            return a==b
        model=runner.alg.actor_critic
        assert same(saved['model_state_dict'],model.state_dict())
        assert same(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict())
        assert same(saved['estimator_optimizer_state_dict'],model.estimator.optimizer.state_dict())
        assert all(not p.requires_grad for p in model.estimator.parameters()) and not hasattr(runner.alg,'teacher')
        frozen={k:v.detach().clone() for k,v in model.estimator.state_dict().items()}
        assert all(g['lr']==runner.alg.learning_rate==1e-5 for g in runner.alg.optimizer.param_groups)
        write(out/'resume_audit.json',dict(exact_model=True,optimizer_restored=True,estimator_frozen=True,teacher_absent=True,source_iteration=300,learning_rate=runner.alg.learning_rate))
        del saved
        last_metrics={};pop=env.pop_side_coordination_metrics
        def captured_metrics():
            result=pop();last_metrics.clear();last_metrics.update(result);return result
        env.pop_side_coordination_metrics=captured_metrics
        original_log,original_save=runner.log,runner.save
        original_save(str(out/'model_0.pt'),infos=metadata)
        def log(locs):
            original_log(locs);completed=locs['it']+1
            assert int((env.commands[:,1]>0).sum())==2048 and int((env.commands[:,1]<0).sum())==2048
            assert same(frozen,model.estimator.state_dict()) and not hasattr(runner.alg,'teacher')
            if completed%50==0:
                old_iteration=runner.current_learning_iteration;runner.current_learning_iteration=completed
                try:
                    path=out/('model_%d.pt'%completed);assert not path.exists()
                    original_save(str(path)+'.tmp',infos=metadata);os.replace(str(path)+'.tmp',path)
                finally:runner.current_learning_iteration=old_iteration
                assert hashlib.sha256(baseline.read_bytes()).hexdigest()==baseline_sha
            write(out/'status.json',dict(state='running',completed_updates=completed,target_updates=TARGET,
                iteration_seconds=locs['collection_time']+locs['learn_time'],pid=os.getpid(),
                rollout_direction_envs=[2048,2048],ppo=runner.alg.metrics,coordination=last_metrics))
        runner.log=log;runner.save=lambda *a,**k:None
        runner.learn(num_learning_iterations=TARGET,init_at_random_ep_len=True)
        assert runner.current_learning_iteration==TARGET
        for step in range(0,TARGET+1,50):assert (out/('model_%d.pt'%step)).is_file()
        runner.writer.flush();runner.writer.close()
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha
        assert hashlib.sha256(baseline.read_bytes()).hexdigest()==baseline_sha
        write(out/'status.json',dict(state='completed',completed_updates=TARGET,target_updates=TARGET))
        print('TRAINING_COMPLETE',flush=True)
    except BaseException as exc:
        state=json.loads((out/'status.json').read_text());state.update(state='failed',error=repr(exc));write(out/'status.json',state);raise
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
