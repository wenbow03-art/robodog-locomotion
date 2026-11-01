"""300 additional updates with the front/rear stance-width penalty."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'HIMLoco/legged_gym/logs2/baselines/stand_walk800_20261027'
SOURCES = {
    'standing': ('gait_150.pt', '655a5022f2e1404c8f7c3dfbb670cc8235710f78e7550e72f2343e4e2c5e9747', 150,
        'flat_lz_q1l_v2_gait/gait_from_posture500_200_20261025_113614_3235946'),
    'forward': ('model_800.pt', '23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086', 800,
        'flat_lz_q1l_v3_right_rear_split/rh_split_from600_to800_20261027_172314_1964027'),
}
FILES = [
    'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
    'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py',
    'HIMLoco/legged_gym/legged_gym/envs/__init__.py',
    'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_stance_width.py',
    'HIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py',
    'HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
    'HIMLoco/rsl_rl/rsl_rl/modules/him_stance_width_actor.py',
    'tools/train_lz_q1l_stance_width.py',
    'tools/run_lz_q1l_stance_width.sh',
]

def write(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)

def same(a, b):
    import torch
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys()==b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple,list)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', choices=list(SOURCES), required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--initial-learning-rate', type=float, choices=[1e-5])
    opts=p.parse_args()
    out=opts.run_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    target=2 if opts.preflight else 300
    write(out/'status.json', dict(state='initializing',completed_updates=0,target_updates=target,preflight=opts.preflight))
    env=None
    try:
        import isaacgym
        import torch
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        from legged_gym.utils.helpers import class_to_dict
        from rsl_rl.runners import HIMOnPolicyRunner
        from train_lz_q1l_v1_support_500 import differences
        task='lz_q1l_flat_stance_width_'+opts.model
        cfg,train=task_registry.get_cfgs(task)
        if opts.initial_learning_rate is not None:
            train.algorithm.learning_rate=opts.initial_learning_rate
        name,digest,iteration,original=SOURCES[opts.model]
        source=BASE/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        original=ROOT/'HIMLoco/legged_gym/logs2'/original
        old_cfg=json.loads((original/'env_config.json').read_text())
        old_cfg['domain_rand'].pop('push_interval',None)
        changes=differences(old_cfg,class_to_dict(cfg))
        assert set(changes)=={'rewards.stance_width_window_s','rewards.stance_width_scale_m','rewards.scales.front_rear_width_gap'},changes
        assert cfg.sim.dt==.001 and cfg.control.decimation==20
        assert cfg.env.num_envs==4096 and train.runner.num_steps_per_env==100
        assert train.runner.save_interval==50
        if opts.preflight: cfg.env.num_envs=32
        write(out/'env_config.json',class_to_dict(cfg))
        write(out/'train_config.json',class_to_dict(train))
        files=set(FILES)
        previous=json.loads((original/'run_metadata.json').read_text())
        files.update(previous.get('code_sha256',{}))
        metadata=dict(task=task,source_checkpoint=str(source),source_sha256=digest,source_iteration=iteration,
            source_run=str(original),additional_iterations=target,save_interval=50,preflight=opts.preflight,
            physics_dt=.001,decimation=20,estimator_frozen=True,configuration_changes=changes,
            policy_training='all_legs',
            teacher_enabled=opts.model=='forward',seed=1,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            initial_learning_rate_override=opts.initial_learning_rate,
            code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(files)})
        for f in files:
            dest=out/'provenance'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest)
        write(out/'run_metadata.json',metadata)
        sys.argv=[sys.argv[0],'--task',task,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0']
        args=get_args();env,_=task_registry.make_env(task,args=args,env_cfg=cfg)
        runner=HIMOnPolicyRunner(env,class_to_dict(train),str(out),device=env.device)
        saved=torch.load(source,map_location=env.device)
        assert saved['iter']==iteration
        model=runner.alg.actor_critic
        model.load_state_dict(saved['model_state_dict'],strict=True)
        model.estimator.optimizer.load_state_dict(saved['estimator_optimizer_state_dict'])
        import copy
        restored_optimizer=copy.deepcopy(saved['optimizer_state_dict'])
        if opts.model=='forward':
            old_params=restored_optimizer['param_groups'][0]['params']
            current_params=runner.alg.optimizer.state_dict()['param_groups'][0]['params']
            assert len(restored_optimizer['param_groups'])==1
            assert old_params==current_params[:len(old_params)]
            assert len(current_params)-len(old_params)==len(list(model.actor.parameters()))
            restored_optimizer['param_groups'][0]['params']=current_params
        runner.alg.optimizer.load_state_dict(restored_optimizer)
        runner.alg.learning_rate=saved['optimizer_state_dict']['param_groups'][0]['lr']
        assert same(model.state_dict(),saved['model_state_dict'])
        assert same(runner.alg.optimizer.state_dict(),restored_optimizer)
        assert same(model.estimator.optimizer.state_dict(),saved['estimator_optimizer_state_dict'])
        saved_learning_rate=runner.alg.learning_rate
        if opts.initial_learning_rate is not None:
            runner.alg.learning_rate=opts.initial_learning_rate
            for group in runner.alg.optimizer.param_groups:group['lr']=runner.alg.learning_rate
        if opts.model=='forward':
            from train_lz_q1l_v3_right_rear_only import attach_teacher
            attach_teacher(runner,env,train)
            runner.alg.assert_frozen_policy()
        assert all(not v.requires_grad for v in model.estimator.parameters())
        frozen={k:v.clone() for k,v in model.estimator.state_dict().items()}
        frozen_optimizer=copy.deepcopy(model.estimator.optimizer.state_dict())
        initial={k:v.clone() for k,v in model.state_dict().items()}
        if opts.model=='forward':
            from rsl_rl.modules.him_right_rear_only import HIMRightRearOnlyActorCritic
            reference=HIMRightRearOnlyActorCritic(env.num_obs,env.num_privileged_obs,env.num_one_step_obs,
                env.num_actions,**class_to_dict(train.policy)).to(env.device)
            reference.configure_right_rear(env.dof_names,env.commands_scale)
            reference.load_state_dict(saved['model_state_dict'])
            with torch.no_grad():
                probe=torch.randn(32,env.num_obs,device=env.device)*.1
                assert torch.equal(reference.act_inference(probe),model.act_inference(probe))
            del reference
            # Verify every output, including non-RH joints, can train the base actor.
            for joint in range(env.num_actions):
                model.zero_grad()
                model.act_inference(probe)[:,joint].sum().backward()
                assert any(v.grad is not None and v.grad.abs().sum()>0 for v in model.actor.parameters())
                assert all(v.grad is None for v in model.estimator.parameters())
            model.zero_grad()
        write(out/'resume_audit.json',dict(exact_model=True,original_optimizer_states_restored=True,
            newly_unfrozen_actor_fresh_optimizer_states=opts.model=='forward',exact_estimator_optimizer=True,
            learning_rate=runner.alg.learning_rate,saved_learning_rate=saved_learning_rate,
            estimator_frozen=True,source_iteration=iteration))
        del saved
        runner.current_learning_iteration=0
        original_log,original_save=runner.log,runner.save
        original_save(str(out/'model_0.pt'),infos=metadata)
        def log(locs):
            original_log(locs)
            completed=locs['it']+1
            assert same(frozen,model.estimator.state_dict()),'Estimator changed'
            assert same(frozen_optimizer,model.estimator.optimizer.state_dict()),'Estimator optimizer changed'
            assert all(torch.isfinite(t).all() for t in model.state_dict().values()),'Nonfinite model'
            metrics=env.pop_stance_width_metrics()
            for key,value in metrics.items():runner.writer.add_scalar('StanceWidth/'+key,value,completed)
            losses={k:float(locs[k]) for k in ['mean_value_loss','mean_surrogate_loss','mean_estimation_loss','mean_swap_loss']}
            if completed%50==0 or (opts.preflight and completed==target):
                old=runner.current_learning_iteration;runner.current_learning_iteration=completed
                try:
                    path=out/('model_%d.pt'%completed)
                    assert not path.exists()
                    original_save(str(path)+'.tmp',infos=metadata);os.replace(str(path)+'.tmp',path)
                finally:runner.current_learning_iteration=old
            write(out/'status.json',dict(state='running',completed_updates=completed,target_updates=target,
                preflight=opts.preflight,pid=os.getpid(),iteration_seconds=locs['collection_time']+locs['learn_time'],
                learning_rate=runner.alg.learning_rate,width=metrics,losses=losses))
        runner.log=log;runner.save=lambda *a,**kw:None
        runner.learn(num_learning_iterations=target,init_at_random_ep_len=True)
        assert runner.current_learning_iteration==target
        actor_prefix='actor.'
        assert any(not torch.equal(v,model.state_dict()[k]) for k,v in initial.items() if k.startswith(actor_prefix))
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        if opts.preflight:
            # Match the runner's inference-mode reset context for legacy buffers.
            ids=torch.arange(env.num_envs,device=env.device)
            with torch.inference_mode():
                env.reset_idx(ids)
            assert not env.width_tracker.cost.any() and not env.width_tracker.samples.any()
            write(out/'checks.json',dict(passed=True,discarded_updates=target,exact_resume=True,
                actor_updated=True,estimator_unchanged=True,estimator_optimizer_unchanged=True,lifecycle_passed=True))
        else:
            for i in range(0,301,50):assert (out/('model_%d.pt'%i)).is_file()
        runner.writer.flush();runner.writer.close()
        status=json.loads((out/'status.json').read_text());status['state']='completed';write(out/'status.json',status)
        print('PREFLIGHT_PASSED' if opts.preflight else 'TRAINING_COMPLETE',flush=True)
    except BaseException as exc:
        status=json.loads((out/'status.json').read_text());status.update(state='failed',error=repr(exc));write(out/'status.json',status);raise
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
