"""Resume native side model250 with tenfold height cost; no teacher or mirror."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SELECTED = json.loads((ROOT/'media/native_side_height_20261029/selected.json').read_text())
SOURCE = Path(SELECTED['checkpoint'])
SHA = SELECTED['checkpoint_sha256']
TASK = 'lz_q1l_flat_native_side_height'
TARGET_UPDATES = 300
FILES = [
    'HIMLoco/rsl_rl/rsl_rl/algorithms/him_frozen_estimator_ppo.py',
    'HIMLoco/rsl_rl/rsl_rl/runners/him_native_side_speed_runner.py',
    'HIMLoco/legged_gym/legged_gym/utils/task_registry.py',
    'tools/train_lz_q1l_native_side_height.py',
    'HIMLoco/legged_gym/legged_gym/envs/__init__.py',
    'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
    'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py',
    'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py',
    'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
    'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
    'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py',
    'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_cadence.py',
    'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_mirror.py',
    'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf',
    'HIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py',
    'HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py',
    'HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
    'HIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py',
]


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a)|set(b)) for p in differences(a.get(k), b.get(k), prefix+'.'+k)]
    return [] if a == b else [prefix]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    opts=parser.parse_args();out=opts.run_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(state='initializing',completed_updates=0,target_updates=TARGET_UPDATES))
    env=runner=None
    try:
        import isaacgym
        import torch
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        from legged_gym.utils.helpers import class_to_dict
        from rsl_rl.runners.him_native_side_speed_runner import HIMNativeSideSpeedRunner
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
        baseline=ROOT/'HIMLoco/legged_gym/logs2/baselines/stand_walk800_20261027/model_800.pt'
        baseline_sha=hashlib.sha256(baseline.read_bytes()).hexdigest()
        assert baseline_sha=='23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086'
        cfg,train=task_registry.get_cfgs(TASK)
        old=json.loads(SOURCE.with_name('env_config.json').read_text())
        old['domain_rand'].pop('push_interval',None)
        changed=differences(old,class_to_dict(cfg))
        assert changed==['.rewards.scales.base_height'],changed
        assert cfg.commands.curriculum==False and cfg.commands.pure_lateral_prob==1.
        old_train=json.loads(SOURCE.with_name('train_config.json').read_text())
        assert old_train['policy']==class_to_dict(train)['policy']
        assert old_train['algorithm']==class_to_dict(train)['algorithm']
        assert train.runner.num_steps_per_env==old_train['runner']['num_steps_per_env']==100
        assert train.runner.max_iterations==TARGET_UPDATES and train.runner.save_interval==50
        assert cfg.sim.dt==.004 and cfg.control.decimation==5
        assert differences(old['rewards'],class_to_dict(cfg)['rewards'])==['.scales.base_height']
        assert old['rewards']['scales']['base_height']==-10.
        assert cfg.rewards.scales.base_height==-100. and cfg.rewards.base_height_target==.30
        assert cfg.rewards.scales.tracking_lateral==3.0
        previous,_=task_registry.get_cfgs('lz_q1l_flat_native_side_speed')
        assert differences(class_to_dict(previous),class_to_dict(cfg))==['.rewards.scales.base_height']
        assert 'anchor' not in class_to_dict(train)
        metadata=dict(task=TASK,source_checkpoint=str(SOURCE),source_sha256=SHA,
            physics_dt=.004,decimation=5,additional_iterations=TARGET_UPDATES,save_interval=50,
            iteration_numbering='Additional height stage updates0..300; source native side speed model250',source_iteration=SELECTED['iteration'],
            baseline_checkpoint=str(baseline),baseline_sha256=baseline_sha,
            load_optimizer=True,seed=1,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            configuration_changes=changed,code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES},
            mirroring='NO mirroring in training or task playback; real left/right50/50.',teacher_enabled=False,lateral_tracking_scale=3.,base_height_scale=-100.,estimator_frozen=True)
        for name in FILES:
            dest=out/'provenance'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        write(out/'run_metadata.json',metadata)
        write(out/'env_config.json',class_to_dict(cfg));write(out/'train_config.json',class_to_dict(train))
        sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0']
        args=get_args()
        if opts.preflight:cfg.env.num_envs=args.num_envs=32
        env,_=task_registry.make_env(TASK,args=args,env_cfg=cfg)
        runner=HIMNativeSideSpeedRunner(env,class_to_dict(train),None if opts.preflight else str(out),device=env.device)
        saved=torch.load(SOURCE,map_location=env.device);assert saved['iter']==SELECTED['iteration']==250
        runner.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
        runner.alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
        runner.alg.actor_critic.estimator.optimizer.load_state_dict(saved['estimator_optimizer_state_dict'])
        def equal_state(a,b):
            if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
            return a==b
        assert equal_state(runner.alg.optimizer.state_dict(),saved['optimizer_state_dict'])
        assert equal_state(runner.alg.actor_critic.estimator.optimizer.state_dict(),saved['estimator_optimizer_state_dict'])
        assert all(g['lr']==runner.alg.learning_rate==1e-5 for g in runner.alg.optimizer.param_groups)
        estimator=runner.alg.actor_critic.estimator
        original_estimator={k:v.detach().clone() for k,v in estimator.state_dict().items()}
        assert all(not p.requires_grad for p in estimator.parameters())
        assert not hasattr(runner.alg,'teacher') and not hasattr(runner.alg,'anchor_loss')
        assert not any(id(p) in {id(q) for q in estimator.parameters()} for g in runner.alg.optimizer.param_groups for p in g['params'])
        assert all(torch.equal(v,runner.alg.actor_critic.state_dict()[k]) for k,v in saved['model_state_dict'].items())
        write(out/'resume_audit.json',dict(exact_model=True,optimizer_restored=True,estimator_frozen=True,
            optimizer_learning_rate=runner.alg.learning_rate,estimator_learning_rate=estimator.learning_rate))
        del saved
        if opts.preflight:
            obs,_=env.reset();model=runner.alg.actor_critic
            counts=env.native_sample_counts
            for ids in [torch.arange(31,device=env.device),torch.arange(17,device=env.device),torch.arange(32,device=env.device)]:env._resample_commands(ids)
            assert int((env.commands[:,1]>0).sum())==env.num_envs//2
            assert int((env.commands[:,1]<0).sum())==env.num_envs//2
            assert (env.commands[:,0]==0).all() and (env.commands[:,2]==0).all()
            obs=env.get_observations();obs[:,:3]=env.commands[:,:3]*env.commands_scale
            model.update_distribution(obs)
            assert torch.equal(model.action_mean,model.act_inference(obs))
            assert torch.equal(env.build_inference_policy(model)(obs),model.act_inference(obs))
            actor_before={k:v.detach().clone() for k,v in model.actor.state_dict().items()}
            assert abs(env.reward_scales['tracking_lateral']-3.*env.dt)<1e-9
            assert abs(env.reward_scales['base_height']-(-100.*env.dt))<1e-9
            height_before=env.root_states[:,2].clone()
            for height in (.30,.25,.175):
                env.root_states[:,2]=height
                raw=env._reward_base_height()
                assert torch.allclose(raw,torch.full_like(raw,(height-.30)**2),atol=1e-7)
            env.root_states[:,2].copy_(height_before)
            for speed in (.3,-.3):
                command_before=env.commands.clone();velocity_before=env.base_lin_vel.clone()
                env.commands.zero_();env.commands[:,1]=speed
                env.base_lin_vel[:,1]=speed
                assert torch.allclose(env._reward_tracking_lateral(),torch.ones(env.num_envs,device=env.device))
                env.base_lin_vel[:,1]=0.
                assert torch.allclose(env._reward_tracking_lateral(),torch.full((env.num_envs,),__import__('math').exp(-.3**2/.04),device=env.device))
                env.commands.copy_(command_before);env.base_lin_vel.copy_(velocity_before)
            # One actual disposable PPO update checks stored likelihoods and
            # estimator isolation and actor learning; formal run reloads source model250 and optimizer states.
            crit=env.get_privileged_observations()
            for step in range(100):
                with torch.inference_mode():
                    action=runner.alg.act(obs,crit)
                    assert torch.allclose(model.get_actions_log_prob(action),runner.alg.transition.actions_log_prob)
                    obs,crit,reward,done,infos,ids,terminal=env.step(action)
                    next_crit=crit.clone();next_crit[ids]=terminal
                    runner.alg.process_env_step(reward,done,infos,next_crit)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            with torch.inference_mode():runner.alg.compute_returns(crit)
            values=runner.alg.update();assert all(__import__('math').isfinite(x) for x in values)
            assert all(torch.equal(v,estimator.state_dict()[k]) for k,v in original_estimator.items())
            assert not hasattr(runner.alg,'teacher')
            assert int((env.commands[:,1]>0).sum())==env.num_envs//2
            assert int((env.commands[:,1]<0).sum())==env.num_envs//2
            assert any(not torch.equal(v,model.actor.state_dict()[k]) for k,v in actor_before.items())
            write(out/'checks.json',dict(passed=True,only_height_scale_changed=True,optimizer_restored=True,source_sha256=SHA,lateral_tracking_scale=3.,actor_updated=True,finite=True,
                envs=32,steps=100,native_no_mirror=True,teacher_absent=True,estimator_frozen=True,sampled_counts=env.native_sample_counts,physics_dt=env.cfg.sim.dt,decimation=env.cfg.control.decimation,
                code_sha256=metadata['code_sha256']))
            write(out/'status.json',dict(state='preflight_passed',completed_updates=0,target_updates=TARGET_UPDATES))
            print('GPU_PREFLIGHT_PASS',flush=True);return
        runner.current_learning_iteration=0
        original_log,original_save=runner.log,runner.save
        original_save(str(out/'model_0.pt'),infos=metadata)
        def log(locs):
            original_log(locs);completed=locs['it']+1
            assert int((env.commands[:,1]>0).sum())==env.num_envs//2
            assert int((env.commands[:,1]<0).sum())==env.num_envs//2
            assert all(torch.equal(v,estimator.state_dict()[k]) for k,v in original_estimator.items())
            assert not hasattr(runner.alg,'teacher')
            if completed%50==0:
                old_iteration=runner.current_learning_iteration;runner.current_learning_iteration=completed
                try:
                    path=out/('model_%d.pt'%completed);assert not path.exists()
                    original_save(str(path)+'.tmp',infos=metadata);os.replace(str(path)+'.tmp',path)
                finally:runner.current_learning_iteration=old_iteration
                assert hashlib.sha256(baseline.read_bytes()).hexdigest()==baseline_sha
            write(out/'status.json',dict(state='running',completed_updates=completed,target_updates=TARGET_UPDATES,
                iteration_seconds=locs['collection_time']+locs['learn_time'],pid=os.getpid(),sampled_counts=env.native_sample_counts,rollout_direction_envs=[env.num_envs//2,env.num_envs//2],ppo=runner.alg.metrics))
        runner.log=log;runner.save=lambda *a,**k:None
        runner.learn(num_learning_iterations=TARGET_UPDATES,init_at_random_ep_len=True)
        assert runner.current_learning_iteration==TARGET_UPDATES
        for step in range(50,TARGET_UPDATES+1,50):assert (out/('model_%d.pt'%step)).is_file()
        runner.writer.flush();runner.writer.close()
        assert hashlib.sha256(baseline.read_bytes()).hexdigest()==baseline_sha
        write(out/'status.json',dict(state='completed',completed_updates=TARGET_UPDATES,target_updates=TARGET_UPDATES))
        print('TRAINING_COMPLETE',flush=True)
    except BaseException as exc:
        state=json.loads((out/'status.json').read_text());state.update(state='failed',error=repr(exc))
        write(out/'status.json',state);raise
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
