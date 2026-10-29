"""Review-only invariants, recorded-trajectory calibration and disposable GPU check."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.lz_q1l_side_coordination import SideCoordinationTracker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'media/front_pose_review_20261029'
TASK = 'lz_q1l_flat_front_pose'


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def diff(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [x for k in sorted(set(a)|set(b)) for x in diff(a.get(k),b.get(k),prefix+'.'+k)]
    return [] if a == b else [prefix]


def same(a, b):
    if isinstance(a, torch.Tensor):return torch.equal(a,b)
    if isinstance(a, dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a, (list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b


def main():
    OUT.mkdir(exist_ok=True)
    cfg,train=task_registry.get_cfgs(TASK);old,_=task_registry.get_cfgs('lz_q1l_flat_side_coordination')
    changes=diff(class_to_dict(old),class_to_dict(cfg))
    assert set(changes)=={'.rewards.front_pose_window_s','.rewards.front_pose_deadband_deg','.rewards.front_pose_scale_deg','.rewards.scales.lateral_front_pose'}
    source=ROOT/'HIMLoco/legged_gym'/train.runner.source_checkpoint
    assert hashlib.sha256(source.read_bytes()).hexdigest()==train.runner.source_sha256
    write('config_audit.json',dict(env=class_to_dict(cfg),train=class_to_dict(train),changes=changes))
    from legged_gym.utils.lz_q1l_front_pose import FrontPoseTracker
    tracker=lambda n=2:FrontPoseTracker(n,'cpu',.02)
    a=tracker();active=torch.ones(2,dtype=torch.bool);q=torch.zeros(2,2,2)
    q[:,1,:]=math.radians(20)
    for _ in range(49):assert (a.update(q,active)==0).all()
    assert torch.allclose(a.update(q,active),torch.full((2,),2.),atol=1e-5)
    before=a.history[:,1].clone();a.reset(torch.tensor([0]));assert a.samples[0]==0 and torch.equal(a.history[:,1],before)
    assert (a.update(q,torch.zeros_like(active))==0).all() and (a.history==0).all()
    # Same mean with alternating opposite leg phases must not be penalized.
    a=tracker()
    for k in range(100):
        q.zero_();q[:,k%2,:]=math.radians(30)
        assert (a.update(q,active)<1e-6).all()
    a=tracker();b=tracker()
    for k in range(100):
        q=torch.randn(2,2,2)
        assert torch.allclose(a.update(q,active),b.update(q.flip(1),active),atol=1e-5)
    with torch.inference_mode():a.update(q,active)
    a.reset(torch.tensor([0]))
    write('synthetic_checks.json',dict(passed=True,window_reset_inactive_phase_and_permutation=True))
    trace=json.loads((ROOT/'media/side_coordination500_replay_20261029/run_20261029_163957_5N9d/result/trace.json').read_text())
    cal=[]
    for seg in range(2):
        a=tracker(1);cost=[]
        for row in trace[seg*250:(seg+1)*250]:
            q=torch.deg2rad(torch.tensor(row['joint_actual_deg']))[[7,8,10,11]].reshape(1,2,2)
            cost.append(float(a.update(q,torch.ones(1,dtype=torch.bool))[0]))
        cal.append(dict(direction=seg,mean_raw=sum(cost[50:])/200,mean_weighted_rate=sum(cost[50:])/200*cfg.rewards.scales.lateral_front_pose))
    write('trace_calibration.json',dict(rows=cal,weight=cfg.rewards.scales.lateral_front_pose,units='rate before dt=.02',scope='weight calibration, not learning evidence'))
    oldmeta=json.loads(source.with_name('run_metadata.json').read_text());files=set(oldmeta['code_sha256'])
    files.update(['HIMLoco/legged_gym/legged_gym/utils/lz_q1l_front_pose.py','HIMLoco/rsl_rl/rsl_rl/runners/him_front_pose_runner.py','tools/check_lz_q1l_front_pose.py','tools/train_lz_q1l_front_pose.py','tools/run_lz_q1l_front_pose.sh','tools/launch_lz_q1l_front_pose.py'])
    write('review_manifest.json',dict(source=str(source),source_sha256=train.runner.source_sha256,code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}))
    print('CPU_TRACE_PASS',cal,flush=True)
    from legged_gym.utils import get_args
    from rsl_rl.runners.him_front_pose_runner import HIMFrontPoseRunner
    sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0','--num_envs','32']
    args=get_args();cfg.env.num_envs=32;env=None
    try:
        env,_=task_registry.make_env(TASK,args=args,env_cfg=cfg)
        runner=HIMFrontPoseRunner(env,class_to_dict(train),log_dir=None,device=env.device)
        model=runner.alg.actor_critic;saved=torch.load(source,map_location=env.device)
        assert same(saved['model_state_dict'],model.state_dict())
        assert same(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict())
        assert same(saved['estimator_optimizer_state_dict'],model.estimator.optimizer.state_dict())
        estimator={k:v.clone() for k,v in model.estimator.state_dict().items()};actor={k:v.clone() for k,v in model.actor.state_dict().items()}
        assert not hasattr(runner.alg,'teacher') and all(not p.requires_grad for p in model.estimator.parameters())
        obs,_=env.reset();crit=env.get_privileged_observations()
        assert torch.equal(env.build_inference_policy(model)(obs),model.act_inference(obs))
        for _ in range(100):
            with torch.inference_mode():
                action=runner.alg.act(obs,crit);obs,crit,reward,done,infos,ids,terminal=env.step(action)
                nxt=crit.clone();nxt[ids]=terminal;runner.alg.process_env_step(reward,done,infos,nxt)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all() and torch.isfinite(env.side_values).all()
            assert int((env.commands[:,1]>0).sum())==16 and int((env.commands[:,1]<0).sum())==16
        with torch.inference_mode():runner.alg.compute_returns(crit)
        losses=runner.alg.update();assert all(math.isfinite(x) for x in losses)
        assert same(estimator,model.estimator.state_dict()) and not same(actor,model.actor.state_dict())
        # Repeated reward reads cannot advance the new trackers.
        state=env.side_tracker.unloaded_time.clone()
        assert torch.equal(env._reward_lateral_load_floor(),env._reward_lateral_load_floor())
        assert torch.equal(state,env.side_tracker.unloaded_time)
        # Reset clears only requested environments. Resampling also clears their windows.
        # Isaac Gym's base reset_buf is created inside rollout inference mode.
        # Exercise lifecycle hooks in that same context, as the training loop does.
        with torch.inference_mode():
            env.reset_idx(torch.tensor([0],device=env.device));assert env.side_tracker.samples[0]==0 and env.front_pose_tracker.samples[0]==0
            env._resample_commands(torch.tensor([1],device=env.device));assert env.side_tracker.samples[1]==0 and env.front_pose_tracker.samples[1]==0
            env.episode_length_buf.fill_(1)
            env.commands[2,1]*=-1
            env.commands[3,:3]=0.
            env._post_physics_step_callback()
            assert env.side_tracker.samples[2]==1 and env.side_tracker.samples[3]==0
            assert env.front_pose_tracker.samples[2]==1 and env.front_pose_tracker.samples[3]==0
            assert not env.side_active[3] and (env.side_values[3]==0).all()
            assert env._reward_feet_air_time()[3]==0
        write('gpu_checks.json',dict(passed=True,envs=32,steps=100,disposable_updates=1,no_checkpoint_saved=True,teacher_absent=True,native_actions=True,estimator_unchanged=True,actor_updated=True,optimizer_restored=True,metrics=env.pop_side_coordination_metrics()))
        print('GPU_PREFLIGHT_PASS',flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
