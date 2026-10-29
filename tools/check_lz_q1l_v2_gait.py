"""Check gait reward invariants and calibrate frozen 400/500 policies; no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2GaitRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from train_lz_q1l_v1_support_500 import differences

ROOT=Path(__file__).resolve().parents[1]
STARTS=np.stack([np.zeros(6),np.random.RandomState(1).uniform(-.03,.03,6),np.random.RandomState(2).uniform(-.03,.03,6)])
COMMANDS=[('stand',0.),('forward_020',.2),('forward_035',.35),('forward_050',.5)]
TERMS=['swing_clearance','feet_slide','rear_force_balance']

def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

class Probe(LZQ1LV2GaitRobot):
    def _resample_commands(self,ids):
        self.commands[ids]=0.;self.command_sample_mode[ids]=0
        self._clear_support(ids);self.v2_air_steps[ids]=0;self._clear_gait_state(ids)

    def _reset_root_states(self,ids):
        self.root_states[ids]=self.base_init_state
        self.root_states[ids,:3]+=self.env_origins[ids]
        self.root_states[ids,7:13]=torch.as_tensor(STARTS,dtype=torch.float,device=self.device)[ids%3]
        ix=ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),gymtorch.unwrap_tensor(ix),len(ix))

def invariants(env):
    cfg=env.cfg.rewards;device=env.device
    def tensor(x):return torch.tensor(x,dtype=torch.float,device=device)
    raw=LZQ1LV2GaitRobot.gait_penalty_values
    force=tensor([[0.,40.,40.,40.]]);gap=tensor([[0.,0.,0.,0.]])
    speed=tensor([[.35,0.,0.,0.]]);zero=torch.zeros_like(speed);rear=tensor([[40.,40.]])
    low=raw(force,gap,speed,zero,rear,cfg)
    high=raw(force,gap+.01,speed,zero,rear,cfg)
    assert low[0,0]>0 and high[0,0]==0 and low[0,1]==0 and low[0,2]==0
    assert raw(force,gap,zero,zero,rear,cfg)[0,0]==0  # No penalty for stationary feet.
    touch=force.clone();touch[0,0]=2.
    assert raw(touch,gap,speed,speed,rear,cfg)[0,1]>0  # Includes lightly loaded scuffing.
    assert raw(force,gap,speed,speed,rear,cfg)[0,1]==0  # Free swing is not sliding.
    assert raw(touch,gap,speed,zero+.02,rear,cfg)[0,1]==0  # Deadband.
    assert raw(force,gap,speed,zero,tensor([[41.,37.]]),cfg)[0,2]==0
    assert raw(force,gap,speed,zero,tensor([[40.,25.]]),cfg)[0,2]>0
    assert raw(force,gap,speed,zero,tensor([[0.,0.]]),cfg)[0,2]>0  # No zero-load loophole.
    # Swapping left/right feet cannot change any of the three scalar penalties.
    gen=torch.Generator(device=device);gen.manual_seed(123)
    f=torch.rand(12,4,generator=gen,device=device)*30
    g=torch.rand(12,4,generator=gen,device=device)*.02
    rv=torch.rand(12,4,generator=gen,device=device)
    gv=torch.rand(12,4,generator=gen,device=device)
    r=torch.rand(12,2,generator=gen,device=device)*60
    perm=[1,0,3,2]
    assert torch.allclose(raw(f,g,rv,gv,r,cfg),raw(f[:,perm],g[:,perm],rv[:,perm],gv[:,perm],r[:,[1,0]],cfg),atol=1e-6)
    assert torch.isfinite(raw(f,g,rv,gv,r,cfg)).all()

    # Exercise the actual per-environment ring buffer and lifecycle hooks.
    ids=torch.arange(env.num_envs,device=device)
    rear_ids=env.feet_indices[env.rear_foot_slots]
    env.commands.zero_();env.commands[:,0]=.35;env._clear_gait_state(ids)
    for i in range(env.rear_force_window_steps):
        env.contact_forces.zero_()
        env.contact_forces[:,rear_ids[i%2],2]=80.
        env._update_gait_rewards()
        if i<env.rear_force_window_steps-1:assert not env._reward_rear_force_balance().any()
    assert torch.allclose(env.rear_force_mean,torch.full_like(env.rear_force_mean,40.))
    assert not env._reward_rear_force_balance().any()
    steps=env.gait_forward_steps.clone();history=env.rear_force_history.clone();cursor=env.rear_force_cursor
    for _ in range(3):
        env._reward_rear_force_balance();env._reward_swing_clearance();env._reward_feet_slide()
    assert torch.equal(steps,env.gait_forward_steps) and torch.equal(history,env.rear_force_history) and cursor==env.rear_force_cursor
    env.reset_idx(ids[:1])
    assert not env.rear_force_history[:,0].any() and env.gait_forward_steps[0]==0
    assert torch.equal(history[:,1:],env.rear_force_history[:,1:])
    env.commands.zero_();env._update_gait_rewards()
    assert not env.gait_penalties.any() and not env.rear_force_sum.any()
    env.commands[:,0]=.35;env._clear_gait_state(ids)
    env.contact_forces.zero_();env.contact_forces[:,rear_ids[0],2]=40.;env.contact_forces[:,rear_ids[1],2]=25.
    for _ in range(env.rear_force_window_steps):env._update_gait_rewards()
    assert (env._reward_rear_force_balance()>0).all()
    env.commands[:,0]=.30;env._update_gait_rewards()
    assert (env.gait_forward_steps==1).all() and not env.gait_penalties.any()
    env.commands[:,2]=.3;env._update_gait_rewards()
    assert not env.gait_penalties.any()
    env.reset()
    return dict(clearance=True,slide=True,balance=True,left_right_invariance=True,
        alternating_support_not_penalized=True,partial_reset_isolated=True,command_change_warmup=True,
        standing_and_turning_disabled=True,reward_calls_do_not_mutate_history=True)

def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',required=True,type=Path)
    opts,rest=parser.parse_known_args();sys.argv=[sys.argv[0]]+rest;args=get_args()
    assert args.task=='lz_q1l_flat_v2_gait'
    out=opts.out.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=Path(json.loads((ROOT/'media/v2_posture_20261024/latest_training.json').read_text())['run_dir'])
    cfg,train=task_registry.get_cfgs(args.task)
    old,old_train=task_registry.get_cfgs('lz_q1l_flat_v2_posture')
    old_dict=class_to_dict(old);archived=json.loads((parent/'env_config.json').read_text())
    archived['domain_rand'].pop('push_interval',None)
    assert old_dict==archived,'Existing posture configuration changed'
    diff=differences(old_dict,class_to_dict(cfg))
    assert all(k.startswith('rewards.') for k in diff),diff
    assert old.rewards.scales.foot_clearance==-.01 and cfg.rewards.scales.foot_clearance==0.
    assert cfg.commands.zero_command_prob==cfg.commands.pure_forward_prob==.5
    assert cfg.sim.dt==.001 and cfg.control.decimation==20
    assert cfg.control.stiffness=={'joint':80.} and cfg.control.damping=={'joint':2.}
    cfg.env.num_envs=args.num_envs=24;cfg.env.episode_length_s=20.
    cfg.commands.resampling_time=1e6;cfg.noise.add_noise=False
    cfg.sim.use_gpu_pipeline=cfg.sim.physx.use_gpu=args.sim_device.startswith('cuda');cfg.sim.physx.num_threads=2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
    args.headless=True;task_registry.task_classes[args.task]=Probe
    sources=[Path(__file__).resolve()]+[ROOT/p for p in (
        'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
        'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py',
        'HIMLoco/legged_gym/legged_gym/envs/__init__.py')]
    (out/'provenance').mkdir()
    for p in sources:shutil.copyfile(p,out/'provenance'/p.name)
    manifest=dict(task=args.task,backend=args.sim_device,training_updates=0,env_diff=diff,
        starts=STARTS.tolist(),commands=COMMANDS,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol='Frozen models 400 and 500, each 4 commands x 3 starts; 2s standing then 8s command. Analyze forward seconds 1-8 before first reset. Reward contribution per second, before common dt multiplication.')
    save(out/'manifest.json',manifest)
    env=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
        for name in TERMS:assert abs(env.reward_scales[name]-getattr(cfg.rewards.scales,name)*env.dt)<1e-8
        assert 'foot_clearance' not in env.reward_scales
        checks=invariants(env);save(out/'checks.json',checks);print('RULE_CHECKS_OK',checks,flush=True)
        policies=[];ckpts=[]
        for iteration in (400,500):
            path=parent/('model_%d.pt'%iteration);state=torch.load(path,map_location=env.device);assert state['iter']==iteration
            runner=HIMOnPolicyRunner(env,class_to_dict(train),None,device=env.device)
            runner.alg.actor_critic.load_state_dict(state['model_state_dict'],strict=True)
            policies.append(runner.get_inference_policy(device=env.device))
            ckpts.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest['checkpoints']=ckpts;save(out/'manifest.json',manifest);save(out/'config.json',class_to_dict(cfg))
        obs,_=env.reset();alive=torch.ones(24,dtype=torch.bool,device=env.device);resets=[]
        data={k:[] for k in ('valid','terms','tracking','fz','gap','vx','height')}
        with torch.no_grad():
            for step in range(500):
                env.commands.zero_();env.command_sample_mode[:]=0
                if step>=100:
                    for model in range(2):
                        for cmd,(_,speed) in enumerate(COMMANDS):
                            ids=slice(model*12+cmd*3,model*12+cmd*3+3)
                            env.commands[ids,0]=speed;env.command_sample_mode[ids]=int(speed>0)
                obs[:,:3]=env.commands[:,:3]*env.commands_scale
                actions=torch.cat([policies[0](obs[:12]),policies[1](obs[12:])])
                obs,_,reward,done,_,_,_=env.step(actions)
                for i in (alive&done.bool()).nonzero().flatten().tolist():resets.append(dict(index=i,time_s=(step+1)*.02))
                alive &= ~done.bool()
                tracking=env._reward_tracking_lin_vel()*cfg.rewards.scales.tracking_lin_vel+env._reward_tracking_ang_vel()*cfg.rewards.scales.tracking_ang_vel
                vals=dict(valid=alive,terms=env.gait_penalties,tracking=tracking,
                    fz=env.contact_forces[:,env.feet_indices,2],gap=env.support_foot_gap,vx=env.base_lin_vel[:,0],height=env.root_states[:,2])
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                for k,v in vals.items():data[k].append(v.cpu().numpy().copy())
                if (step+1)%100==0:print('STEP',step+1,flush=True)
        a={k:np.asarray(v) for k,v in data.items()};np.savez_compressed(out/'traces.npz',**a)
        assert all(np.isfinite(v).all() for v in a.values())
        weights=np.array([getattr(cfg.rewards.scales,k) for k in TERMS]);summaries=[]
        for model,iteration in enumerate((400,500)):
            for cmd,(name,speed) in enumerate(COMMANDS):
                ids=np.arange(model*12+cmd*3,model*12+cmd*3+3);mask=a['valid'][150:,ids]
                row=dict(iteration=iteration,command=name,trials=3,failures=sum(r['index'] in ids for r in resets),valid_fraction=float(mask.mean()))
                if mask.any():
                    terms=a['terms'][150:,ids][mask];weighted=terms*weights;tracking=a['tracking'][150:,ids][mask]
                    if speed==0:assert not terms.any(),'New gait penalties applied to standing'
                    row.update(raw_mean=terms.mean(0).tolist(),weighted_mean=weighted.mean(0).tolist(),
                        tracking_mean=float(tracking.mean()),penalty_to_tracking_ratio=float(-weighted.sum(1).mean()/tracking.mean()),
                        vx=float(a['vx'][150:,ids][mask].mean()),height_m=float(a['height'][150:,ids][mask].mean()))
                summaries.append(row)
        for ck in ckpts:assert hashlib.sha256(Path(ck['path']).read_bytes()).hexdigest()==ck['sha256']
        save(out/'report.json',dict(checks=checks,terms=TERMS,weights=weights.tolist(),cases=summaries,resets=resets,
            training_updates=0,checkpoint_weights_unchanged=True,finite=True))
        print('RESULT',json.dumps(summaries),flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
