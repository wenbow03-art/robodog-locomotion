"""Standing-load reward invariants and frozen gait200 CPU smoke test; no training."""
import hashlib
import json
from pathlib import Path
import sys

import isaacgym
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2GaitHeightRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from check_lz_q1l_v2_gait import invariants as gait_invariants
from train_lz_q1l_v1_support_500 import differences

ROOT=Path(__file__).resolve().parents[1]


def checks(env):
    cfg=env.cfg.rewards
    def value(x):
        return env.stand_force_penalty_values(torch.tensor(x,dtype=torch.float,device=env.device),cfg)
    assert not value([[40,40,40,40],[20,20,60,60],[41,39,60,60]]).any()
    assert value([[70,10,10,70]])[0,0]>0  # Equal whole-side totals still penalized.
    assert value([[0,0,80,80]])[0,0]==0 and value([[0,0,80,80]])[0,1]>.49
    assert value([[0,0,0,0]])[0,1]==1
    assert not value([[10,30,30,30]])[0,1]
    assert value([[5,35,30,30]])[0,1]>0
    assert torch.equal(value([[-10,30,30,30]]),value([[0,30,30,30]]))
    force=torch.tensor([[72.,14.,21.,42.],[70.,10.,10.,70.]],device=env.device)
    for order in ([1,0,3,2],[2,3,0,1]):
        assert torch.allclose(env.stand_force_penalty_values(force,cfg),env.stand_force_penalty_values(force[:,order],cfg))
    assert torch.isfinite(value([[0,0,0,0],[1e-8,0,0,0]])).all()
    ids=torch.arange(env.num_envs,device=env.device);foot_ids=env.feet_indices[env.stand_force_slots]
    def fill():
        env.contact_forces.zero_()
        env.contact_forces[:,foot_ids,2]=torch.tensor([72.,14.,21.,42.],device=env.device)
    env.commands.zero_();env._clear_stand_force(ids);fill()
    for i in range(env.stand_force_ready_steps):
        env._update_stand_force_rewards()
        if i<env.stand_force_ready_steps-1:assert not env.stand_force_penalties.any()
    assert torch.allclose(env.stand_force_mean,torch.tensor([72.,14.,21.,42.],device=env.device).expand(env.num_envs,-1))
    assert (env._reward_stand_pair_force_balance()>0).all()
    history=env.stand_force_history.clone();steps=env.stand_force_steps.clone();cursor=env.stand_force_cursor
    for _ in range(3):env._reward_stand_pair_force_balance();env._reward_stand_low_load()
    assert torch.equal(history,env.stand_force_history) and torch.equal(steps,env.stand_force_steps) and cursor==env.stand_force_cursor
    env.reset_idx(ids[:1]);assert not env.stand_force_history[:,0].any()
    assert torch.equal(history[:,1:],env.stand_force_history[:,1:])
    env._resample_commands(ids[1:2]);assert not env.stand_force_history[:,1].any()
    assert torch.equal(history[:,2:],env.stand_force_history[:,2:])
    for command in ([.35,0,0],[0,.2,0],[0,0,.3]):
        env.commands[:,:3]=torch.tensor(command,device=env.device);env._update_stand_force_rewards()
        assert not env.stand_force_penalties.any() and not env.stand_force_sum.any()
    env.commands.zero_();fill();env._update_stand_force_rewards()
    assert (env.stand_force_steps==1).all() and not env.stand_force_penalties.any()
    # Check sliding-window replacement after wrap-around against a direct mean.
    env._clear_stand_force(ids);samples=[]
    for i in range(2*env.stand_force_window_steps+7):
        env.contact_forces[:,foot_ids,2]=float(i+1);samples.append(float(i+1))
        env._update_stand_force_rewards()
        if len(samples)>=env.stand_force_window_steps:
            expected=np.mean(samples[-env.stand_force_window_steps:])
            assert torch.allclose(env.stand_force_mean,torch.full_like(env.stand_force_mean,expected))
    env.reset()
    return dict(pair_balance=True,front_rear_distribution_allowed=True,diagonal_bias_detected=True,
        low_load_guard=True,zero_total_safe=True,left_right_symmetry=True,grace=True,
        rolling_window=True,partial_reset_isolated=True,resample_clears=True,
        command_change_clears=True,movement_disabled=True,reward_getters_pure=True)


def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    sys.argv=[sys.argv[0],'--task','lz_q1l_flat_v2_gait_height','--headless','--seed','1',
              '--sim_device','cpu','--rl_device','cpu','--num_envs','6']
    args=get_args();cfg,train=task_registry.get_cfgs(args.task)
    old,_=task_registry.get_cfgs('lz_q1l_flat_v2_gait')
    diff=differences(class_to_dict(old),class_to_dict(cfg))
    expected={'rewards.scales.base_height','rewards.scales.stand_pair_force_balance','rewards.scales.stand_low_load',
        'rewards.stand_force_window_s','rewards.stand_force_grace_s','rewards.stand_pair_force_tolerance',
        'rewards.stand_pair_force_epsilon_n','rewards.stand_min_load_fraction'}
    assert set(diff)==expected,diff
    assert cfg.rewards.base_height_target==.30 and cfg.rewards.scales.base_height==-30.
    assert old.rewards.scales.base_height==-10.
    cfg.env.num_envs=6;cfg.env.episode_length_s=20.;cfg.commands.resampling_time=1e6
    cfg.noise.add_noise=False;cfg.sim.use_gpu_pipeline=cfg.sim.physx.use_gpu=False;cfg.sim.physx.num_threads=2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
    run=Path(json.loads((ROOT/'media/v2_gait_20261025/latest_training.json').read_text())['run_dir'])
    checkpoint=run/'model_200.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest=='fab8580057253836256d4d41743176070dd3fca9c0cf4aa718871f5d337d85a2'
    env=runner=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
        assert type(env) is LZQ1LV2GaitHeightRobot
        for term in ('stand_pair_force_balance','stand_low_load'):
            assert abs(env.reward_scales[term]-getattr(cfg.rewards.scales,term)*env.dt)<1e-8
        rules=checks(env);rules['parent_gait_checks']=gait_invariants(env)
        print('RULE_CHECKS_OK',rules,flush=True)
        runner=HIMOnPolicyRunner(env,class_to_dict(train),None,device='cpu')
        runner.alg.actor_critic.load_state_dict(torch.load(checkpoint,map_location='cpu')['model_state_dict'],strict=True)
        policy=runner.get_inference_policy(device='cpu');obs,_=env.reset();alive=torch.ones(6,dtype=torch.bool)
        traces={};resets=[]
        with torch.no_grad():
            for step in range(250):
                env.commands.zero_();env.command_sample_mode.zero_()
                if step>=100:env.commands[3:,0]=.35;env.command_sample_mode[3:]=1
                obs[:,:3]=env.commands[:,:3]*env.commands_scale
                obs,_,reward,done,_,_,_=env.step(policy(obs))
                for i in (alive&done.bool()).nonzero().flatten().tolist():resets.append(dict(index=i,step=step))
                alive &= ~done.bool()
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                if step>=100:assert not env.stand_force_penalties[3:].any()
                assert not env.gait_penalties[:3].any()
                for key,value in dict(valid=alive,standing_cost=env.stand_force_penalties,walking_cost=env.gait_penalties,
                    force=env.contact_forces[:,env.feet_indices,2],height=env.root_states[:,2],vx=env.base_lin_vel[:,0]).items():
                    traces.setdefault(key,[]).append(value.cpu().numpy().copy())
        arrays={k:np.asarray(v) for k,v in traces.items()};np.savez_compressed(out/'traces.npz',**arrays)
        summary=[]
        for label,ids in [('standing',slice(0,3)),('forward',slice(3,6))]:
            valid=arrays['valid'][150:,ids]
            costs=arrays['standing_cost'][150:,ids][valid]
            assert len(costs)>0
            summary.append(dict(command=label,valid_samples=int(valid.sum()),
                raw_cost_mean=costs.mean(0).tolist(),weighted_cost_mean=(costs*np.array([-.5,-.2])).mean(0).tolist(),
                mean_force_n=arrays['force'][150:,ids][valid].mean(0).tolist()))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
        sources=[Path(__file__).resolve()]+[ROOT/p for p in ('HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py','HIMLoco/legged_gym/legged_gym/envs/__init__.py')]
        report=dict(checks=rules,config_diff=diff,checkpoint=str(checkpoint),checkpoint_sha256=digest,
            checkpoint_unchanged=True,training_updates=0,finite=True,resets=resets,cases=summary,
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('COMPLETE',json.dumps(report),flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
