"""One-sided left-load reward invariants and frozen gait150 CPU smoke test; no training."""
import hashlib
import json
from pathlib import Path
import sys

import isaacgym
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2SideLoadRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from check_lz_q1l_v2_gait import invariants as gait_invariants
from train_lz_q1l_v1_support_500 import differences

ROOT=Path(__file__).resolve().parents[1]


def checks(env):
    cfg=env.cfg.rewards
    def value(x):
        return env.side_force_penalty_value(torch.tensor(x,dtype=torch.float,device=env.device),cfg)
    assert not value([[80,80],[60,100],[0,0],[0,1]]).any()
    vals=value([[80,80],[88,72],[96,64],[104,56],[160,0]])
    assert (vals[1:]>vals[:-1]).all()
    assert torch.allclose(value([[88,72]]),torch.tensor([(16/161)**2],device=env.device))
    assert value([[10,0]])[0]>0 and value([[0,10]])[0]==0
    assert torch.equal(value([[-10,30]]),value([[0,30]]))
    assert torch.isfinite(value([[0,0],[1e-8,0]])).all()
    ids=torch.arange(env.num_envs,device=env.device)
    foot_ids=env.feet_indices[env.side_foot_slots]  # LH, LF, RH, RF
    def fill(v=(50.,50.,30.,30.)):
        env.contact_forces.zero_()
        env.contact_forces[:,foot_ids,2]=torch.tensor(v,device=env.device)
    for command in ([0,0,0],[.35,0,0]):
        env.commands[:,:3]=torch.tensor(command,device=env.device)
        env._clear_side_force(ids);fill()
        for i in range(env.side_force_window_steps):
            env._update_side_force_rewards()
            if i<env.side_force_window_steps-1:assert not env.side_force_penalty.any()
        assert torch.allclose(env.side_force_mean,torch.tensor([100.,60.],device=env.device).expand(env.num_envs,-1))
        assert (env._reward_left_load_excess()>0).all()
    history=env.side_force_history.clone();steps=env.side_force_steps.clone();cursor=env.side_force_cursor
    for _ in range(3):env._reward_left_load_excess()
    assert torch.equal(history,env.side_force_history) and torch.equal(steps,env.side_force_steps) and cursor==env.side_force_cursor
    env.reset_idx(ids[:1]);assert not env.side_force_history[:,0].any()
    assert torch.equal(history[:,1:],env.side_force_history[:,1:])
    env._resample_commands(ids[1:2]);assert not env.side_force_history[:,1].any()
    assert torch.equal(history[:,2:],env.side_force_history[:,2:])
    for command in ([0,.2,0],[0,0,.3]):
        env.commands[:,:3]=torch.tensor(command,device=env.device);env._update_side_force_rewards()
        assert not env.side_force_penalty.any() and not env.side_force_sum.any()
    env.commands.zero_();fill();env._update_side_force_rewards()
    assert (env.side_force_steps==1).all() and not env.side_force_penalty.any()
    # Directly check window wrap-around and per-side sums.
    env._clear_side_force(ids);samples=[]
    for i in range(2*env.side_force_window_steps+7):
        fill((float(i+1),float(i+1),1.,1.));samples.append(float(i+1))
        env._update_side_force_rewards()
        if len(samples)>=env.side_force_window_steps:
            expected=torch.tensor([2*np.mean(samples[-env.side_force_window_steps:]),2.],device=env.device,dtype=torch.float)
            assert torch.allclose(env.side_force_mean,expected.expand(env.num_envs,-1))
    # Alternating full left/right support is balanced AFTER time averaging.
    env._clear_side_force(ids)
    for i in range(env.side_force_window_steps):
        fill((80.,80.,0.,0.) if i%2 else (0.,0.,80.,80.))
        env._update_side_force_rewards()
    assert not env.side_force_penalty.any()
    # Right-biased measured forces must produce exactly zero, even at low total load.
    env._clear_side_force(ids);fill((0.,0.,.5,.5))
    for _ in range(env.side_force_window_steps):env._update_side_force_rewards()
    assert not env.side_force_penalty.any()
    env.reset()
    return dict(left_only=True,right_heavy_zero=True,monotonic=True,squared_formula=True,
        zero_total_safe=True,alternating_support_allowed=True,window_ready=True,
        rolling_window=True,partial_reset_isolated=True,resample_clears=True,
        command_change_clears=True,stand_and_forward_enabled=True,other_commands_disabled=True,
        named_foot_grouping=True,reward_getters_pure=True)


def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    sys.argv=[sys.argv[0],'--task','lz_q1l_flat_v2_side_load_strong','--headless','--seed','1',
              '--sim_device','cpu','--rl_device','cpu','--num_envs','6']
    args=get_args();cfg,train=task_registry.get_cfgs(args.task)
    old,_=task_registry.get_cfgs('lz_q1l_flat_v2_gait')
    diff=differences(class_to_dict(old),class_to_dict(cfg))
    expected={'rewards.scales.left_load_excess','rewards.side_force_window_s','rewards.side_force_epsilon_n'}
    assert set(diff)==expected,diff
    assert cfg.rewards.base_height_target==.30 and cfg.rewards.scales.base_height==-10.
    assert old.rewards.scales.base_height==-10.
    archived=json.loads((ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_gait/gait_from_posture500_200_20261025_113614_3235946/env_config.json').read_text())
    archived['domain_rand'].pop('push_interval',None)
    assert class_to_dict(old)==archived
    assert cfg.rewards.scales.orientation==-.2
    assert train.runner.max_iterations==300 and train.runner.save_interval==10
    assert cfg.env.num_envs==4096 and train.runner.num_steps_per_env==100
    assert cfg.commands.zero_command_prob==cfg.commands.pure_forward_prob==.5
    cfg.env.num_envs=6;cfg.env.episode_length_s=20.;cfg.commands.resampling_time=1e6
    cfg.noise.add_noise=False;cfg.sim.use_gpu_pipeline=cfg.sim.physx.use_gpu=False;cfg.sim.physx.num_threads=2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
    from train_lz_q1l_v2_side_load_strong import SOURCE
    checkpoint=SOURCE;digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest=='655a5022f2e1404c8f7c3dfbb670cc8235710f78e7550e72f2343e4e2c5e9747'
    env=runner=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
        assert type(env) is LZQ1LV2SideLoadRobot
        for term in ('left_load_excess',):
            assert abs(env.reward_scales[term]-getattr(cfg.rewards.scales,term)*env.dt)<1e-8
        rules=checks(env);rules['parent_gait_checks']=gait_invariants(env)
        print('RULE_CHECKS_OK',rules,flush=True)
        runner=HIMOnPolicyRunner(env,class_to_dict(train),None,device='cpu')
        saved=torch.load(checkpoint,map_location='cpu')
        runner.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
        runner.alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
        runner.alg.actor_critic.estimator.optimizer.load_state_dict(saved['estimator_optimizer_state_dict'])
        runner.current_learning_iteration=saved['iter']
        assert runner.current_learning_iteration==150
        assert runner.alg.optimizer.state and runner.alg.actor_critic.estimator.optimizer.state
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
                assert not env.gait_penalties[:3].any()
                for key,value in dict(valid=alive,side_cost=env.side_force_penalty[:,None],walking_cost=env.gait_penalties,
                    force=env.contact_forces[:,env.feet_indices,2],height=env.root_states[:,2],vx=env.base_lin_vel[:,0]).items():
                    traces.setdefault(key,[]).append(value.cpu().numpy().copy())
        arrays={k:np.asarray(v) for k,v in traces.items()};np.savez_compressed(out/'traces.npz',**arrays)
        summary=[]
        for label,ids in [('standing',slice(0,3)),('forward',slice(3,6))]:
            valid=arrays['valid'][150:,ids]
            costs=arrays['side_cost'][150:,ids][valid]
            assert len(costs)>0
            summary.append(dict(command=label,valid_samples=int(valid.sum()),
                raw_cost_mean=costs.mean(0).tolist(),weighted_cost_mean=(costs*np.array([-10.])).mean(0).tolist(),
                mean_force_n=arrays['force'][150:,ids][valid].mean(0).tolist()))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
        sources=[Path(__file__).resolve()]+[ROOT/p for p in ('HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py','HIMLoco/legged_gym/legged_gym/envs/__init__.py')]
        report=dict(checks=rules,config_diff=diff,checkpoint=str(checkpoint),checkpoint_sha256=digest,
            checkpoint_unchanged=True,training_updates=0,finite=True,resets=resets,cases=summary,
            optimizer_restore_checked=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('COMPLETE',json.dumps(report),flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
