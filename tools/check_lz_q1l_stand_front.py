"""Validate standing posture gates and front-load history, then frozen GPU playback."""
import hashlib,json,sys
from pathlib import Path
import isaacgym
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2StandFrontRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from check_lz_q1l_v2_gait import invariants as gait_checks
from check_lz_q1l_stand_balance import checks as standing_checks
from train_lz_q1l_v2_stand_front import ROOT,SOURCE,SOURCE_SHA256,EXPECTED_ADDITIONS,REVIEW
from train_lz_q1l_v1_support_500 import differences


def checks(env):
    cfg=env.cfg.rewards;dev=env.device;ids=torch.arange(env.num_envs,device=dev)
    def t(x):return torch.tensor(x,dtype=torch.float,device=dev)
    def penalty(x):return env.front_force_penalty_value(t(x),cfg)
    assert not penalty([[40,40],[60,60],[41,39]]).any()
    assert penalty([[60,30]])[0]>0 and penalty([[0,0]])[0]==1
    assert torch.equal(penalty([[-1,30]]),penalty([[0,30]]))
    assert torch.equal(penalty([[60,30]]),penalty([[30,60]]))
    assert torch.isfinite(penalty([[0,0],[1e-8,0]])).all()
    front_ids=env.feet_indices[env.front_foot_slots]
    assert env.front_force_window_steps==50 and env.stand_posture_grace_steps==25
    def tick():
        env._update_gait_rewards();env._update_front_force_rewards()
    env.commands.zero_();env.commands[:,0]=.35;env._clear_gait_state(ids)
    # Alternating support averages to equal load; no instantaneous symmetry constraint.
    for i in range(50):
        env.contact_forces.zero_();env.contact_forces[:,front_ids[i%2],2]=80.;tick()
        if i<49:assert not env.front_force_penalty.any()
    assert torch.allclose(env.front_force_mean,torch.full_like(env.front_force_mean,40.))
    assert not env.front_force_penalty.any()
    env._clear_gait_state(ids)
    samples=[]
    for i in range(107):
        env.contact_forces[:,front_ids,2]=t([60+i,30]);samples.append([60+i,30]);tick()
        if i>=49:
            expected=t(np.mean(samples[-50:],axis=0).tolist())
            assert torch.allclose(env.front_force_mean,expected.expand(env.num_envs,-1))
    assert (env.front_force_penalty>0).all()
    history=env.front_force_history.clone();cursor=env.front_force_cursor
    for _ in range(3):env._reward_front_force_balance()
    assert torch.equal(history,env.front_force_history) and cursor==env.front_force_cursor
    env.reset_idx(ids[:1]);assert not env.front_force_history[:,0].any()
    assert torch.equal(history[:,1:],env.front_force_history[:,1:])
    env._resample_commands(ids[1:2]);assert not env.front_force_history[:,1].any()
    assert torch.equal(history[:,2:],env.front_force_history[:,2:])
    for cmd in ([0,0,0],[0,.2,0],[0,0,.3]):
        env.commands[:,:3]=t(cmd);tick()
        assert not env.front_force_penalty.any() and not env.front_force_sum.any()
    env.commands.zero_();env.commands[:,0]=.35;tick()
    assert (env.gait_forward_steps==1).all() and not env.front_force_penalty.any()
    env.commands[:,0]=.3;tick()
    assert (env.gait_forward_steps==1).all() and not env.front_force_penalty.any()
    # Stand-only extras have 0.5 s grace, no lower/upper-height asymmetry.
    env.commands.zero_();env._clear_stand_force(ids)
    env.root_states[:,2]=.34;env.projected_gravity[:]=t([.1,.2,-.974679434])
    for i in range(25):
        env._update_stand_force_rewards()
        if i<24:assert not env._reward_stand_height_extra().any() and not env._reward_stand_orientation_extra().any()
    assert torch.allclose(env._reward_stand_height_extra(),torch.full((env.num_envs,),.0016,device=dev),atol=1e-7)
    assert torch.allclose(env._reward_stand_orientation_extra(),torch.full((env.num_envs,),.05,device=dev))
    env.root_states[:,2]=.26
    assert torch.allclose(env._reward_stand_height_extra(),torch.full((env.num_envs,),.0016,device=dev),atol=1e-7)
    for cmd in ([.35,0,0],[0,.2,0],[0,0,.3]):
        env.commands[:,:3]=t(cmd)
        assert not env._reward_stand_height_extra().any() and not env._reward_stand_orientation_extra().any()
        env._update_stand_force_rewards()
    env.commands.zero_();env._update_stand_force_rewards()
    assert (env.stand_force_steps==1).all() and not env._reward_stand_height_extra().any()
    env.reset()
    return dict(front_symmetry=True,zero_load_guard=True,front_window=True,alternating_support_allowed=True,
        history_reset_and_resample=True,external_command_reset=True,getters_pure=True,
        stand_grace=True,height_above_and_below=True,standing_only_extras=True)


def main():
    args=get_args();assert args.task=='lz_q1l_flat_v2_stand_front'
    out=REVIEW;out.mkdir(parents=True,exist_ok=False)
    cfg,train=task_registry.get_cfgs(args.task)
    old,_=task_registry.get_cfgs('lz_q1l_flat_v2_upright')
    archived=json.loads((SOURCE.parent/'env_config.json').read_text());archived['domain_rand'].pop('push_interval',None)
    assert archived==class_to_dict(old)
    diff=differences(archived,class_to_dict(cfg))
    assert diff=={k:dict(v1=None,this_run=v) for k,v in EXPECTED_ADDITIONS.items()},diff
    assert train.runner.max_iterations==300 and train.runner.save_interval==50
    assert cfg.env.num_envs==4096 and train.runner.num_steps_per_env==100
    old_train=json.loads((SOURCE.parent/'train_config.json').read_text())
    assert set(differences(old_train,class_to_dict(train)))=={'runner.max_iterations','runner.experiment_name'}
    assert cfg.rewards.scales.base_height==-30 and cfg.rewards.scales.orientation==-1
    assert cfg.rewards.base_height_target==.30
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256
    cfg.env.num_envs=args.num_envs=6;cfg.env.episode_length_s=20.;cfg.commands.resampling_time=1e6
    cfg.noise.add_noise=False;cfg.sim.use_gpu_pipeline=cfg.sim.physx.use_gpu=args.sim_device.startswith('cuda');cfg.sim.physx.num_threads=2
    for k in dir(cfg.domain_rand):
        if not k.startswith('_') and isinstance(getattr(cfg.domain_rand,k),bool):setattr(cfg.domain_rand,k,False)
    env=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
        assert type(env) is LZQ1LV2StandFrontRobot
        for term in ('stand_height_extra','stand_orientation_extra','front_force_balance'):
            assert abs(env.reward_scales[term]-getattr(cfg.rewards.scales,term)*env.dt)<1e-8
        rules=checks(env);rules['parent_gait']=all(gait_checks(env).values());rules['parent_standing']=all(standing_checks(env).values())
        print('RULE_CHECKS_OK',rules,flush=True)
        runner=HIMOnPolicyRunner(env,class_to_dict(train),None,device=env.device)
        runner.load(str(SOURCE),load_optimizer=True)
        assert runner.current_learning_iteration==150 and runner.alg.optimizer.state and runner.alg.actor_critic.estimator.optimizer.state
        policy=runner.get_inference_policy(device=env.device)
        obs,_=env.reset();trace=[];resets=[]
        with torch.no_grad():
            for step in range(250):
                env.commands.zero_();env.command_sample_mode.zero_()
                if step>=100:env.commands[3:,0]=.35;env.command_sample_mode[3:]=1
                obs[:,:3]=env.commands[:,:3]*env.commands_scale
                obs,_,reward,done,_,_,_=env.step(policy(obs))
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                values=torch.stack([env._reward_stand_height_extra(),env._reward_stand_orientation_extra(),env._reward_front_force_balance()],dim=-1)
                assert torch.isfinite(values).all() and (values>=0).all()
                if step>=100:assert not values[3:,:2].any()
                assert not values[:3,2].any()
                trace.append(values.cpu().numpy().copy())
                for i in done.nonzero().flatten().tolist():resets.append(dict(index=i,step=step))
                if (step+1)%50==0:print('STEP',step+1,flush=True)
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256
        a=np.asarray(trace);np.savez_compressed(out/'traces.npz',costs=a)
        sources=[Path(__file__).resolve(),ROOT/'tools/train_lz_q1l_v2_stand_front.py']+[ROOT/p for p in (
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py','HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py','HIMLoco/legged_gym/legged_gym/envs/__init__.py')]
        result=dict(checks=rules,finite=True,training_updates=0,checkpoint_sha256=SOURCE_SHA256,checkpoint_unchanged=True,
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},config_diff=diff,resets=resets,
            saved_learning_rate=runner.alg.optimizer.param_groups[0]['lr'],
            mean_stand_cost=a[150:,:3].mean((0,1)).tolist(),mean_forward_cost=a[150:,3:].mean((0,1)).tolist())
        (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('COMPLETE',json.dumps(result),flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
