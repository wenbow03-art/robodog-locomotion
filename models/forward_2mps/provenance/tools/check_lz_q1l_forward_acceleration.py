"""Meaningful preflight checks: command timing, reward windows, PPO action path, gates."""
import copy
from types import SimpleNamespace
import torch
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LForwardAccelerationRobot
from legged_gym.utils.lz_q1l_acceleration import AccelerationCurriculum
from train_lz_q1l_forward_speed2 import rules


def check(env,runner,cfg):
    results=rules(runner,env)
    # All three rates reach the target at the configured times, independently of dt.
    for dt in [.02,.01]:
        n=3
        fake=SimpleNamespace(cfg=cfg,dt=dt,commands=torch.zeros(n,4),command_sample_mode=torch.zeros(n,dtype=torch.long),
                             ramp_rate=torch.tensor([.5,.75,1.]),ramp_target=torch.full((n,),2.))
        prev=torch.zeros(n)
        for tick in range(round(7/dt)):
            fake.episode_length_buf=torch.full((n,),tick,dtype=torch.long)
            LZQ1LForwardAccelerationRobot._next_ramp_commands(fake)
            current=fake.commands[:,0]
            assert (current>=prev-1e-6).all() and (current<=2).all()
            assert ((current-prev)<=fake.ramp_rate*dt+1e-6).all()
            if tick<round(2/dt):assert not current.any()
            prev=current.clone()
        assert torch.allclose(prev,torch.full((n,),2.))
    gate=AccelerationCurriculum(cfg.acceleration)
    def report(stage,passed=True):
        return dict(groups=[dict(acceleration=a,trials=[dict(passed=passed) for _ in range(cfg.acceleration.evaluation_trials)])
                            for a in cfg.acceleration.levels[:stage+1]])
    assert not gate.consider(50,report(0))['promoted']
    assert not gate.consider(100,report(0,False))['promoted'] and gate.streak==0
    assert not gate.consider(150,report(0))['promoted']
    assert gate.consider(200,report(0))['promoted'] and gate.stage==1
    assert not gate.consider(250,dict(error='evaluation failed',groups=[]))['promoted']
    assert not gate.consider(300,report(1))['promoted']
    saved=gate.state_dict();restored=AccelerationCurriculum(cfg.acceleration);restored.load_state_dict(saved)
    assert restored.consider(350,report(1))['promoted'] and restored.stage==2
    assert not restored.consider(400,report(2))['promoted']
    try:restored.consider(400,report(2))
    except ValueError:pass
    else:raise AssertionError('Duplicate evaluation accepted')
    # Real environment histories must span continuously changing speed commands.
    with torch.inference_mode():
        env.reset()
        ids=torch.arange(env.num_envs,device=env.device)
        env.commands.zero_();env.commands[:,0]=.2;env._clear_gait_state(ids)
        env.contact_forces[:,env.feet_indices[env.rear_foot_slots[0]],2]=40.
        env.contact_forces[:,env.feet_indices[env.rear_foot_slots[1]],2]=20.
        for step in range(80):
            env.commands[:,0]=.2+.01*step
            env._post_physics_step_callback()
        assert (env.gait_forward_steps==80).all()
        assert (env.width_tracker.samples==env.width_tracker.steps).all()
        assert (env._reward_rear_force_balance()>0).all()
        before=env.rear_force_history.clone();width_before=env.width_tracker.samples.clone()
        env.reset_idx(ids[:1])
        assert env.gait_forward_steps[0]==0 and env.width_tracker.samples[0]==0
        assert torch.equal(before[:,1:],env.rear_force_history[:,1:])
        assert torch.equal(width_before[1:],env.width_tracker.samples[1:])
        env.commands.zero_();env._post_physics_step_callback()
        assert not env.gait_forward_steps.any() and not env.gait_penalties.any()
        obs,priv=env.reset()
        raw=runner.alg.act(obs,priv);raw_before=raw.clone()
        expected_standing=env.ramp_standing_policy(obs).clone()
        obs,priv,reward,done,info,ids,terminal=env.step(raw)
        next_obs=priv.clone();next_obs[ids]=terminal
        runner.alg.process_env_step(reward,done,info,next_obs)
        assert torch.equal(raw,raw_before)
        assert torch.equal(runner.alg.storage.actions[0],raw_before)
        assert torch.allclose(env.actions,expected_standing)
        assert torch.allclose(obs[:,33:45],env.actions)
        assert torch.allclose(obs[:,:3],env.commands[:,:3]*env.commands_scale)
        assert torch.allclose(priv[:,:3],env.commands[:,:3]*env.commands_scale)
        runner.alg.storage.clear()
        # Frozen rollout includes the warmup, live ramp, full blend and hold.
        obs,_=env.reset()
        env.ramp_target[:]=.5
        env.ramp_rate[:]=.5
        resets=0;max_history=0
        for step in range(200):
            command_before=env.commands[:,0].clone()
            obs,_,reward,done,_,_,_=env.step(runner.alg.actor_critic.act_inference(obs))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert torch.allclose(env.ramp_step_metrics[:,0],command_before)
            assert torch.allclose(obs[:,:3],env.commands[:,:3]*env.commands_scale)
            resets+=int(done.sum());max_history=max(max_history,int(env.gait_forward_steps.max()))
        assert max_history>env.rear_force_window_steps, max_history
    results.update(command_rates=True,command_reward_alignment=True,continuous_reward_windows=True,
                   partial_reset_isolated=True,mode_reset=True,raw_ppo_actions_preserved=True,
                   executed_actions_in_observations=True,standing_blend=True,curriculum_gates=True,
                   curriculum_roundtrip=True,frozen_rollout_finite=True,witness_resets=resets)
    return results
