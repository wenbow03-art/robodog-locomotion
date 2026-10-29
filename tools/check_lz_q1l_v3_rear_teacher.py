"""Frozen CPU replay and backward-only new-v3 checks. No parameter updates."""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import isaacgym
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.lz_q1l_mirror import LZQ1LMirror
from rsl_rl.runners import HIMOnPolicyRunner
from train_lz_q1l_v3_rear_teacher import (
    ROOT, REVIEW, TASK, SOURCE, SOURCE_SHA256, NEW_SOURCES, verify_baseline, attach_teacher)
from train_lz_q1l_v1_support_500 import differences


def state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def unchanged(module, before):
    return all(torch.equal(v, before[k]) for k, v in module.state_dict().items())


def rules(alg, env):
    results = {}
    mirror = alg.mirror
    obs = torch.randn(8, 270)
    original = obs.clone()
    assert torch.equal(mirror.history(mirror.history(obs)), obs)
    for frame in range(6):
        transformed = mirror.history(obs)[:, frame*45:(frame+1)*45]
        source = obs[:, frame*45:(frame+1)*45]
        assert torch.equal(transformed[:, :3], source[:, :3]*torch.tensor([1., -1., -1.]))
        assert torch.equal(transformed[:, 3:6], source[:, 3:6]*torch.tensor([-1., 1., -1.]))
        assert torch.equal(transformed[:, 6:9], source[:, 6:9]*torch.tensor([1., -1., 1.]))
        for start in (9, 21, 33):
            assert torch.equal(transformed[:, start:start+12], mirror.joints(source[:, start:start+12]))
    assert torch.equal(original, obs)
    results['all_six_frames_mirrored_without_mutation'] = True
    for names in (env.dof_names, list(reversed(env.dof_names))):
        m = LZQ1LMirror(names)
        values = torch.arange(12.).unsqueeze(0)
        reflected = m.joints(values)
        for kind, sign in [('roll', -1.), ('pitch01', 1.), ('pitch02', 1.)]:
            assert reflected[0, names.index(kind+'_leg_b_r_joint')] == sign*values[0, names.index(kind+'_leg_b_l_joint')]
    results['named_joint_mapping_and_signs'] = True
    rear, other = alg.teacher_targets(obs)
    with torch.no_grad():
        teacher_mirrored = alg.teacher.act_inference(mirror.history(obs))
        left_ids = [env.dof_names.index(k+'_leg_b_l_joint') for k in ('roll', 'pitch01', 'pitch02')]
        assert torch.equal(rear, teacher_mirrored[:, left_ids]*torch.tensor([-1., 1., 1.]))
        assert torch.equal(other, alg.teacher.act_inference(obs)[:, alg.other_indices])
    assert not rear.requires_grad and not other.requires_grad
    results['teacher_left_to_student_right_target'] = True
    commands = torch.tensor([[0,0,0],[.2,0,0],[.5,0,0],[0,.2,0],[0,0,.3],[-.3,0,0],[.3,.2,0],[.3,0,.3]])
    obs[:, :3] = commands*alg.command_scale
    assert alg.command_mask(obs).tolist() == [True,True,True,False,False,False,False,False]
    results['stand_forward_only'] = True
    saved_coeffs = alg.rear_teacher_coef, alg.other_legs_coef
    rear, other = alg.teacher_targets(obs)
    mean = torch.zeros(8,12)
    mean[:, alg.rear_indices] = rear+1.
    mean[:, alg.other_indices] = other+1.
    for coefficients, active, inactive in [((1.,0.), alg.rear_indices, alg.other_indices),
                                           ((0.,1.), alg.other_indices, alg.rear_indices)]:
        alg.rear_teacher_coef, alg.other_legs_coef = coefficients
        values = mean.clone().requires_grad_(True)
        loss, stats = alg.distillation_loss(obs, values)
        loss.backward()
        assert torch.allclose(loss, torch.tensor(1.))
        assert (values.grad[:3, active] > 0).all()
        assert not values.grad[:, inactive].any() and not values.grad[3:].any()
        assert stats['samples'] == 3
    alg.rear_teacher_coef, alg.other_legs_coef = saved_coeffs
    values = mean.clone().requires_grad_(True)
    loss, _ = alg.distillation_loss(obs, values)
    assert torch.allclose(loss, torch.tensor(.12))
    for coefficient in (0., .01, .02, .03, .04):
        alg.other_legs_coef = coefficient
        values = mean.clone().requires_grad_(True)
        loss, _ = alg.distillation_loss(obs, values)
        loss.backward()
        assert torch.allclose(loss, torch.tensor(.1+coefficient))
        assert (values.grad[:3, alg.rear_indices]>0).all()
        assert bool(values.grad[:3, alg.other_indices].abs().sum()>0) == (coefficient>0)
    alg.rear_teacher_coef, alg.other_legs_coef = saved_coeffs
    results['all_five_sweep_coefficients'] = True
    for zero_mode in ('no_commands', 'zero_weights'):
        x = obs[3:] if zero_mode == 'no_commands' else obs
        if zero_mode == 'zero_weights': alg.rear_teacher_coef = alg.other_legs_coef = 0.
        values = torch.ones(len(x),12,requires_grad=True)
        loss, stats = alg.distillation_loss(x, values); loss.backward()
        assert loss.item() == 0 and not values.grad.any()
    alg.rear_teacher_coef, alg.other_legs_coef = saved_coeffs
    results['weighted_mse_gradient_scope_and_zero_cases'] = True
    alg.actor_critic.update_distribution(obs)
    mu = alg.actor_critic.action_mean.clone()
    sigma = alg.actor_critic.action_std.clone()
    alg.teacher_targets(obs)
    assert torch.equal(mu, alg.actor_critic.action_mean) and torch.equal(sigma, alg.actor_critic.action_std)
    loss, _ = alg.distillation_loss(obs, alg.actor_critic.action_mean)
    alg.optimizer.zero_grad(); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in alg.actor_critic.actor.parameters())
    assert all(p.grad is None for p in alg.teacher.parameters())
    assert all(p.grad is None for p in alg.actor_critic.estimator.parameters())
    optimizer_ids = {id(p) for g in alg.optimizer.param_groups for p in g['params']}
    assert not optimizer_ids.intersection(id(p) for p in alg.teacher.parameters())
    assert all(not p.requires_grad for p in alg.teacher.parameters())
    results['actor_gradient_teacher_frozen_optimizer_separate'] = True
    results['teacher_does_not_replace_student_distribution'] = True
    return results


def main():
    verify_baseline()
    REVIEW.mkdir(parents=True, exist_ok=False)
    cfg, train = task_registry.get_cfgs(TASK)
    archived = json.loads((SOURCE.parent/'env_config.json').read_text())
    archived['domain_rand'].pop('push_interval', None)
    assert class_to_dict(cfg) == archived
    assert train.runner.max_iterations == 0
    train_diff = differences(json.loads((SOURCE.parent/'train_config.json').read_text()), class_to_dict(train))
    cfg.env.num_envs = 6; cfg.env.episode_length_s = 20.; cfg.commands.resampling_time = 1e6
    cfg.noise.add_noise = False
    cfg.sim.use_gpu_pipeline = cfg.sim.physx.use_gpu = False; cfg.sim.physx.num_threads = 2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key), bool): setattr(cfg.domain_rand,key,False)
    sys.argv = [sys.argv[0], '--task', TASK, '--headless', '--seed', '1', '--sim_device', 'cpu', '--rl_device', 'cpu', '--num_envs', '6']
    env = None
    try:
        env, _ = task_registry.make_env(TASK, args=get_args(), env_cfg=cfg)
        runner = HIMOnPolicyRunner(env,class_to_dict(train),None,device='cpu')
        alg = runner.alg
        saved = torch.load(SOURCE,map_location='cpu')
        alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
        alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
        alg.actor_critic.estimator.optimizer.load_state_dict(saved['estimator_optimizer_state_dict'])
        assert alg.optimizer.state and alg.actor_critic.estimator.optimizer.state
        attach_teacher(runner,env,train)
        before = state(alg.actor_critic); teacher_before = state(alg.teacher)
        checks = rules(alg, env)
        checks['baseline_environment_exactly_preserved'] = True
        checks['both_optimizer_states_restore'] = True
        policy = runner.get_inference_policy(device='cpu')
        obs, _ = env.reset(); traces = {}; resets = []
        with torch.no_grad():
            for step in range(250):
                env.commands.zero_(); env.command_sample_mode.zero_()
                if step>=100: env.commands[3:,0]=.35; env.command_sample_mode[3:]=1
                obs[:,:3] = env.commands[:,:3]*env.commands_scale
                actions = policy(obs)
                rear, other = alg.teacher_targets(obs)
                assert torch.allclose(other,actions[:,alg.other_indices],atol=1e-6)
                delta = rear-actions[:,alg.rear_indices]
                for key,value in dict(target=rear,original=actions[:,alg.rear_indices],delta=delta).items():
                    assert torch.isfinite(value).all()
                    traces.setdefault(key,[]).append(value.numpy().copy())
                obs,_,reward,done,_,_,_ = env.step(actions)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                for index in done.nonzero().flatten().tolist(): resets.append(dict(step=step,index=index))
        np.savez_compressed(REVIEW/'traces.npz',**{k:np.asarray(v) for k,v in traces.items()})
        cases=[]
        for label,ids in [('standing',slice(0,3)),('forward_0.35',slice(3,6))]:
            delta=np.asarray(traces['delta'])[150:,ids]
            cases.append(dict(command=label,mean_delta_deg=(delta.mean((0,1))*.25*180/np.pi).tolist(),
                rms_delta_deg=(np.sqrt((delta**2).mean((0,1)))*.25*180/np.pi).tolist(),
                weighted_rear_loss=float(.1*(delta**2).mean())))
        checks['frozen_replay_finite'] = True
        # Exercise the PPO integration/backward path only. Both updating routines
        # are intercepted: estimator.update also normalizes prototype weights in-place.
        alg.init_storage(6,4,[env.num_obs],[env.num_privileged_obs],[12])
        with torch.no_grad():
            for _ in range(4):
                critic=env.get_privileged_observations()
                actions=alg.act(obs.clone(),critic.clone())
                obs,next_critic,reward,done,info,_,_=env.step(actions)
                alg.process_env_step(reward,done,info,next_critic)
            alg.compute_returns(next_critic)
        alg.num_learning_epochs=alg.num_mini_batches=1
        with patch.object(alg.actor_critic.estimator,'update',return_value=(0.,0.)) as estimator_stub:
            with patch.object(alg.optimizer,'step',return_value=None) as step_stub:
                losses=alg.update()
                assert step_stub.call_count==1 and estimator_stub.call_count==1
        assert all(np.isfinite(losses))
        assert unchanged(alg.actor_critic,before) and unchanged(alg.teacher,teacher_before)
        checks['ppo_backward_only_with_updates_intercepted'] = True
        checks['student_teacher_parameters_unchanged'] = True
        # A frozen deployment policy works with no teacher attached.
        alg.teacher=None
        with torch.no_grad(): assert torch.isfinite(runner.get_inference_policy(device='cpu')(obs)).all()
        assert set(alg.actor_critic.state_dict())==set(saved['model_state_dict'])
        checks['inference_and_checkpoint_layout_need_no_teacher'] = True
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256
        sources=set(NEW_SOURCES)|set(json.loads((SOURCE.parent/'run_metadata.json').read_text())['code_sha256'])
        report=dict(checks=checks,training_updates=0,parameters_unchanged=True,
            checkpoint_sha256=SOURCE_SHA256,config_diff=dict(env={},training=train_diff),
            replay=dict(envs=6,steps=250,dt=env.dt,resets=resets,cases=cases),
            dry_backward=dict(optimizer_step_intercepted=True,estimator_update_stubbed=True,losses=losses),
            source_sha256={str(ROOT/p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(sources)})
        (REVIEW/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('V3_REVIEW_CHECKS_OK',json.dumps(dict(checks=checks,cases=cases,resets=resets,training_updates=0)),flush=True)
    finally:
        if env is not None: env.gym.destroy_sim(env.sim)


if __name__=='__main__': main()
