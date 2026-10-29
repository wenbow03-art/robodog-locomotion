"""Two-arm pilot: fixed V8 action std versus std ramped to half over 50 updates.

Both arms train actor, critic and estimator, keep V8 environment rewards and
command distribution, reset optimizers, and use the same fixed learning rate.
Std is fixed within each rollout AND its PPO update. Evaluation is a separate
process and cannot mutate the training simulator or random-number generators.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import isaacgym  # noqa: F401
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from calibrate_lz_q1l_estimator import SOURCE, ROOT, write_json


def noise_factor(arm, update_index):
    if not 0 <= update_index < 50:
        raise ValueError(update_index)
    return 1. if arm == 'fixed' else 1. - .5 * update_index / 49.


def external_evaluation(checkpoint, out_dir, seeds=(1, 2)):
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(ROOT / 'tools/eval_lz_q1l_checkpoint_pair.py'),
               '--checkpoint', str(checkpoint), '--out-dir', str(out_dir),
               '--seeds', *[str(seed) for seed in seeds]]
    with (out_dir / 'eval.log').open('w') as log:
        subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
    return json.loads((out_dir / 'assessment.json').read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=['fixed', 'annealed'])
    opts = parser.parse_args()
    out = opts.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Refusing to reuse an existing arm')
    started = time.time()
    manifest = dict(arm=opts.arm, source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    training_seed=101, num_envs=1024, steps_per_update=100, total_updates=50,
                    total_training_steps=5120000, eval_every=10, evaluation_seeds=[1, 2],
                    learning_rate=1e-4, learning_rate_schedule='fixed', load_optimizer=False,
                    std_parameter_trainable=False, noise_factors=[noise_factor(opts.arm, i) for i in range(50)],
                    entropy_coef=0., reason_entropy_zero='std is externally fixed; Gaussian entropy is independent of actor mean in both arms',
                    environment='V8 rewards, commands, curriculum and domain randomization; only num_envs and training seed differ from V8',
                    pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    limitation='One training seed, small-budget pilot; nominal deterministic evaluation does not establish sim-to-real robustness.')
    (out / 'provenance').mkdir()
    paths = [Path(__file__), ROOT / 'env.sh', ROOT / 'tools/run_lz_q1l_noise_ablation.sh',
             ROOT / 'tools/eval_lz_q1l_checkpoint_pair.py', ROOT / 'tools/calibrate_lz_q1l_estimator.py',
             ROOT / 'HIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py',
             ROOT / 'HIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py',
             ROOT / 'HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py',
             ROOT / 'HIMLoco/rsl_rl/rsl_rl/storage/him_rollout_storage.py',
             ROOT / 'HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
             ROOT / 'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
             ROOT / 'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py',
             ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py',
             ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
             ROOT / 'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf']
    manifest['code_sha256s'] = {}
    for path in paths:
        manifest['code_sha256s'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, out / 'provenance' / path.name)
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'status.json', dict(state='initializing', updates=0))
    env = writer = None
    try:
        torch.manual_seed(0)
        witness = torch.randn(8, 8, device='cuda:0')
        assert torch.isfinite(witness @ witness).all()
        print('CUDA_WITNESS', torch.cuda.get_device_name(0), torch.__version__, flush=True)
        sys.argv = [sys.argv[0], '--task', 'lz_q1l_flat_v8', '--headless', '--seed', '101',
                    '--num_envs', '1024', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0']
        args = get_args()
        cfg, train_cfg = task_registry.get_cfgs(args.task)
        cfg.env.num_envs = 1024
        original_cfg = json.loads((SOURCE.parent / 'env_config.json').read_text())
        for key in ['rewards', 'commands', 'control', 'domain_rand']:
            assert class_to_dict(getattr(cfg, key)) == original_cfg[key], key
        train_cfg.seed = 101
        cfg.seed = 101
        train_cfg.algorithm.learning_rate = 1e-4
        train_cfg.algorithm.schedule = 'fixed'
        train_cfg.algorithm.entropy_coef = 0.
        train_cfg.runner.num_steps_per_env = 100
        train_cfg.runner.resume = False
        train_cfg.runner.max_iterations = 50
        write_json(out / 'env_config.json', class_to_dict(cfg))
        write_json(out / 'train_config.json', class_to_dict(train_cfg))
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        runner = HIMOnPolicyRunner(env, class_to_dict(train_cfg), None, device=args.rl_device)
        source = torch.load(SOURCE, map_location=env.device)
        actor = runner.alg.actor_critic
        actor.load_state_dict(source['model_state_dict'], strict=True)
        actor.std.requires_grad_(False)
        initial_std = actor.std.detach().clone()
        assert torch.all(initial_std > 0)
        actor.train()
        env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
        obs = env.get_observations()
        critic_obs = env.get_privileged_observations()
        writer = SummaryWriter(str(out / 'tensorboard'))
        metrics, assessments = [], []
        for it in range(50):
            tick = time.time()
            factor = noise_factor(opts.arm, it)
            with torch.no_grad():
                actor.std.copy_(initial_std * factor)
            episode_terms = {}
            failures = timeouts = 0
            reward_sum = 0.
            with torch.inference_mode():
                for step in range(100):
                    actions = runner.alg.act(obs, critic_obs)
                    obs, critic_obs, rewards, dones, infos, term_ids, term_obs = env.step(actions)
                    next_critic_obs = critic_obs.clone()
                    next_critic_obs[term_ids] = term_obs
                    runner.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                    failures += int((dones.bool() & ~env.time_out_buf).sum())
                    timeouts += int((dones.bool() & env.time_out_buf).sum())
                    reward_sum += float(rewards.mean())
                    for key, value in infos.get('episode', {}).items():
                        episode_terms.setdefault(key, []).append(float(value))
                runner.alg.compute_returns(critic_obs)
                storage = runner.alg.storage
                assert torch.allclose(storage.sigma, (initial_std * factor).view(1, 1, -1).expand_as(storage.sigma))
                # Evaluate behavior likelihood before any updates: ratio must be 1.
                # Keep the original batch shape: changing GEMM shapes can change
                # TF32 rounding and falsely look like a changed behavior policy.
                actor.update_distribution(storage.observations[0])
                log_prob = actor.get_actions_log_prob(storage.actions[0])
                mismatch = float((log_prob - storage.actions_log_prob[0, :, 0]).abs().max())
                if mismatch > 1e-3:
                    raise RuntimeError('Behavior likelihood mismatch: %g' % mismatch)
            collection_seconds = time.time() - tick
            losses = runner.alg.update()
            assert torch.equal(actor.std.detach(), initial_std * factor)
            assert all(torch.isfinite(p).all() for p in actor.parameters())
            if not all(math.isfinite(float(x)) for x in losses):
                raise RuntimeError('Nonfinite training loss')
            row = dict(update=it+1, factor=factor, mean_std=float(actor.std.mean()),
                       value_loss=float(losses[0]), surrogate_loss=float(losses[1]),
                       estimator_last_minibatch_loss=float(losses[2]), swap_last_minibatch_loss=float(losses[3]),
                       mean_reward_per_step=reward_sum / 100., failures=failures, timeouts=timeouts,
                       collection_seconds=collection_seconds, update_seconds=time.time()-tick,
                       behavior_log_prob_max_error=mismatch,
                       episode={k: float(np.mean(v)) for k, v in episode_terms.items()})
            metrics.append(row)
            for key in ['factor', 'mean_std', 'value_loss', 'surrogate_loss', 'estimator_last_minibatch_loss',
                        'mean_reward_per_step', 'failures', 'timeouts']:
                writer.add_scalar(key, row[key], it+1)
            for key, value in row['episode'].items():
                writer.add_scalar('Episode/' + key, value, it+1)
            write_json(out / 'training_metrics.json', metrics)
            write_json(out / 'status.json', dict(state='training', updates=it+1, noise_factor=factor))
            print('UPDATE', opts.arm, json.dumps({k:v for k,v in row.items() if k != 'episode'}), flush=True)
            if (it + 1) % 10 == 0:
                checkpoint = out / ('model_%03d.pt' % (it+1))
                runner.current_learning_iteration = it+1
                runner.save(str(checkpoint), infos=dict(parent=str(SOURCE), arm=opts.arm,
                                                      completed_updates=it+1, noise_factor=factor))
                # Ensure both halves of the intended intervention actually trained.
                state = actor.state_dict()
                assert any(not torch.equal(state[k], source['model_state_dict'][k]) for k in state if k.startswith('actor.'))
                assert any(not torch.equal(state[k], source['model_state_dict'][k]) for k in state if k.startswith('estimator.encoder.'))
                write_json(out / 'status.json', dict(state='evaluating', updates=it+1, noise_factor=factor))
                writer.flush()
                assessment = external_evaluation(checkpoint, out / ('eval_%03d' % (it+1)))
                assessments.append(dict(update=it+1, noise_factor=factor, **assessment))
                write_json(out / 'assessments.json', assessments)
                print('ASSESSMENT', opts.arm, it+1, json.dumps(assessment), flush=True)
        eligible = [a for a in assessments if a['relative_gate_pass']]
        selected = max(eligible, key=lambda a:a['left_mae_improvement_mean']) if eligible else None
        confirmation = None
        if selected:
            confirmation = external_evaluation(Path(selected['checkpoint']), out / 'confirmation', seeds=(3, 4))
        unchanged = hashlib.sha256(SOURCE.read_bytes()).hexdigest() == manifest['source_sha256']
        assert unchanged
        outcome = dict(state='completed', arm=opts.arm, updates=50, elapsed_seconds=time.time()-started,
                       source_unchanged=unchanged, selected_candidate=selected, confirmation=confirmation,
                       promote=bool(selected and confirmation['relative_gate_pass']), assessments=assessments)
        write_json(out / 'result.json', outcome)
        write_json(out / 'status.json', outcome)
        print('ARM_COMPLETE', opts.arm, flush=True)
    except BaseException as error:
        write_json(out / 'status.json', dict(state='failed', error=repr(error)))
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
