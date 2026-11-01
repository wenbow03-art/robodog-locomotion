"""Train the approved V1 support variant from scratch, exactly 500 updates."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TASK = 'lz_q1l_flat_v1_support'


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def differences(before, after, prefix=''):
    result = {}
    for key in sorted(set(before) | set(after)):
        path = prefix + key
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            result.update(differences(left, right, path + '.'))
        elif left != right:
            result[path] = {'v1': left, 'this_run': right}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--preflight', action='store_true')
    opts = parser.parse_args()
    if not opts.preflight and opts.run_dir is None:
        parser.error('--run-dir is required for training')
    import isaacgym  # Must precede torch.
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners import HIMOnPolicyRunner

    cfg, train = task_registry.get_cfgs(TASK)
    v1_cfg, v1_train = task_registry.get_cfgs('lz_q1l_flat')
    env_diff = differences(class_to_dict(v1_cfg), class_to_dict(cfg))
    train_diff = differences(class_to_dict(v1_train), class_to_dict(train))
    expected_env = {
        'sim.dt', 'control.decimation', 'domain_rand.action_delay_steps',
        'commands.zero_command_prob', 'commands.pure_forward_prob', 'commands.pure_forward_speed_range',
        'rewards.stand_command_threshold', 'rewards.stand_support_grace_s', 'rewards.stand_support_air_time_s',
        'rewards.stand_support_height_m', 'rewards.stand_support_contact_force_n', 'rewards.scales.stand_unsupported'}
    assert set(env_diff) == expected_env, env_diff
    assert set(train_diff) == {'runner.experiment_name', 'runner.max_iterations', 'runner.save_interval', 'runner.log_directory'}, train_diff
    assert train.runner.max_iterations == 500 and train.runner.save_interval == 50
    assert cfg.env.num_envs == 4096 and train.runner.num_steps_per_env == 100
    assert train.seed == 1 and not train.runner.resume
    assert cfg.sim.dt == .001 and cfg.control.decimation == 20
    print('CONFIG_AUDIT', json.dumps({'env': env_diff, 'training': train_diff}), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started')
    torch.manual_seed(1)
    x = torch.randn(8, 8, device='cuda:0')
    y = x @ x.T
    assert torch.isfinite(y).all()
    torch.cuda.synchronize()
    print('CUDA_WITNESS', torch.cuda.get_device_name(0), tuple(y.shape), float(y.sum()), flush=True)
    sys.argv = [sys.argv[0], '--task', TASK, '--headless', '--seed', '1',
        '--sim_device', 'cuda:0', '--rl_device', 'cuda:0']
    args = get_args()
    out, runner, env = None, None, None
    completed = 0
    started = time.time()
    try:
        if opts.preflight:
            # Diagnostic size only; production cfg above is audited at 4096 envs.
            cfg.env.num_envs = args.num_envs = 32
            env, _ = task_registry.make_env(TASK, args=args, env_cfg=cfg)
            assert env.device == 'cuda:0' and env.sim_params.physx.use_gpu
            obs, _ = env.reset()
            with torch.no_grad():
                for _ in range(100):
                    obs, privileged, reward, _, _, _, _ = env.step(torch.zeros((32, 12), device=env.device))
                    assert torch.isfinite(obs).all() and torch.isfinite(privileged).all() and torch.isfinite(reward).all()
                    assert torch.isfinite(env.support_foot_gap).all()
            print('GPU_SIM_WITNESS', json.dumps({'envs': 32, 'steps': 100, 'dt': cfg.sim.dt,
                'decimation': cfg.control.decimation, 'finite': True,
                'peak_torch_memory_mb': torch.cuda.max_memory_allocated() / 2**20}), flush=True)
            return

        out = opts.run_dir.resolve()
        if any(out.glob('model_*.pt')) or any(out.glob('events.out.tfevents.*')):
            raise RuntimeError('Refusing to reuse a training run')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'status.json', {'state': 'initializing', 'completed_updates': 0, 'target_updates': 500})
        write_json(out / 'config_diff_from_v1.json', {'env': env_diff, 'training': train_diff})
        write_json(out / 'env_config.json', class_to_dict(cfg))
        write_json(out / 'train_config.json', class_to_dict(train))
        sources = [Path(__file__).resolve(), ROOT / 'tools/train_lz_q1l_v1_support_500.sh']
        sources += [ROOT / name for name in (
            'HIMLoco/legged_gym/legged_gym/envs/__init__.py',
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py',
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
            'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
            'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py',
            'HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
            'HIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py',
            'HIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py')]
        hashes = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            target = out / 'provenance' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata = dict(task=TASK, source_checkpoint=None, initialization='random_from_scratch',
            total_updates=500, save_interval=50, checkpoint_iterations=list(range(50, 501, 50)),
            num_envs=4096, rollout_steps_per_update=100, seed=1, pid=os.getpid(),
            gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), source_sha256=hashes,
            baseline='Current original V1 configuration; all differences audited and stored',
            started_unix=started, skill='/home/xiaohengli/.codex/skills/run-experiment/SKILL.md')
        write_json(out / 'run_metadata.json', metadata)
        env, _ = task_registry.make_env(TASK, args=args, env_cfg=cfg)
        runner = HIMOnPolicyRunner(env, class_to_dict(train), str(out), device=args.rl_device)
        assert runner.current_learning_iteration == 0 and env.num_envs == 4096
        original_log, original_save = runner.log, runner.save

        def log_and_checkpoint(locs):
            nonlocal completed
            for name in ('mean_value_loss', 'mean_surrogate_loss', 'mean_estimation_loss', 'mean_swap_loss'):
                if not math.isfinite(float(locs[name])):
                    raise RuntimeError('Nonfinite training loss: ' + name)
            original_log(locs)
            completed = locs['it'] + 1
            if completed % 50 == 0:
                path = out / ('model_%d.pt' % completed)
                if path.exists():
                    raise RuntimeError('Refusing to overwrite ' + str(path))
                previous = runner.current_learning_iteration
                runner.current_learning_iteration = completed
                try:
                    original_save(str(path) + '.tmp', infos=metadata)
                    os.replace(str(path) + '.tmp', path)
                finally:
                    runner.current_learning_iteration = previous
                runner.writer.flush()
                print('CHECKPOINT_SAVED', path, flush=True)
            elapsed = time.time() - started
            write_json(out / 'status.json', dict(state='running', completed_updates=completed,
                target_updates=500, elapsed_s=elapsed, estimated_remaining_s=elapsed / completed * (500 - completed),
                last_update_s=float(locs['collection_time'] + locs['learn_time']),
                mean_reward=float(sum(locs['rewbuffer']) / len(locs['rewbuffer'])) if len(locs['rewbuffer']) else None))

        runner.log = log_and_checkpoint
        # The legacy saver uses zero-based loop indices; the hook saves after
        # exactly 50/100/.../500 completed updates, with matching metadata.
        runner.save = lambda *args, **kwargs: None
        runner.learn(num_learning_iterations=500, init_at_random_ep_len=True)
        assert runner.current_learning_iteration == 500 and completed == 500
        checkpoint_checks = []
        for iteration in range(50, 501, 50):
            path = out / ('model_%d.pt' % iteration)
            state = torch.load(path, map_location='cpu')
            assert state['iter'] == iteration
            assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
            checkpoint_checks.append(dict(iteration=iteration, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            del state
        runner.writer.flush()
        write_json(out / 'status.json', dict(state='completed', completed_updates=500, target_updates=500,
            elapsed_s=time.time() - started, checkpoints=checkpoint_checks))
        print('TRAINING_COMPLETE', out, flush=True)
    except BaseException as exc:
        if out is not None:
            write_json(out / 'status.json', dict(state='failed', completed_updates=completed, target_updates=500, error=repr(exc)))
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
