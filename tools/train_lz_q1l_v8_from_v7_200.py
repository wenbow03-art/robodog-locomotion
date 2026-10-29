"""Fine-tune V7 model_200 under V8 rewards for exactly 200 new updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from export_v8_training_curves import export_curves

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'HIMLoco/legged_gym/logs1/flat_lz_q1l_v7/Oct13_11-20-32_v7_ft_v2_200_200iters_20261013_111930/model_200.pt'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--curves-dir', type=Path, default=Path('/home/xiaohengli/Program/training-curves'))
    opts = parser.parse_args()
    out = opts.run_dir.resolve()
    if any(out.glob('model_*.pt')) or any(out.glob('events.out.tfevents.*')):
        raise RuntimeError('Refusing to reuse an existing training run')
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'task': 'lz_q1l_flat_v8', 'source_checkpoint': str(SOURCE),
                'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'additional_iterations': 200, 'save_interval': 50,
                'checkpoint_iterations': [50, 100, 150, 200],
                'load_optimizer': False, 'iteration_numbering': 'new V8 updates starting at zero',
                'curves_directory': str(opts.curves_dir), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'seed': 1, 'pid': os.getpid()}
    write_json(out / 'run_metadata.json', metadata)
    write_json(out / 'status.json', {'state': 'initializing', 'completed_updates': 0, 'target_updates': 200})
    env = runner = None
    try:
        import isaacgym  # Must precede torch.
        import torch
        import legged_gym
        import rsl_rl
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        from legged_gym.utils.helpers import class_to_dict
        from rsl_rl.runners import HIMOnPolicyRunner

        for module, expected in [(legged_gym, ROOT / 'HIMLoco/legged_gym'),
                                 (rsl_rl, ROOT / 'HIMLoco/rsl_rl')]:
            actual = Path(module.__file__).resolve()
            if expected.resolve() not in actual.parents:
                raise RuntimeError('Unexpected import: ' + str(actual))
        print('IMPORT_PATHS', legged_gym.__file__, rsl_rl.__file__, flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; training has not started')
        checkpoint = torch.load(str(SOURCE), map_location='cpu')
        assert checkpoint['iter'] == 200
        sys.argv = [sys.argv[0], '--task', 'lz_q1l_flat_v8', '--headless', '--seed', '1',
                    '--max_iterations', '200', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0']
        args = get_args()
        cfg, train_cfg = task_registry.get_cfgs(args.task)
        assert cfg.commands.pure_lateral_prob == 0.25
        assert cfg.commands.pure_lateral_speed_range == [0.25, 0.45]
        assert cfg.rewards.scales.lateral_error == -0.5
        train_cfg.runner.max_iterations = 200
        train_cfg.runner.save_interval = 50
        train_cfg.runner.resume = False
        train_cfg.runner.run_name = out.name
        write_json(out / 'env_config.json', class_to_dict(cfg))
        write_json(out / 'train_config.json', class_to_dict(train_cfg))
        env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        runner = HIMOnPolicyRunner(env, class_to_dict(train_cfg), str(out), device=args.rl_device)
        runner.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'], strict=True)
        assert runner.current_learning_iteration == 0
        print('RUN_CONFIG ' + json.dumps(metadata), flush=True)
        original_log, original_save = runner.log, runner.save

        def log_and_checkpoint(locs):
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
                print('CHECKPOINT_SAVED', path, flush=True)
            write_json(out / 'status.json', {'state': 'running', 'completed_updates': completed,
                                           'target_updates': 200})

        runner.log = log_and_checkpoint
        runner.save = lambda *args, **kwargs: None
        runner.learn(num_learning_iterations=200, init_at_random_ep_len=True)
        assert runner.current_learning_iteration == 200
        for step in (50, 100, 150, 200):
            saved = torch.load(str(out / ('model_%d.pt' % step)), map_location='cpu')
            assert saved['iter'] == step
        runner.writer.flush()
        runner.writer.close()
        write_json(out / 'status.json', {'state': 'exporting_curves', 'completed_updates': 200})
        artifacts = export_curves(out, opts.curves_dir)
        write_json(out / 'status.json', {'state': 'completed', 'completed_updates': 200,
                                       'target_updates': 200, 'artifacts': artifacts})
        print('TRAINING_COMPLETE ' + json.dumps(artifacts), flush=True)
    except BaseException as exc:
        previous = json.loads((out / 'status.json').read_text())
        previous.update(state='failed', error=repr(exc))
        write_json(out / 'status.json', previous)
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
