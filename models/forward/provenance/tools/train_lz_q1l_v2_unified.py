"""Explicit new-V2 fine-tuning entrypoint; never starts on import."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from train_lz_q1l_v1_support_500 import differences, write_json

ROOT = Path(__file__).resolve().parents[1]
TASK = 'lz_q1l_flat_v2_unified'
SOURCE = ROOT / 'HIMLoco/legged_gym/logs2/flat_lz_q1l_v1_support/v1_support_scratch_500_20261024_113050_2745173/model_500.pt'
SOURCE_SHA256 = 'efaf1f486d0264ec8539a8c4ef50b5a556f8fe818294ca6b65b631f3b5bcb3b1'


def load_new_v1_weights(runner):
    import torch
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('New V1 checkpoint does not match the reviewed source')
    state = torch.load(str(SOURCE), map_location=runner.device)
    assert state['iter'] == 500
    assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
    runner.alg.actor_critic.load_state_dict(state['model_state_dict'], strict=True)
    assert runner.current_learning_iteration == 0
    assert not runner.alg.optimizer.state
    assert not runner.alg.actor_critic.estimator.optimizer.state
    return state


def install_logging(runner, out, metadata, target, after_update=None):
    """Save after exact completed updates; retain terminal partial intervals."""
    original_log, original_save = runner.log, runner.save
    started = time.time()

    def log(locs):
        for key in ('mean_value_loss', 'mean_surrogate_loss', 'mean_estimation_loss', 'mean_swap_loss'):
            if not math.isfinite(float(locs[key])):
                raise RuntimeError('Nonfinite loss: ' + key)
        original_log(locs)
        completed = locs['it'] + 1
        diagnostics = runner.env.take_diagnostics()
        with (out / 'direction_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=completed, directions=diagnostics), allow_nan=False) + '\n')
        for direction, row in diagnostics.items():
            for key in ('velocity', 'abs_velocity_error', 'force_z_N', 'contact', 'gap_m',
                        'contact_slip_m_s', 'longest_noncontact_s', 'max_gap_m'):
                for i, value in enumerate(row[key]):
                    if value is not None:
                        runner.writer.add_scalar('Directions/%s/%s_%d' % (direction, key, i), value, completed)
            runner.writer.add_scalar('Directions/%s/falls_per_env_second' % direction,
                                     row['fall_events'] / row['exposure_env_seconds'], completed)
        if completed % metadata.get('save_interval', 50) == 0 or completed == target:
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
        write_json(out / 'status.json', dict(state='running', completed_updates=completed,
                   target_updates=target, elapsed_s=time.time() - started))
        if after_update is not None:
            after_update(completed)

    runner.log = log
    runner.save = lambda *args, **kwargs: None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, required=True, help='Number of NEW V2 updates; choose after review')
    parser.add_argument('--gpu', required=True, help='Physical GPU index or UUID chosen after checking free memory')
    parser.add_argument('--run-dir', type=Path, required=True, help='New directory under logs2/flat_lz_q1l_v2')
    opts = parser.parse_args()
    if opts.iterations <= 0:
        parser.error('--iterations must be positive')
    out = opts.run_dir.resolve()
    log_root = ROOT / 'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2'
    if log_root.resolve() not in out.parents or out.exists():
        parser.error('--run-dir must be a new subdirectory of logs2/flat_lz_q1l_v2')
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    import isaacgym  # Must precede torch.
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners import HIMOnPolicyRunner
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; training not started')
    cfg, train = task_registry.get_cfgs(TASK)
    baseline, baseline_train = task_registry.get_cfgs('lz_q1l_flat_v1_support')
    train.runner.max_iterations = opts.iterations
    sys.argv = [sys.argv[0], '--task', TASK, '--headless', '--seed', str(train.seed),
                '--sim_device', 'cuda:0', '--rl_device', 'cuda:0']
    args = get_args()
    out.mkdir(parents=True, exist_ok=False)
    env = runner = None
    try:
        write_json(out / 'status.json', dict(state='initializing', target_updates=opts.iterations))
        write_json(out / 'config_diff_from_new_v1.json', dict(
            env=differences(class_to_dict(baseline), class_to_dict(cfg)),
            training=differences(class_to_dict(baseline_train), class_to_dict(train))))
        env, _ = task_registry.make_env(TASK, args=args, env_cfg=cfg)
        assert env.device == 'cuda:0' and env.num_envs == 4096
        runner = HIMOnPolicyRunner(env, class_to_dict(train), str(out), device=args.rl_device)
        load_new_v1_weights(runner)
        env.take_diagnostics()  # Exclude constructor/reset warm-up.
        metadata = dict(task=TASK, source_checkpoint=str(SOURCE), source_sha256=SOURCE_SHA256,
            initialization='Full actor/critic/estimator weights; fresh PPO and estimator optimizers',
            additional_iterations=opts.iterations, save_interval=50,
            checkpoint_iterations=sorted(set(list(range(50, opts.iterations + 1, 50)) + [opts.iterations])),
            gpu=opts.gpu, pid=os.getpid(), seed=train.seed,
            feet=[env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])[i] for i in env.feet_indices.tolist()],
            metrics='Raw training policy; 50 Hz snapshots; falls per exposure, not trial failure probability')
        sources = [Path(__file__).resolve(), ROOT / 'tools/train_lz_q1l_v1_support_500.py']
        sources += [ROOT / p for p in (
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py',
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py',
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
            'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
            'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py',
            'HIMLoco/legged_gym/legged_gym/envs/__init__.py',
            'HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
            'HIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py',
            'HIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py',
            'HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py',
            'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf')]
        metadata['code_sha256'] = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            target = out / 'provenance' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            metadata['code_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        for name, value in [('run_metadata', metadata), ('env_config', class_to_dict(cfg)), ('train_config', class_to_dict(train))]:
            write_json(out / (name + '.json'), value)
        install_logging(runner, out, metadata, opts.iterations)
        runner.learn(num_learning_iterations=opts.iterations, init_at_random_ep_len=True)
        assert runner.current_learning_iteration == opts.iterations
        for iteration in metadata['checkpoint_iterations']:
            state = torch.load(str(out / ('model_%d.pt' % iteration)), map_location='cpu')
            assert state['iter'] == iteration
            assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        write_json(out / 'status.json', dict(state='completed', completed_updates=opts.iterations))
    except BaseException as exc:
        status = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {}
        status.update(state='failed', error=repr(exc))
        write_json(out / 'status.json', status)
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
