"""Wait for the original V7 run, then resume both optimizers to update 500."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def restore_config(obj, values):
    for key, value in values.items():
        if key == 'init_member_classes':
            continue
        if isinstance(value, dict):
            current = getattr(obj, key)
            if isinstance(current, dict):
                setattr(obj, key, value.copy())
            else:
                restore_config(current, value)
        else:
            setattr(obj, key, value)


def train(run, out, source_step, target, smoke=False):
    import isaacgym  # noqa: F401; must precede torch
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from rsl_rl.runners import HIMOnPolicyRunner

    sys.argv = [sys.argv[0], '--task', 'lz_q1l_flat_v7', '--sim_device', 'cuda:0',
                '--rl_device', 'cuda:0', '--headless']
    args = get_args()
    cfg, _ = task_registry.get_cfgs(args.task)
    restore_config(cfg, json.loads((run / 'env_config.json').read_text()))
    if smoke:
        cfg.env.num_envs = 64
    train_cfg = json.loads((run / 'train_config.json').read_text())
    train_cfg['runner'].update(max_iterations=target, resume=False, save_interval=50)
    source = run / ('model_%d.pt' % source_step)
    checkpoint = torch.load(str(source), map_location='cpu')
    assert checkpoint['iter'] == source_step
    for key in ['model_state_dict', 'optimizer_state_dict', 'estimator_optimizer_state_dict']:
        assert checkpoint[key], 'Missing checkpoint component: ' + key
    if any(out.glob('model_*.pt')) and out != run:
        raise RuntimeError('Smoke output already contains checkpoints')
    for step in range(source_step + 1, target + 1):
        if (out / ('model_%d.pt' % step)).exists():
            raise RuntimeError('Refusing to overwrite an existing checkpoint')
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner = HIMOnPolicyRunner(env, train_cfg, str(out), device=args.rl_device)
    runner.load(str(source), load_optimizer=True)
    # load() restores Adam groups but not the separate adaptive-LR scalar.
    runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']
    estimator = runner.alg.actor_critic.estimator
    estimator.learning_rate = estimator.optimizer.param_groups[0]['lr']
    assert runner.current_learning_iteration == source_step
    expected = list(range((source_step // 50 + 1) * 50, target + 1, 50))
    if target not in expected:
        expected.append(target)
    metadata = dict(source_checkpoint=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    start_update=source_step, target_updates=target, additional_updates=target-source_step,
                    save_interval=50, expected_checkpoints=expected, load_optimizer=True,
                    load_estimator_optimizer=True, learning_rate=runner.alg.learning_rate,
                    simulation_state_reinitialized=True, smoke=smoke,
                    pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write_json(out / 'extension_metadata.json', metadata)
    write_json(out / 'extension_train_config.json', train_cfg)
    print('RESUME_VERIFIED ' + json.dumps(metadata), flush=True)
    original_log, original_save = runner.log, runner.save

    def log_and_save(locs):
        original_log(locs)
        completed = locs['it'] + 1
        if completed in expected:
            path = out / ('model_%d.pt' % completed)
            assert not path.exists()
            old = runner.current_learning_iteration
            runner.current_learning_iteration = completed
            try:
                original_save(str(path) + '.tmp', infos=metadata)
                os.replace(str(path) + '.tmp', path)
            finally:
                runner.current_learning_iteration = old
            print('CHECKPOINT_SAVED ' + str(path), flush=True)
        state = dict(state='running', completed_updates=completed, target_updates=target)
        write_json(out / 'status.json', state)
        write_json(out / 'extension_status.json', state)

    runner.log = log_and_save
    runner.save = lambda *a, **kw: None
    runner.learn(num_learning_iterations=target-source_step, init_at_random_ep_len=True)
    assert runner.current_learning_iteration == target
    for step in expected:
        state = torch.load(str(out / ('model_%d.pt' % step)), map_location='cpu')
        assert state['iter'] == step
    runner.writer.flush()
    runner.writer.close()
    state = dict(state='completed', completed_updates=target, target_updates=target)
    write_json(out / 'status.json', state)
    write_json(out / 'extension_status.json', state)
    print('TRAINING_COMPLETE %d updates' % target, flush=True)
    env.gym.destroy_sim(env.sim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke-dir', type=Path)
    opts = parser.parse_args()
    run = opts.run_dir.resolve()
    out = opts.smoke_dir.resolve() if opts.smoke_dir else run
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'extension.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if opts.smoke_dir:
                train(run, out, 100, 101, smoke=True)
                return
            original = json.loads((run / 'run_metadata.json').read_text())
            record = run.parents[1] / 'v7_run_records' / run.name.split('_', 2)[2]
            write_json(out / 'extension_status.json', dict(state='waiting_for_original_200',
                       target_updates=500, pid=os.getpid(), original_pid=original['pid']))
            print('WAITING: original run will finish 200; queued continuation 200 -> 500', flush=True)
            while True:
                try:
                    os.kill(original['pid'], 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                if not alive:
                    status = json.loads((run / 'status.json').read_text())
                    if status.get('state') != 'completed' or status.get('completed_updates') != 200:
                        raise RuntimeError('Original training exited before successful update 200')
                    if (record / 'completed_model_path.txt').exists():
                        assert (record / 'exit_code.txt').read_text().strip() == '0'
                        break
                    # Original shell may still be copying provenance after Python exits.
                    if (record / 'exit_code.txt').exists() and (record / 'exit_code.txt').read_text().strip() != '0':
                        raise RuntimeError('Original launcher failed')
                time.sleep(15)
            train(run, out, 200, 500)
        except BaseException as exc:
            write_json(out / 'extension_status.json', dict(state='failed', error=repr(exc), target_updates=101 if opts.smoke_dir else 500))
            raise


if __name__ == '__main__':
    main()
